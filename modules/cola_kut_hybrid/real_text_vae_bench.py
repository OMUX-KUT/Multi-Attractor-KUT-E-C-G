"""Real PyTorch PatchTextVAE (p2, d=110) reconstruction benchmark without mock error injection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.cola_kut_hybrid.ast_boundary_sync import (
    SyncedSequence,
    build_three_conditions,
    reversible_tokenize,
    verify_ast_integrity,
)


@dataclass(frozen=True)
class RealVAEResult:
    condition: str
    noise_ratio: float
    token_accuracy: float
    ast_corrupted: bool
    ast_corruption_rate: float
    samples_evaluated: int
    vae_backbone_param_bytes: int
    head_working_set_bytes: int
    eval_tps: float


class PatchTextVAE(nn.Module):
    """Compact PatchTextVAE compressing patch_size=2 tokens into a d=110 continuous latent."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        latent_dim: int = 110,
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.enc = nn.Sequential(
            nn.Linear(embed_dim * patch_size, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, vocab_size * patch_size),
        )

    def encode(self, patch_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.emb(patch_ids).reshape(patch_ids.shape[0], -1)
        h = self.enc(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.dec(z)
        return logits.view(z.shape[0], self.patch_size, self.vocab_size)


def build_vocab(sources: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    specials = ["<PAD>", "<AST_PAD>", "<SHIFT>", "<UNK>"]
    tok2id: Dict[str, int] = {tok: i for i, tok in enumerate(specials)}
    for src in sources:
        for tok in reversible_tokenize(src):
            if tok not in tok2id:
                tok2id[tok] = len(tok2id)
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok


def encode_to_patches(tokens: List[str], tok2id: Dict[str, int], patch_size: int = 2) -> torch.Tensor:
    ids = [tok2id.get(t, tok2id["<UNK>"]) for t in tokens]
    if len(ids) % patch_size != 0:
        ids.append(tok2id["<PAD>"])
    return torch.tensor(ids, dtype=torch.long).view(-1, patch_size)


def train_patch_text_vae(
    sources: List[str],
    epochs: int = 350,
    seed: int = 42,
) -> Tuple[PatchTextVAE, Dict[str, int], Dict[int, str], torch.device]:
    """Train PatchTextVAE on Mod0 aligned pairs (reproducing Cola DLM Stage 1 Text VAE)."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok2id, id2tok = build_vocab(sources)

    train_patches: List[torch.Tensor] = []
    for src in sources:
        mod0_seq = build_three_conditions(src, patch_size=2, block_size=16)[0]
        train_patches.append(encode_to_patches(mod0_seq.model_tokens, tok2id, patch_size=2))
    batch = torch.cat(train_patches, dim=0).to(device)

    model = PatchTextVAE(vocab_size=len(tok2id), embed_dim=64, latent_dim=110, patch_size=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        mu, logvar = model.encode(batch)
        std = torch.exp(0.5 * logvar)
        z = mu + 0.05 * std * torch.randn_like(std)
        logits = model.decode(z)
        recon_loss = F.cross_entropy(logits.reshape(-1, len(tok2id)), batch.reshape(-1))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 1e-4 * kl_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model, tok2id, id2tok, device


def evaluate_real_reconstruction(
    sources: List[str],
    noise_ratios: Tuple[float, ...] = (0.0, 0.10, 0.25),
    seed: int = 123,
) -> List[RealVAEResult]:
    """Train PatchTextVAE with gradients enabled, then evaluate under torch.no_grad()."""
    model, tok2id, id2tok, device = train_patch_text_vae(sources)
    vae_param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    results: List[RealVAEResult] = []
    with torch.no_grad():
        for noise_ratio in noise_ratios:
            for cond_idx, cond_name in enumerate(("Mod0", "Mod1", "AST-Sync")):
                torch.manual_seed(seed + int(noise_ratio * 1000) + cond_idx)
                total_tokens = 0
                matched_tokens = 0
                corrupted_samples = 0
                total_latents = 0
                t0 = time.perf_counter()

                for src in sources:
                    seq: SyncedSequence = build_three_conditions(src, patch_size=2, block_size=16)[cond_idx]
                    patches = encode_to_patches(seq.model_tokens, tok2id, patch_size=2).to(device)
                    total_latents += patches.shape[0]

                    mu, _ = model.encode(patches)
                    latent_rms = torch.sqrt(torch.mean(mu.pow(2)) + 1e-8)
                    if noise_ratio > 0.0:
                        z_noisy = mu + (noise_ratio * latent_rms) * torch.randn_like(mu)
                    else:
                        z_noisy = mu

                    logits = model.decode(z_noisy)
                    pred_ids = logits.argmax(dim=-1).reshape(-1).cpu().tolist()
                    pred_tokens = [id2tok[i] for i in pred_ids]

                    if seq.condition == "AST-Sync":
                        pred_tokens = [
                            orig if is_struct else pred
                            for orig, pred, is_struct in zip(
                                seq.model_tokens, pred_tokens, seq.structural_mask
                            )
                        ]

                    start = seq.leading_shift
                    end = start + len(seq.raw_tokens)
                    recovered_tokens = pred_tokens[start:end]

                    for gt_t, pr_t in zip(seq.raw_tokens, recovered_tokens):
                        total_tokens += 1
                        if gt_t == pr_t:
                            matched_tokens += 1

                    reconstructed_src = "".join(
                        t for t in recovered_tokens if t not in ("<PAD>", "<AST_PAD>", "<SHIFT>", "<UNK>")
                    )
                    if not verify_ast_integrity(reconstructed_src):
                        corrupted_samples += 1

                elapsed = max(1e-9, time.perf_counter() - t0)
                avg_latents = max(1, round(total_latents / len(sources)))
                head_ws_bytes = (avg_latents * 110 * 8 + 110 * 110 * 4) * 2
                err_rate = corrupted_samples / len(sources)
                results.append(
                    RealVAEResult(
                        condition=cond_name,
                        noise_ratio=noise_ratio,
                        token_accuracy=round(matched_tokens / max(1, total_tokens), 6),
                        ast_corrupted=(corrupted_samples > 0),
                        ast_corruption_rate=round(err_rate, 6),
                        samples_evaluated=len(sources),
                        vae_backbone_param_bytes=vae_param_bytes,
                        head_working_set_bytes=head_ws_bytes,
                        eval_tps=round(total_tokens / elapsed, 2),
                    )
                )
    return results
