"""Hold-out generalization & high-noise separation benchmark (Mod0 vs Mod1 vs AST-Sync)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from modules.cola_kut_hybrid.ast_boundary_sync import (
    SyncedSequence,
    build_three_conditions,
    verify_ast_integrity,
)
from modules.cola_kut_hybrid.real_text_vae_bench import (
    PatchTextVAE,
    build_vocab,
    encode_to_patches,
)


@dataclass(frozen=True)
class HoldoutNoiseRow:
    noise_ratio: float
    condition: str
    token_accuracy: float
    ast_corruption_rate: float
    corrupted_count: int
    total_trials: int
    sample_error_preview: str


@dataclass(frozen=True)
class HoldoutAuditReport:
    train_sample_count: int
    holdout_sample_count: int
    trials_per_sample: int
    noise_grid: List[float]
    rows: List[HoldoutNoiseRow]
    target_band_summary: List[Dict[str, object]]
    needs_typeclass_decoder: bool
    verdict: str


TRAIN_SOURCES: List[str] = [
    "def kut_ricci_step(x: float, dt: float) -> float:\n    return x - dt * (x * x)\n",
    "def energy_compute_ratio(c: float, e: float) -> float:\n    return c / max(e, 1e-9)\n",
    "def mod0_pad(length: int, patch: int) -> int:\n    return (patch - (length % patch)) % patch\n",
    "def verify_gate(l3_bytes: int, tps: float) -> bool:\n    return (l3_bytes <= 3840000) and (tps >= 500.0)\n",
    "def clamp_unit(v: float) -> float:\n    return min(1.0, max(0.0, v))\n",
    "def linear_blend(a: float, b: float, w: float) -> float:\n    return a * (1.0 - w) + b * w\n",
    "def block_count(n: int, bs: int) -> int:\n    return (n + bs - 1) // max(bs, 1)\n",
    "def residual_norm(a: float, b: float) -> float:\n    return abs(a - b)\n",
    "def safe_div(num: float, den: float) -> float:\n    return num / (den + 1e-9)\n",
    "def is_aligned(n: int, p: int) -> bool:\n    return (n % p) == 0\n",
    "def damp_step(v: float, rate: float) -> float:\n    return v * (1.0 - rate)\n",
    "def clip_margin(x: float, lim: float) -> float:\n    return min(lim, max(-lim, x))\n",
    "def scale_energy(c: float, k: float) -> float:\n    return c * k + 0.0\n",
    "def check_positive(x: float) -> bool:\n    return x > 0.0\n",
    "def pair_sum(a: int, b: int) -> int:\n    return a + b\n",
    "def pair_diff(a: int, b: int) -> int:\n    return a - b\n",
]

# Unseen expression structures composed from shared lexical vocabulary (strict Hold-out)
HOLDOUT_SOURCES: List[str] = [
    "def holdout_ricci_mix(a: float, b: float, dt: float) -> float:\n    return (a - dt * b) + abs(a - b)\n",
    "def holdout_gate_ratio(c: float, e: float, l3_bytes: int) -> bool:\n    return (c / max(e, 1e-9) >= 1.0) and (l3_bytes <= 3840000)\n",
    "def holdout_pad_blocks(length: int, patch: int, bs: int) -> int:\n    return ((length + (patch - (length % patch)) % patch) + bs - 1) // max(bs, 1)\n",
    "def holdout_clamp_blend(x: float, v: float, w: float) -> float:\n    return min(1.0, max(0.0, x * (1.0 - w) + v * w))\n",
    "def holdout_damp_div(num: float, den: float, rate: float) -> float:\n    return (num / (den + 1e-9)) * (1.0 - rate)\n",
    "def holdout_align_check(n: int, p: int, tps: float) -> bool:\n    return ((n % p) == 0) and (tps >= 500.0)\n",
    "def holdout_margin_energy(c: float, k: float, lim: float) -> float:\n    return min(lim, max(-lim, c * k))\n",
    "def holdout_pair_balance(a: int, b: int) -> bool:\n    return (a + b) >= (a - b)\n",
]


def train_generalized_vae(
    train_sources: List[str],
    all_vocab_sources: List[str],
    epochs: int = 220,
    seed: int = 42,
) -> Tuple[PatchTextVAE, Dict[str, int], Dict[int, str], torch.device]:
    """Train PatchTextVAE strictly on TRAIN_SOURCES with moderate KL regularization."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok2id, id2tok = build_vocab(all_vocab_sources)

    train_patches: List[torch.Tensor] = []
    for src in train_sources:
        mod0_seq = build_three_conditions(src, patch_size=2, block_size=16)[0]
        train_patches.append(encode_to_patches(mod0_seq.model_tokens, tok2id, patch_size=2))
    batch = torch.cat(train_patches, dim=0).to(device)

    model = PatchTextVAE(vocab_size=len(tok2id), embed_dim=64, latent_dim=110, patch_size=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)

    model.train()
    for _ in range(epochs):
        mu, logvar = model.encode(batch)
        std = torch.exp(0.5 * logvar)
        z = mu + 0.15 * std * torch.randn_like(std)
        logits = model.decode(z)
        recon_loss = F.cross_entropy(logits.reshape(-1, len(tok2id)), batch.reshape(-1))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 5e-3 * kl_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model, tok2id, id2tok, device


def run_holdout_sweep(
    noise_grid: Tuple[float, ...] = (0.0, 0.25, 0.40, 0.60, 0.80, 1.00),
    trials_per_sample: int = 10,
    seed: int = 2026,
) -> HoldoutAuditReport:
    model, tok2id, id2tok, device = train_generalized_vae(
        TRAIN_SOURCES, TRAIN_SOURCES + HOLDOUT_SOURCES
    )

    rows: List[HoldoutNoiseRow] = []
    with torch.no_grad():
        for noise_ratio in noise_grid:
            for cond_idx, cond_name in enumerate(("Mod0", "Mod1", "AST-Sync")):
                total_tokens = 0
                matched_tokens = 0
                corrupted = 0
                total_trials = 0
                preview = ""

                for s_idx, src in enumerate(HOLDOUT_SOURCES):
                    seq: SyncedSequence = build_three_conditions(src, patch_size=2, block_size=16)[cond_idx]
                    patches = encode_to_patches(seq.model_tokens, tok2id, patch_size=2).to(device)
                    mu, _ = model.encode(patches)
                    latent_rms = torch.sqrt(torch.mean(mu.pow(2)) + 1e-8)

                    for trial in range(trials_per_sample):
                        total_trials += 1
                        # Identical noise seed across Mod0 / AST-Sync for paired comparison
                        torch.manual_seed(seed + int(noise_ratio * 1000) * 100 + s_idx * 31 + trial)
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
                        recovered = pred_tokens[start:end]

                        for gt_t, pr_t in zip(seq.raw_tokens, recovered):
                            total_tokens += 1
                            if gt_t == pr_t:
                                matched_tokens += 1

                        rec_src = "".join(
                            t for t in recovered if t not in ("<PAD>", "<AST_PAD>", "<SHIFT>", "<UNK>")
                        )
                        if not verify_ast_integrity(rec_src):
                            corrupted += 1
                            if not preview:
                                preview = rec_src.replace("\n", "\\n")[:120]

                rows.append(
                    HoldoutNoiseRow(
                        noise_ratio=noise_ratio,
                        condition=cond_name,
                        token_accuracy=round(matched_tokens / max(1, total_tokens), 6),
                        ast_corruption_rate=round(corrupted / max(1, total_trials), 6),
                        corrupted_count=corrupted,
                        total_trials=total_trials,
                        sample_error_preview=preview,
                    )
                )

    # Evaluate falsification criterion across degraded noise points (Mod0 tok_acc < 0.98)
    by_noise: Dict[float, Dict[str, HoldoutNoiseRow]] = {}
    for r in rows:
        by_noise.setdefault(r.noise_ratio, {})[r.condition] = r

    target_summary: List[Dict[str, object]] = []
    needs_redesign = False
    for nr in noise_grid:
        m0 = by_noise[nr]["Mod0"]
        as_row = by_noise[nr]["AST-Sync"]
        delta_err = round(m0.ast_corruption_rate - as_row.ast_corruption_rate, 6)
        in_target_band = (0.75 <= m0.token_accuracy <= 0.96)
        if in_target_band:
            # Falsification trigger: AST-Sync does not reduce error significantly or exceeds 5%
            if as_row.ast_corruption_rate > 0.05 or (
                m0.ast_corruption_rate > 0.05 and delta_err < 0.10
            ):
                needs_redesign = True
        target_summary.append(
            {
                "noise_ratio": nr,
                "mod0_tok_acc": m0.token_accuracy,
                "mod0_ast_err": m0.ast_corruption_rate,
                "ast_sync_tok_acc": as_row.token_accuracy,
                "ast_sync_ast_err": as_row.ast_corruption_rate,
                "ast_err_reduction": delta_err,
                "in_target_degradation_band": in_target_band,
            }
        )

    verdict = (
        "FALSIFIED_REDESIGN_TYPECLASS_DECODER"
        if needs_redesign
        else "SUPPORTED_STRUCTURAL_MASK_EFFECTIVE"
    )
    return HoldoutAuditReport(
        train_sample_count=len(TRAIN_SOURCES),
        holdout_sample_count=len(HOLDOUT_SOURCES),
        trials_per_sample=trials_per_sample,
        noise_grid=list(noise_grid),
        rows=rows,
        target_band_summary=target_summary,
        needs_typeclass_decoder=needs_redesign,
        verdict=verdict,
    )


def main() -> None:
    t0 = time.perf_counter()
    report = run_holdout_sweep()
    elapsed = time.perf_counter() - t0
    out_path = Path("logs/holdout_high_noise_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"=== HOLDOUT HIGH-NOISE AUDIT ({elapsed:.2f}s) | VERDICT={report.verdict} ===")
    for item in report.target_band_summary:
        tag = "*" if item["in_target_degradation_band"] else " "
        print(
            f"{tag}[noise={item['noise_ratio']:4.2f}] "
            f"Mod0(acc={item['mod0_tok_acc']:.4f}, AST_err={item['mod0_ast_err']:6.1%}) | "
            f"AST-Sync(acc={item['ast_sync_tok_acc']:.4f}, AST_err={item['ast_sync_ast_err']:6.1%}) | "
            f"Delta={item['ast_err_reduction']:+6.1%}"
        )


if __name__ == "__main__":
    main()
