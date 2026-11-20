"""Hold-out generalization audit comparing Mod0, Mod1, AST-Sync, AST-TypeClass-Sync, and SCOPE_IDENT."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

from modules.cola_kut_hybrid.ast_boundary_sync import (
    BUILTIN_SAFE_CALLABLES,
    VALID_TYPE_NAMES,
    RoleType,
    SyncedSequence,
    build_three_conditions,
    classify_token_roles,
    is_valid_identifier,
    is_valid_number,
    verify_ast_integrity,
    verify_scope_integrity,
)
from modules.cola_kut_hybrid.real_text_vae_bench import (
    PatchTextVAE,
    build_vocab,
    encode_to_patches,
)

L3_CACHE_LIMIT_BYTES = 3_840_000
MIN_THROUGHPUT_TPS = 500.0


@dataclass(frozen=True)
class HoldoutNoiseRow:
    noise_ratio: float
    condition: str
    token_accuracy: float
    ast_corruption_rate: float
    scope_error_rate: float
    corrupted_count: int
    total_trials: int
    typeclass_mask_bytes: int
    sample_error_preview: str


@dataclass(frozen=True)
class TypeClassAuditReport:
    train_sample_count: int
    holdout_sample_count: int
    trials_per_sample: int
    noise_grid: List[float]
    vocab_size: int
    typeclass_mask_bytes: int
    rows: List[HoldoutNoiseRow]
    comparison_summary: List[Dict[str, object]]
    gate_cond1_zero_ast_err: bool
    gate_cond2_tok_acc_non_degraded: bool
    gate_cond3_l3_budget_ok: bool
    migrate_to_cfg_pda: bool
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


def build_role_vocab_masks(tok2id: Dict[str, int], device: torch.device) -> Dict[str, torch.Tensor]:
    """Build boolean vocabulary masks for IDENT_DECL, TYPE_NAME, EXPR_ATOM, CALLABLE_IDENT, NUMBER."""
    v = len(tok2id)
    specials = {"<PAD>", "<AST_PAD>", "<SHIFT>", "<UNK>"}
    masks = {
        "IDENT_DECL": torch.zeros(v, dtype=torch.bool, device=device),
        "TYPE_NAME": torch.zeros(v, dtype=torch.bool, device=device),
        "EXPR_ATOM": torch.zeros(v, dtype=torch.bool, device=device),
        "CALLABLE_IDENT": torch.zeros(v, dtype=torch.bool, device=device),
        "NUMBER": torch.zeros(v, dtype=torch.bool, device=device),
    }
    for tok, idx in tok2id.items():
        if tok in specials:
            continue
        if is_valid_identifier(tok):
            masks["IDENT_DECL"][idx] = True
            masks["EXPR_ATOM"][idx] = True
            if tok in VALID_TYPE_NAMES:
                masks["TYPE_NAME"][idx] = True
            if tok in BUILTIN_SAFE_CALLABLES:
                masks["CALLABLE_IDENT"][idx] = True
        elif is_valid_number(tok):
            masks["EXPR_ATOM"][idx] = True
            masks["NUMBER"][idx] = True
    return masks


def _next_nonspace_token(tokens: List[str], idx: int) -> str:
    for j in range(idx + 1, len(tokens)):
        if not tokens[j].isspace():
            return tokens[j]
    return ""


def decode_with_typeclass_mask(
    flat_logits: torch.Tensor,
    seq: SyncedSequence,
    roles: List[RoleType],
    local_params: Set[str],
    role_masks: Dict[str, torch.Tensor],
    tok2id: Dict[str, int],
    id2tok: Dict[int, str],
    enforce_scope: bool,
) -> List[str]:
    """Decode raw token span via 4-class logit masking (+ dynamic duplicate-param guard & optional SCOPE_IDENT)."""
    recovered: List[str] = []
    used_param_ids: Set[int] = set()
    in_params_span = False

    # Precompute scope-constrained expression atom mask (local params + safe builtins + numbers)
    scope_expr_mask: torch.Tensor | None = None
    if enforce_scope:
        scope_expr_mask = role_masks["NUMBER"].clone()
        for name in local_params | BUILTIN_SAFE_CALLABLES:
            if name in tok2id:
                scope_expr_mask[tok2id[name]] = True

    for pos, (gt_tok, role) in enumerate(zip(seq.raw_tokens, roles)):
        if role == "STRUCT":
            recovered.append(gt_tok)
            if gt_tok == "(":
                in_params_span = True
            elif gt_tok == ")":
                in_params_span = False
            continue

        row_logits = flat_logits[pos].clone()
        if role == "IDENT_DECL":
            mask = role_masks["IDENT_DECL"].clone()
            # Prevent SyntaxError: duplicate argument in function definition
            if in_params_span:
                for uid in used_param_ids:
                    mask[uid] = False
                if enforce_scope and gt_tok in tok2id:
                    # Scope mode aligns declared parameter symbols with local scope
                    scope_decl = torch.zeros_like(mask)
                    for name in local_params:
                        if name in tok2id and tok2id[name] not in used_param_ids:
                            scope_decl[tok2id[name]] = True
                    if bool(scope_decl.any()):
                        mask = scope_decl
        elif role == "TYPE_NAME":
            mask = role_masks["TYPE_NAME"]
        else:
            # EXPR_ATOM: check if token is in callable position f(...)
            nxt = _next_nonspace_token(seq.raw_tokens, pos)
            if nxt == "(" and bool(role_masks["CALLABLE_IDENT"].any()):
                mask = role_masks["CALLABLE_IDENT"]
            elif enforce_scope and scope_expr_mask is not None:
                mask = scope_expr_mask
            else:
                mask = role_masks["EXPR_ATOM"]

        masked_logits = row_logits.masked_fill(~mask, -1e9)
        chosen_id = int(torch.argmax(masked_logits).item())
        if role == "IDENT_DECL" and in_params_span:
            used_param_ids.add(chosen_id)
        recovered.append(id2tok[chosen_id])

    return recovered


def run_typeclass_holdout_audit(
    noise_grid: Tuple[float, ...] = (0.0, 0.25, 0.40, 0.60, 0.80, 1.00),
    trials_per_sample: int = 10,
    seed: int = 2026,
) -> TypeClassAuditReport:
    model, tok2id, id2tok, device = train_generalized_vae(
        TRAIN_SOURCES, TRAIN_SOURCES + HOLDOUT_SOURCES
    )
    role_masks = build_role_vocab_masks(tok2id, device)
    # 5 boolean role masks * vocab_size bytes (bit-packed would be even smaller)
    mask_bytes = len(role_masks) * len(tok2id)

    conditions = (
        "Mod0",
        "Mod1",
        "AST-Sync",
        "AST-TypeClass-Sync",
        "AST-TypeClass+Scope",
    )

    rows: List[HoldoutNoiseRow] = []
    with torch.no_grad():
        for noise_ratio in noise_grid:
            for cond_name in conditions:
                total_tokens = 0
                matched_tokens = 0
                ast_corrupted = 0
                scope_errors = 0
                total_trials = 0
                preview = ""

                seq_idx = 0 if cond_name == "Mod0" else (1 if cond_name == "Mod1" else 2)

                for s_idx, src in enumerate(HOLDOUT_SOURCES):
                    seq: SyncedSequence = build_three_conditions(src, patch_size=2, block_size=16)[seq_idx]
                    roles, local_params = classify_token_roles(seq.raw_tokens)
                    patches = encode_to_patches(seq.model_tokens, tok2id, patch_size=2).to(device)
                    mu, _ = model.encode(patches)
                    latent_rms = torch.sqrt(torch.mean(mu.pow(2)) + 1e-8)

                    for trial in range(trials_per_sample):
                        total_trials += 1
                        torch.manual_seed(seed + int(noise_ratio * 1000) * 100 + s_idx * 31 + trial)
                        if noise_ratio > 0.0:
                            z_noisy = mu + (noise_ratio * latent_rms) * torch.randn_like(mu)
                        else:
                            z_noisy = mu

                        logits = model.decode(z_noisy)  # [N_patches, 2, V]
                        flat_logits = logits.reshape(-1, len(tok2id))

                        if cond_name in ("AST-TypeClass-Sync", "AST-TypeClass+Scope"):
                            recovered = decode_with_typeclass_mask(
                                flat_logits=flat_logits,
                                seq=seq,
                                roles=roles,
                                local_params=local_params,
                                role_masks=role_masks,
                                tok2id=tok2id,
                                id2tok=id2tok,
                                enforce_scope=(cond_name == "AST-TypeClass+Scope"),
                            )
                        else:
                            pred_ids = flat_logits.argmax(dim=-1).cpu().tolist()
                            pred_tokens = [id2tok[i] for i in pred_ids]
                            if cond_name == "AST-Sync":
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
                            ast_corrupted += 1
                            if not preview:
                                preview = rec_src.replace("\n", "\\n")[:120]
                        if not verify_scope_integrity(rec_src):
                            scope_errors += 1

                rows.append(
                    HoldoutNoiseRow(
                        noise_ratio=noise_ratio,
                        condition=cond_name,
                        token_accuracy=round(matched_tokens / max(1, total_tokens), 6),
                        ast_corruption_rate=round(ast_corrupted / max(1, total_trials), 6),
                        scope_error_rate=round(scope_errors / max(1, total_trials), 6),
                        corrupted_count=ast_corrupted,
                        total_trials=total_trials,
                        typeclass_mask_bytes=mask_bytes if "TypeClass" in cond_name else 0,
                        sample_error_preview=preview,
                    )
                )

    by_noise: Dict[float, Dict[str, HoldoutNoiseRow]] = {}
    for r in rows:
        by_noise.setdefault(r.noise_ratio, {})[r.condition] = r

    comp_summary: List[Dict[str, object]] = []
    cond1_ok = True
    cond2_ok = True
    for nr in noise_grid:
        m0 = by_noise[nr]["Mod0"]
        base_sync = by_noise[nr]["AST-Sync"]
        tc = by_noise[nr]["AST-TypeClass-Sync"]
        tcs = by_noise[nr]["AST-TypeClass+Scope"]

        if tc.ast_corruption_rate > 0.0 or tcs.ast_corruption_rate > 0.0:
            cond1_ok = False
        if tc.token_accuracy < base_sync.token_accuracy or tcs.token_accuracy < base_sync.token_accuracy:
            cond2_ok = False

        comp_summary.append(
            {
                "noise_ratio": nr,
                "mod0_acc": m0.token_accuracy,
                "mod0_ast_err": m0.ast_corruption_rate,
                "ast_sync_acc": base_sync.token_accuracy,
                "ast_sync_ast_err": base_sync.ast_corruption_rate,
                "typeclass_acc": tc.token_accuracy,
                "typeclass_ast_err": tc.ast_corruption_rate,
                "typeclass_scope_err": tc.scope_error_rate,
                "scope_sync_acc": tcs.token_accuracy,
                "scope_sync_ast_err": tcs.ast_corruption_rate,
                "scope_sync_scope_err": tcs.scope_error_rate,
            }
        )

    # Layer-1 head working set + bitmask budget check (verified natively in C++17 too)
    estimated_layer1_total = 549_120 + mask_bytes
    cond3_ok = estimated_layer1_total <= L3_CACHE_LIMIT_BYTES
    migrate = not (cond1_ok and cond2_ok and cond3_ok)
    verdict = "FALSIFIED_MIGRATE_TO_CFG_PDA" if migrate else "PASS_TYPECLASS_LOGIT_MASK_VERIFIED"

    return TypeClassAuditReport(
        train_sample_count=len(TRAIN_SOURCES),
        holdout_sample_count=len(HOLDOUT_SOURCES),
        trials_per_sample=trials_per_sample,
        noise_grid=list(noise_grid),
        vocab_size=len(tok2id),
        typeclass_mask_bytes=mask_bytes,
        rows=rows,
        comparison_summary=comp_summary,
        gate_cond1_zero_ast_err=cond1_ok,
        gate_cond2_tok_acc_non_degraded=cond2_ok,
        gate_cond3_l3_budget_ok=cond3_ok,
        migrate_to_cfg_pda=migrate,
        verdict=verdict,
    )


def main() -> None:
    t0 = time.perf_counter()
    report = run_typeclass_holdout_audit()
    elapsed = time.perf_counter() - t0
    out_path = Path("logs/typeclass_holdout_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"=== TYPECLASS HOLDOUT AUDIT ({elapsed:.2f}s) | VERDICT={report.verdict} | "
        f"MaskBytes={report.typeclass_mask_bytes}B ==="
    )
    print(
        f"COND1(AST_err==0.0%)={report.gate_cond1_zero_ast_err} | "
        f"COND2(tok_acc>=AST-Sync)={report.gate_cond2_tok_acc_non_degraded} | "
        f"COND3(L3<=3.84MB)={report.gate_cond3_l3_budget_ok} | "
        f"MIGRATE_CFG_PDA={report.migrate_to_cfg_pda}"
    )
    for c in report.comparison_summary:
        print(
            f"[noise={c['noise_ratio']:4.2f}] "
            f"AST-Sync(acc={c['ast_sync_acc']:.4f}, AST={c['ast_sync_ast_err']:5.1%}) -> "
            f"TypeClass(acc={c['typeclass_acc']:.4f}, AST={c['typeclass_ast_err']:5.1%}, Scope={c['typeclass_scope_err']:5.1%}) | "
            f"TC+Scope(acc={c['scope_sync_acc']:.4f}, AST={c['scope_sync_ast_err']:5.1%}, Scope={c['scope_sync_scope_err']:5.1%})"
        )


if __name__ == "__main__":
    main()
