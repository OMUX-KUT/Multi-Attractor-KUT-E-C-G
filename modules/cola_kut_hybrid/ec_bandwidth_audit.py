"""Integrated Layer-1/Layer-2 auditor combining Real PatchTextVAE, C++17 L3 bench, and Lean 4."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from modules.cola_kut_hybrid.real_text_vae_bench import (
    RealVAEResult,
    evaluate_real_reconstruction,
)

L3_CACHE_LIMIT_BYTES = 3_840_000
MIN_THROUGHPUT_TPS = 500.0
ESTIMATED_2B_BACKBONE_FP16_BYTES = 4_000_000_000
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


@dataclass(frozen=True)
class UnifiedAuditSummary:
    lean4_verified: Optional[bool]
    lean4_axioms_detected: List[str]
    layer1_l3_limit_bytes: int
    layer1_min_tps: float
    layer2_ref_2b_backbone_bytes: int
    real_vae_results: List[RealVAEResult]


def run_lean4_strict_check(lean_file: Path) -> Tuple[Optional[bool], List[str]]:
    if not lean_file.exists():
        return None, []
    proc = subprocess.run(["lean", str(lean_file)], capture_output=True, text=True, check=False)
    out = proc.stdout + proc.stderr
    found = set()
    for line in out.splitlines():
        m = re.search(r"depends on axioms:\s*\[([^\]]+)\]", line)
        if m:
            for ax in m.group(1).split(","):
                ax = ax.strip()
                if ax:
                    found.add(ax)
    extra = found - ALLOWED_AXIOMS
    ok = (
        proc.returncode == 0
        and "warning:" not in out.lower()
        and "error:" not in out.lower()
        and "sorry" not in out.lower()
        and len(extra) == 0
    )
    return ok, sorted(found)


def main() -> None:
    samples = [
        "def kut_ricci_step(x: float, dt: float) -> float:\n    return x - dt * (x * x)\n",
        "def energy_compute_ratio(c: float, e: float) -> float:\n    return c / max(e, 1e-9)\n",
        "def mod0_pad(length: int, patch: int) -> int:\n    return (patch - (length % patch)) % patch\n",
        "def verify_gate(l3_bytes: int, tps: float) -> bool:\n    return (l3_bytes <= 3840000) and (tps >= 500.0)\n",
    ]
    lean_ok, axioms = run_lean4_strict_check(Path("lean4_verifier/kut_cola_bridge.lean"))
    vae_results = evaluate_real_reconstruction(samples, noise_ratios=(0.0, 0.10, 0.25))

    summary = UnifiedAuditSummary(
        lean4_verified=lean_ok,
        lean4_axioms_detected=axioms,
        layer1_l3_limit_bytes=L3_CACHE_LIMIT_BYTES,
        layer1_min_tps=MIN_THROUGHPUT_TPS,
        layer2_ref_2b_backbone_bytes=ESTIMATED_2B_BACKBONE_FP16_BYTES,
        real_vae_results=vae_results,
    )

    out_path = Path("logs/cola_kut_hybrid_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"LEAN4_VERIFIED={lean_ok} | AXIOMS={axioms}")
    for r in vae_results:
        print(
            f"[noise={r.noise_ratio:4.2f}][{r.condition:8s}] "
            f"tok_acc={r.token_accuracy:8.4f} | AST_err={r.ast_corruption_rate:6.1%} | "
            f"L1_Head={r.head_working_set_bytes:,}B | L2_VAE={r.vae_backbone_param_bytes:,}B"
        )


if __name__ == "__main__":
    main()
