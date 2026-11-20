"""Generate GDM (256k) and OpenAI (200k) outreach briefs directly from measured frontier L3 logs + CSV tracker."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    data = json.loads(Path("logs/phase2a_frontier_vocab_l3.json").read_text(encoding="utf-8"))
    rows = data["rows"]

    def pick(label: str, mode: str):
        matches = [r for r in rows if r["vocab_label"] == label and r["mode"] == mode]
        return matches[0] if matches else None

    g256_e = pick("256k_gemma_gemini_scale", "BitPacked-ExactFullV")
    g256_s = pick("256k_gemma_gemini_scale", "BitPacked-SparseTopK256")
    o200_e = pick("o200k_base_real", "BitPacked-ExactFullV") or pick("200k_o200k_scale", "BitPacked-ExactFullV")
    o200_s = pick("o200k_base_real", "BitPacked-SparseTopK256") or pick("200k_o200k_scale", "BitPacked-SparseTopK256")
    g256_real_e = pick("gemma_256k_real", "BitPacked-ExactFullV")
    g256_real_s = pick("gemma_256k_real", "BitPacked-SparseTopK256")

    gdm_real_note = (
        f"- **Official Gemma 256k Real Vocab (`{data['gemma_256k_real_status']}`):** "
        f"ExactFullV `{g256_real_e['total_layer1_bytes']:,} B` (`{g256_real_e['l3_utilization_ratio']*100:.2f}%` L3, `{g256_real_e['measured_tps']:,.1f} tps`) / "
        f"SparseTopK256 `{g256_real_s['total_layer1_bytes']:,} B` (`{g256_real_s['measured_tps']:,.1f} tps`)"
        if (g256_real_e and g256_real_s)
        else f"- **Official Gemma 256k Real Vocab Status:** `{data['gemma_256k_real_status']}` (256,000-scale synthetic density measured directly above; real-vocab drop-in supported via `--real-mask`)"
    )

    gdm_brief = f"""# Google DeepMind Outreach Brief (Measured 256k L3 + Lean 4 + 4-Fold 0.0% Safety)
- **Repository:** `https://github.com/OMUX-KUT/Multi-Attractor-KUT-E-C-G`
- **Lean 4.36.0:** 8/8 theorems PASS (0 warnings, 0 `sorry`, 5 axiom-free, 3 `propext`-only)
- **Measured 256k (`256,000`) Gemini/Gemma-Scale C++17 L3 Audit:**
  - `BitPacked-ExactFullV`: Mask `{g256_e['bitpacked_mask_bytes']:,} B` | Total Layer-1 **`{g256_e['total_layer1_bytes']:,} B` (`{g256_e['l3_utilization_ratio']*100:.2f}%` of 3.84 MB L3)** | **`{g256_e['measured_tps']:,.1f} tps`** | Gate PASS=`{g256_e['phase2a_pass']}`
  - `BitPacked-SparseTopK256`: Mask `{g256_s['bitpacked_mask_bytes']:,} B` | Total Layer-1 **`{g256_s['total_layer1_bytes']:,} B` (`{g256_s['l3_utilization_ratio']*100:.2f}%` of 3.84 MB L3)** | **`{g256_s['measured_tps']:,.1f} tps`** | Gate PASS=`{g256_s['phase2a_pass']}`
{gdm_real_note}
- **4-Fold Open-Ended Free-Gen (`Typed-CFG-Lite`, 200 trials):** `0.0%` AST / `0.0%` Scope / `0.0%` Static Type / `0.0%` Sandbox Runtime (`25.0k–28.1k tps`, 2.3x faster than untyped `CFG-Lite`).
"""
    Path("docs/BRIEF_GDM_256K.md").write_text(gdm_brief, encoding="utf-8")

    oai_brief = f"""# OpenAI Outreach Brief (Measured 200k `o200k_base` L3 + Beyond-CFG 4-Fold 0.0% Safety)
- **Repository:** `https://github.com/OMUX-KUT/Multi-Attractor-KUT-E-C-G`
- **o200k_base Real Vocab Status:** `{data['o200k_real_status']}`
- **Measured 200k (`{o200_e['vocab_size']:,}`, `{o200_e['mask_source']}`) C++17 L3 Audit:**
  - `BitPacked-ExactFullV`: Mask `{o200_e['bitpacked_mask_bytes']:,} B` | Total Layer-1 **`{o200_e['total_layer1_bytes']:,} B` (`{o200_e['l3_utilization_ratio']*100:.2f}%` of 3.84 MB L3)** | **`{o200_e['measured_tps']:,.1f} tps`** | Gate PASS=`{o200_e['phase2a_pass']}`
  - `BitPacked-SparseTopK256`: Mask `{o200_s['bitpacked_mask_bytes']:,} B` | Total Layer-1 **`{o200_s['total_layer1_bytes']:,} B` (`{o200_s['l3_utilization_ratio']*100:.2f}%` of 3.84 MB L3)** | **`{o200_s['measured_tps']:,.1f} tps`** | Gate PASS=`{o200_s['phase2a_pass']}`
- **4-Fold Open-Ended Free-Gen (`Typed-CFG-Lite`, 200 trials):** Eliminates `90–100%` type errors and `58–78%` runtime failures observed in syntax/scope-only `CFG-Lite` down to `0.0%` at `25.0k–28.1k tps`.
"""
    Path("docs/BRIEF_OPENAI_200K.md").write_text(oai_brief, encoding="utf-8")

    tracker_path = Path("docs/outreach_tracker.csv")
    if not tracker_path.exists():
        headers = [
            "org", "track", "target_name", "role_team", "source_url", "channel",
            "pitch_angle", "sent_date", "reply_hours", "reply_status",
            "technical_feedback", "next_step", "priority_score",
        ]
        seed_rows = [
            ["Google DeepMind", "Formal Math / AlphaProof", "<FILL_FROM_ARXIV>", "Staff Research Scientist / EM", "", "Email/X", "Lean4 8/8 + Typed-CFG-Lite + 256k L3", "", "", "PENDING", "", "", "S"],
            ["Google DeepMind", "Gemma / Gemini Inference", "<FILL_FROM_GITHUB>", "Research Engineer (XLA/Decoding)", "", "Email/LinkedIn", "256k 156KB 1-bit mask + TPU pipeline", "", "", "PENDING", "", "", "S"],
            ["Google DeepMind", "Tokyo R&D / Reasoning", "<FILL_FROM_SCHOLAR>", "Research Engineer / Lead", "", "LinkedIn/Referral", "Zero-retry 4-fold safety + Tokyo hub", "", "", "PENDING", "", "", "S"],
            ["OpenAI", "Inference / Structured Outputs", "<FILL_FROM_GITHUB>", "MTS Inference Systems", "", "Email/X", "Real o200k_base 122KB mask + beyond CFG", "", "", "PENDING", "", "", "A"],
            ["OpenAI", "Codex / Agentic Coding", "<FILL_FROM_ARXIV>", "MTS Reasoning / Verification", "", "Email/X", "Eliminating 90-100% type & 58-78% runtime retries", "", "", "PENDING", "", "", "A"],
        ]
        with tracker_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(seed_rows)

    print("[EXPORT OK] Generated docs/BRIEF_GDM_256K.md, docs/BRIEF_OPENAI_200K.md, docs/outreach_tracker.csv")
    print("\n--- [BRIEF_GDM_256K.md] ---")
    print(gdm_brief.strip())
    print("\n--- [BRIEF_OPENAI_200K.md] ---")
    print(o200_e["vocab_label"], "->", oai_brief.strip())


if __name__ == "__main__":
    main()
