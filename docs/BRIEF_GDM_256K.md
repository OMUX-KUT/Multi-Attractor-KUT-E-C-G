# Google DeepMind Outreach Brief (Measured 256k L3 + Lean 4 + 4-Fold 0.0% Safety)
- **Repository:** `https://github.com/OMUX-KUT/Multi-Attractor-KUT-E-C-G`
- **Lean 4.36.0:** 8/8 theorems PASS (0 warnings, 0 `sorry`, 5 axiom-free, 3 `propext`-only)
- **Measured 256k (`256,000`) Gemini/Gemma-Scale C++17 L3 Audit:**
  - `BitPacked-ExactFullV`: Mask `160,000 B` | Total Layer-1 **`1,221,120 B` (`31.80%` of 3.84 MB L3)** | **`11,228.4 tps`** | Gate PASS=`True`
  - `BitPacked-SparseTopK256`: Mask `160,000 B` | Total Layer-1 **`710,656 B` (`18.51%` of 3.84 MB L3)** | **`1,996,504.2 tps`** | Gate PASS=`True`
- **Official Gemma 256k Real Vocab Status:** `SKIPPED_NEEDS_HF_TOKEN_OR_LOCAL_GEMMA_TOKENIZER` (256,000-scale synthetic density measured directly above; real-vocab drop-in supported via `--real-mask`)
- **4-Fold Open-Ended Free-Gen (`Typed-CFG-Lite`, 200 trials):** `0.0%` AST / `0.0%` Scope / `0.0%` Static Type / `0.0%` Sandbox Runtime (`25.0k–28.1k tps`, 2.3x faster than untyped `CFG-Lite`).
