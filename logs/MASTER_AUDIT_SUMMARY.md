# Master Audit Summary (Phase 1--3)

- Lean 4 Formal Verification: `8/8 theorems PASS` (`5` axiom-free, `3` depend only on `propext`)
- Phase 1 Hold-out Verdict: `PASS_TYPECLASS_LOGIT_MASK_VERIFIED`
- Phase 2A 128k ExactFullV: `893,184 B` (23.26% L3), `23,124.9 tps`
- Phase 2A 128k SparseTopK256: `632,576 B` (16.47% L3), `2,749,366.2 tps`
- Phase 2B CFG-Lite Verdict: `PASS_PHASE2B_CFG_LITE_FREEGEN`
- Phase 3 Typed-CFG-Lite Verdict: `PASS_PHASE3_TYPED_CFG_LITE_4FOLD_SAFE` (200 generations: AST=True, Scope=True, Type=True, Runtime=True, TPS>=500=True)
