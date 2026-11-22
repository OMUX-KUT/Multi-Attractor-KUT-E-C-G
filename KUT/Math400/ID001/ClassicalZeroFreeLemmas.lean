/-
  Module      : KUT.Math400.ID001.ClassicalZeroFreeLemmas
  Version     : v1.2.1 (Route A: Layer B-2 3-4-1 Trigonometric Inequality, deprecation-clean)
  Archive ID  : KUT-MATH-400-ID001
  Slug        : riemann-hypothesis
  Status      : OPEN (Database v1.0.0)
  Policy      : Zero-Sorry / 3-Axiom Whitelist {propext, Classical.choice, Quot.sound}
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import KUT.Math400.ID001.ZetaVerificationScope

namespace KUT.Math400.ID001.LayerB2

/-! ### Layer B-2 Core: Mertens / de la Vallée Poussin 3-4-1 不等式群（Zero-Sorry） -/

/-- 3-4-1 三角恒等式：3 + 4 cos θ + cos(2θ) = 2 (1 + cos θ)² -/
theorem trig_341_eq_two_sq (θ : ℝ) :
    3 + 4 * Real.cos θ + Real.cos (2 * θ) = 2 * (1 + Real.cos θ) ^ 2 := by
  rw [Real.cos_two_mul]
  ring

/-- 3-4-1 三角不等式：すべての実数 θ に対して 3 + 4 cos θ + cos(2θ) ≥ 0 -/
theorem trig_341_nonneg (θ : ℝ) :
    0 ≤ 3 + 4 * Real.cos θ + Real.cos (2 * θ) := by
  rw [trig_341_eq_two_sq]
  positivity

/-- 任意の非負重み w ≥ 0 を掛けた 3-4-1 項の非負性（ディリクレ級数の各項評価用） -/
theorem weighted_trig_341_nonneg {w : ℝ} (hw : 0 ≤ w) (θ : ℝ) :
    0 ≤ w * (3 + 4 * Real.cos θ + Real.cos (2 * θ)) :=
  mul_nonneg hw (trig_341_nonneg θ)

/-- 対数領域の 3-4-1 非負性から指数積 |ζ(σ)|³ |ζ(σ+it)|⁴ |ζ(σ+2it)| ≥ 1 形式への変換補題 -/
theorem exp_product_341_ge_one {a b c : ℝ} (h : 0 ≤ 3 * a + 4 * b + c) :
    1 ≤ Real.exp a ^ 3 * Real.exp b ^ 4 * Real.exp c := by
  have hEq : Real.exp a ^ 3 * Real.exp b ^ 4 * Real.exp c = Real.exp (3 * a + 4 * b + c) := by
    rw [← Real.exp_natCast_mul, ← Real.exp_natCast_mul, ← Real.exp_add, ← Real.exp_add]
    push_cast
    ring
  rw [hEq]
  exact Real.one_le_exp h

/-- 古典ゼロフリー領域仕様（ClassicalZeroFreeRegionSpec）が存在すれば臨界線右側近傍に零点なし -/
theorem classical_zero_free_implies_no_zero_in_region
    (spec : KUT.Math400.ID001.ClassicalZeroFreeRegionSpec)
    (s : ℂ) (hs_re : 1 - spec.c / Real.log (|s.im| + 2) ≤ s.re) (hs_ne : s ≠ 1) :
    riemannZeta s ≠ 0 :=
  spec.zero_free s hs_re hs_ne

/-! ### 公理汚染チェック（3公理ホワイトリスト監査用：全5定理） -/
#print axioms trig_341_eq_two_sq
#print axioms trig_341_nonneg
#print axioms weighted_trig_341_nonneg
#print axioms exp_product_341_ge_one
#print axioms classical_zero_free_implies_no_zero_in_region

end KUT.Math400.ID001.LayerB2
