/-
  Module      : KUT.Math400.ID004.YangMillsScope
  Version     : v1.6.0 (Route 1: ID004 Yang-Mills Existence and Mass Gap, Zero-Sorry)
  Archive ID  : KUT-MATH-400-ID004
  Slug        : yang-mills-existence-and-mass-gap
  Status      : OPEN (Database v1.0.0)
  Policy      : Zero-Sorry / 3-Axiom Whitelist {propext, Classical.choice, Quot.sound}
-/

import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace KUT.Math400.ID004

/-- コンパクト単純ゲージ群 G に付随する4次元量子ヤン・ミルズ理論のスペクトル仕様 -/
structure QuantumYangMillsSpec where
  /-- ハミルトニアン H のスペクトル（エネルギー固有値集合） -/
  Spectrum : Set ℝ
  /-- 真空状態エネルギー E₀ = 0 ∈ Spectrum -/
  vacuum_in_spectrum : (0 : ℝ) ∈ Spectrum
  /-- エネルギー非負性（スペクトル条件：すべての E ∈ Spectrum に対し 0 ≤ E） -/
  spectrum_nonneg : ∀ E : ℝ, E ∈ Spectrum → 0 ≤ E

/-- 質量ギャップ Δ > 0 の存在定義：真空以外の全スペクトルが Δ 以上である -/
def HasMassGap (T : QuantumYangMillsSpec) (Δ : ℝ) : Prop :=
  0 < Δ ∧ ∀ E : ℝ, E ∈ T.Spectrum → E ≠ 0 → Δ ≤ E

/--
  KUT-MATH-400-ID004 検証対象命題：ヤン・ミルズ存在と質量ギャップ（Prop 定義のみ。公理化しない）
-/
def KUTYangMillsMassGapProp (T : QuantumYangMillsSpec) : Prop :=
  ∃ Δ : ℝ, HasMassGap T Δ

/-! ### 無条件・Zero-Sorry 検証定理群（スペクトル孤立・質量ギャップ単調性・相関減衰） -/

/-- 質量ギャップ Δ > 0 が存在すれば、開区間 (0, Δ) にエネルギー準位は存在しない -/
theorem no_spectrum_in_mass_gap_interval (T : QuantumYangMillsSpec) {Δ E : ℝ}
    (hGap : HasMassGap T Δ) (hE : E ∈ T.Spectrum) (hpos : 0 < E) :
    Δ ≤ E :=
  hGap.2 E hE (ne_of_gt hpos)

/-- 質量ギャップ Δ が成立すれば、任意のより小さい正数 Δ' ∈ (0, Δ] も質量ギャップである -/
theorem mass_gap_monotone_downward (T : QuantumYangMillsSpec) {Δ Δ' : ℝ}
    (hGap : HasMassGap T Δ) (hpos : 0 < Δ') (hle : Δ' ≤ Δ) :
    HasMassGap T Δ' := by
  refine ⟨hpos, fun E hE hne => ?_⟩
  exact le_trans hle (hGap.2 E hE hne)

/-- 質量ギャップ Δ > 0 が存在すれば、開区間 (0, Δ) 内の任意の値はスペクトルに属さない -/
theorem open_gap_disjoint_spectrum (T : QuantumYangMillsSpec) {Δ E : ℝ}
    (hGap : HasMassGap T Δ) (h0 : 0 < E) (hlt : E < Δ) :
    E ∉ T.Spectrum := by
  intro hE
  have hge := no_spectrum_in_mass_gap_interval T hGap hE h0
  linarith

/-- 質量ギャップ Δ > 0 に対応する相関長 ξ = 1 / Δ は厳密に正である -/
theorem correlation_length_pos {Δ : ℝ} (hΔ : 0 < Δ) :
    0 < 1 / Δ := by
  positivity

/-- 質量ギャップ Δ > 0 が大きいほど、相関長 ξ = 1 / Δ は短くなる（単調減少性） -/
theorem correlation_length_antitone {Δ₁ Δ₂ : ℝ} (h1 : 0 < Δ₁) (hle : Δ₁ ≤ Δ₂) :
    1 / Δ₂ ≤ 1 / Δ₁ :=
  one_div_le_one_div_of_le h1 hle

/-! ### 公理汚染チェック（3公理ホワイトリスト監査用：全5定理） -/
#print axioms no_spectrum_in_mass_gap_interval
#print axioms mass_gap_monotone_downward
#print axioms open_gap_disjoint_spectrum
#print axioms correlation_length_pos
#print axioms correlation_length_antitone

end KUT.Math400.ID004
