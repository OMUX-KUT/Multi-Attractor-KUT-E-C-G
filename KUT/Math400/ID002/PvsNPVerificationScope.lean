/-
  Module      : KUT.Math400.ID002.PvsNPVerificationScope
  Version     : v1.5.0 (Route 1: ID002 P vs NP Problem, Zero-Sorry)
  Archive ID  : KUT-MATH-400-ID002
  Slug        : p-versus-np-problem
  Status      : OPEN (Database v1.0.0)
  Policy      : Zero-Sorry / 3-Axiom Whitelist {propext, Classical.choice, Quot.sound}
-/

import Mathlib.Data.Set.Basic
import Mathlib.Tactic.Linarith

namespace KUT.Math400.ID002

/-- 形式言語（アルファベット α 上の有限文字列の集合） -/
abbrev Language (α : Type) := Set (List α)

/-- 多項式時間計算量バウンドの定義：∃ c k, ∀ n, f n ≤ c * (n + 1)^k -/
def PolyBound (f : ℕ → ℕ) : Prop :=
  ∃ c k : ℕ, ∀ n : ℕ, f n ≤ c * (n + 1) ^ k

/--
  計算複雑性クラス（P, NP および多項式時間帰着）の公理的インターフェース仕様
  （特定のチューリング機械符号化に依存せず、構造的帰結を厳密検証するための契約）
-/
structure ComplexityClassSpec (α : Type) where
  ClassP : Set (Language α)
  ClassNP : Set (Language α)
  p_subset_np : ClassP ⊆ ClassNP
  PolyTimeReducible : Language α → Language α → Prop
  reducible_refl : ∀ L, PolyTimeReducible L L
  reducible_trans : ∀ A B C, PolyTimeReducible A B → PolyTimeReducible B C → PolyTimeReducible A C
  p_closed_under_reduction : ∀ A B, PolyTimeReducible A B → B ∈ ClassP → A ∈ ClassP

/-- NP 困難性（NP-hard）の定義 -/
def IsNPHard {α : Type} (S : ComplexityClassSpec α) (L : Language α) : Prop :=
  ∀ A, A ∈ S.ClassNP → S.PolyTimeReducible A L

/-- NP 完全性（NP-complete）の定義 -/
def IsNPComplete {α : Type} (S : ComplexityClassSpec α) (L : Language α) : Prop :=
  L ∈ S.ClassNP ∧ IsNPHard S L

/--
  KUT-MATH-400-ID002 検証対象命題：P = NP 問題（Prop 定義のみ。公理化しない）
-/
def KUTPvsNPProp {α : Type} (S : ComplexityClassSpec α) : Prop :=
  S.ClassP = S.ClassNP

/-! ### 無条件・Zero-Sorry 検証定理群（Cook–Levin 構造帰結・多項式バウンド） -/

/-- P = NP は NP ⊆ P と同値である（P ⊆ NP より） -/
theorem p_eq_np_iff_np_subset_p {α : Type} (S : ComplexityClassSpec α) :
    KUTPvsNPProp S ↔ S.ClassNP ⊆ S.ClassP := by
  constructor
  · intro hEq
    rw [KUTPvsNPProp] at hEq
    rw [hEq]
  · intro hSub
    exact Set.Subset.antisymm S.p_subset_np hSub

/-- 任意の NP 完全問題が P に属するならば P = NP が成立する -/
theorem np_complete_in_p_implies_p_eq_np {α : Type} (S : ComplexityClassSpec α)
    {L : Language α} (hNPC : IsNPComplete S L) (hLP : L ∈ S.ClassP) :
    KUTPvsNPProp S := by
  rw [p_eq_np_iff_np_subset_p]
  intro A hANP
  exact S.p_closed_under_reduction A L (hNPC.2 A hANP) hLP

/-- 対偶：P ≠ NP ならば、いかなる NP 完全問題も P に属さない -/
theorem p_ne_np_implies_np_complete_not_in_p {α : Type} (S : ComplexityClassSpec α)
    (hNe : ¬ KUTPvsNPProp S) {L : Language α} (hNPC : IsNPComplete S L) :
    L ∉ S.ClassP := by
  intro hLP
  exact hNe (np_complete_in_p_implies_p_eq_np S hNPC hLP)

/-- NP 困難性は多項式時間帰着に関して上方閉である -/
theorem np_hard_upward_closed {α : Type} (S : ComplexityClassSpec α)
    {A B : Language α} (hA : IsNPHard S A) (hAB : S.PolyTimeReducible A B) :
    IsNPHard S B := by
  intro X hXNP
  exact S.reducible_trans X A B (hA X hXNP) hAB

/-- 任意の定数関数は多項式時間バウンドを満たす -/
theorem poly_bound_const (c : ℕ) : PolyBound (fun _ => c) := by
  refine ⟨c, 0, fun n => ?_⟩
  simp

/-! ### 公理汚染チェック（3公理ホワイトリスト監査用：全5定理） -/
#print axioms p_eq_np_iff_np_subset_p
#print axioms np_complete_in_p_implies_p_eq_np
#print axioms p_ne_np_implies_np_complete_not_in_p
#print axioms np_hard_upward_closed
#print axioms poly_bound_const

end KUT.Math400.ID002
