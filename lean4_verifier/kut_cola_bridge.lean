namespace KUT.ColaHybrid

def L3_CACHE_LIMIT_BYTES : Nat := 3840000
def MIN_THROUGHPUT_TPS : Nat := 500

/-- Layer 1（KUT-110D軽量ヘッド）とLayer 2（バックボーン）を分離した監査状態 -/
structure HybridDecodeState where
  tokenLen : Nat
  patchSize : Nat
  blockSize : Nat
  astBrokenCount : Nat
  sorryCount : Nat
  headWorkingSetBytes : Nat
  backboneParamBytes : Nat
  headThroughputTps : Nat
  deriving Repr

def computeAstSyncPad (len patchSize : Nat) : Nat :=
  if patchSize = 0 then 0
  else (patchSize - (len % patchSize)) % patchSize

/-- 定理1: AST境界同期パディング適用後は常に Mod0 整列となる（split完全バージョン非依存版） -/
theorem ast_sync_guarantees_mod0 (len patchSize : Nat) (h : patchSize > 0) :
    (len + computeAstSyncPad len patchSize) % patchSize = 0 := by
  unfold computeAstSyncPad
  split
  · next h0 =>
    rw [h0] at h
    exact False.elim (Nat.lt_irrefl 0 h)
  · next _ =>
    cases hmod : len % patchSize with
    | zero =>
      rw [Nat.sub_zero, Nat.mod_self, Nat.add_zero]
      exact hmod
    | succ r =>
      have hlt : len % patchSize < patchSize := Nat.mod_lt len h
      have hpos : 0 < len % patchSize := by
        rw [hmod]
        exact Nat.zero_lt_succ r
      have hsub_lt : patchSize - len % patchSize < patchSize := Nat.sub_lt h hpos
      rw [← hmod]
      rw [Nat.mod_eq_of_lt hsub_lt]
      rw [Nat.add_mod]
      rw [Nat.mod_eq_of_lt hsub_lt]
      rw [Nat.add_comm (len % patchSize) (patchSize - len % patchSize)]
      rw [Nat.sub_add_cancel (Nat.le_of_lt hlt)]
      exact Nat.mod_self patchSize

def HybridGatePass (s : HybridDecodeState) : Prop :=
  s.patchSize > 0 ∧
  s.tokenLen % s.patchSize = 0 ∧
  s.astBrokenCount = 0 ∧
  s.sorryCount = 0 ∧
  s.headWorkingSetBytes ≤ L3_CACHE_LIMIT_BYTES ∧
  s.headThroughputTps ≥ MIN_THROUGHPUT_TPS

def FalsificationTriggered (s : HybridDecodeState) : Prop :=
  s.astBrokenCount > 0 ∨
  s.sorryCount > 0 ∨
  s.headWorkingSetBytes > L3_CACHE_LIMIT_BYTES ∨
  s.headThroughputTps < MIN_THROUGHPUT_TPS

/-- 定理2: ゲート通過と反証発火の排他性 -/
theorem gate_and_falsification_exclusive (s : HybridDecodeState) :
    HybridGatePass s → ¬ FalsificationTriggered s := by
  intro ⟨_, _, hast, hsorry, hl3, htps⟩ hfals
  cases hfals with
  | inl h =>
    rw [hast] at h
    exact Nat.lt_irrefl 0 h
  | inr hrest =>
    cases hrest with
    | inl h =>
      rw [hsorry] at h
      exact Nat.lt_irrefl 0 h
    | inr hrest2 =>
      cases hrest2 with
      | inl h => exact Nat.lt_irrefl _ (Nat.lt_of_lt_of_le h hl3)
      | inr h => exact Nat.lt_irrefl _ (Nat.lt_of_lt_of_le h htps)

def shouldFallbackToCram (s : HybridDecodeState) : Bool :=
  decide (s.headWorkingSetBytes > L3_CACHE_LIMIT_BYTES) ||
  decide (s.headThroughputTps < MIN_THROUGHPUT_TPS)

/-- 定理3: CRAM単体回帰フラグの健全性 -/
theorem cram_fallback_sound (s : HybridDecodeState) :
    shouldFallbackToCram s = true → FalsificationTriggered s := by
  intro h
  unfold shouldFallbackToCram at h
  cases h1 : decide (s.headWorkingSetBytes > L3_CACHE_LIMIT_BYTES) with
  | false =>
    rw [h1, Bool.false_or] at h
    exact Or.inr (Or.inr (Or.inr (of_decide_eq_true h)))
  | true =>
    exact Or.inr (Or.inr (Or.inl (of_decide_eq_true h1)))

/-- 定理4: Layer 2 バックボーン容量の変更は Layer 1 L3ゲート合否に影響しない -/
theorem head_l3_isolation_sound (s : HybridDecodeState) (newBackboneBytes : Nat) :
    HybridGatePass s → HybridGatePass { s with backboneParamBytes := newBackboneBytes } := by
  intro h
  exact h

/-- 4種型クラス＋スコープ内変数制約ロール -/
inductive TokenRole where
  | struct
  | identDecl
  | typeName
  | exprAtom
  | scopeIdent
  deriving DecidableEq, Repr

def countRoleViolations {α : Type} (allowed : TokenRole → α → Bool) :
    List (TokenRole × α) → Nat
  | [] => 0
  | (r, tok) :: rest =>
    (match allowed r tok with | true => 0 | false => 1) +
    countRoleViolations allowed rest

def AllRoleClosed {α : Type} (allowed : TokenRole → α → Bool) :
    List (TokenRole × α) → Prop
  | [] => True
  | (r, tok) :: rest => allowed r tok = true ∧ AllRoleClosed allowed rest

/-- 定理5: ロジットマスクによる型クラス閉包は構文ロール違反件数 0 を無公理で保証する -/
theorem typeclass_closure_zero_violation {α : Type}
    (allowed : TokenRole → α → Bool)
    (stream : List (TokenRole × α))
    (hClosed : AllRoleClosed allowed stream) :
    countRoleViolations allowed stream = 0 := by
  induction stream with
  | nil => rfl
  | cons hd tl ih =>
    cases hd with
    | mk r tok =>
      cases hClosed with
      | intro hhd htl =>
        unfold countRoleViolations
        rw [hhd, ih htl]

/-- 定理6: 型クラス・ロジットマスク常駐メモリ加算後もL3上限内ならGateを通過し続ける -/
theorem typeclass_mask_budget_sound (s : HybridDecodeState) (bitmaskBytes : Nat)
    (hPass : HybridGatePass s)
    (hBudget : s.headWorkingSetBytes + bitmaskBytes ≤ L3_CACHE_LIMIT_BYTES) :
    HybridGatePass { s with headWorkingSetBytes := s.headWorkingSetBytes + bitmaskBytes } := by
  cases hPass with
  | intro h1 h23456 =>
    cases h23456 with
    | intro h2 h3456 =>
      cases h3456 with
      | intro h3 h456 =>
        cases h456 with
        | intro h4 h56 =>
          cases h56 with
          | intro _ h6 =>
            exact ⟨h1, h2, h3, h4, hBudget, h6⟩

/-- Phase 3: 厳格分離型（int / float / bool） -/
inductive SimpleTy where
  | tInt
  | tFloat
  | tBool
  deriving DecidableEq, Repr

def tyEq : SimpleTy → SimpleTy → Bool
  | .tInt, .tInt => true
  | .tFloat, .tFloat => true
  | .tBool, .tBool => true
  | _, _ => false

theorem tyEq_refl (t : SimpleTy) : tyEq t t = true := by
  cases t <;> rfl

/-- 非ゼロ分母ガード（|x| + 1 ≥ 1）の形式モデル -/
def safePositiveDenom (absVal : Nat) : Nat := absVal + 1

/-- 定理7 (Phase 3): 非ゼロ分母ガード適用後の除数は常に 0 ではない（無公理） -/
theorem safe_denom_nonzero (absVal : Nat) : safePositiveDenom absVal ≠ 0 :=
  Nat.succ_ne_zero absVal

def countTypeMismatches : List (SimpleTy × SimpleTy) → Nat
  | [] => 0
  | (exp, act) :: rest =>
    (match tyEq exp act with | true => 0 | false => 1) + countTypeMismatches rest

def AllTypeMatched : List (SimpleTy × SimpleTy) → Prop
  | [] => True
  | (exp, act) :: rest => exp = act ∧ AllTypeMatched rest

/-- 定理8 (Phase 3): Typed-CFG-Lite の型指向閉包は型不一致件数 0 を無公理で保証する -/
theorem typed_cfg_zero_mismatch (pairs : List (SimpleTy × SimpleTy))
    (h : AllTypeMatched pairs) :
    countTypeMismatches pairs = 0 := by
  induction pairs with
  | nil => rfl
  | cons hd tl ih =>
    cases hd with
    | mk exp act =>
      cases h with
      | intro heq htl =>
        subst heq
        unfold countTypeMismatches
        rw [tyEq_refl exp, ih htl]

#print axioms ast_sync_guarantees_mod0
#print axioms gate_and_falsification_exclusive
#print axioms cram_fallback_sound
#print axioms head_l3_isolation_sound
#print axioms typeclass_closure_zero_violation
#print axioms typeclass_mask_budget_sound
#print axioms safe_denom_nonzero
#print axioms typed_cfg_zero_mismatch

end KUT.ColaHybrid
