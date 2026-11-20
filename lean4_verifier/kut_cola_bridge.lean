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

#print axioms ast_sync_guarantees_mod0
#print axioms gate_and_falsification_exclusive
#print axioms cram_fallback_sound
#print axioms head_l3_isolation_sound

end KUT.ColaHybrid
