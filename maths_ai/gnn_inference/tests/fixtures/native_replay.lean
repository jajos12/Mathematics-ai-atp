import Init

namespace ReplaySmoke

def keep (n : Nat) : Nat := n

theorem localProof (P : Prop) (h : P) : P := by
  exact h

theorem twoGoals (P : Prop) (h : P) : P ∧ P := by
  constructor
  · exact h
  · exact h

theorem localDefinition (n : Nat) : keep n = n := by
  rfl

theorem applyProof (P Q : Prop) (h : P → Q) (hp : P) : Q := by
  apply h
  exact hp

theorem rewriteProof (a b : Nat) (h : a = b) : a = b := by
  rw [h]

theorem truth : True := by
  exact True.intro

theorem letProof (n : Nat) : n = n := by
  let m := n
  have hm : m = m := rfl
  exact hm

theorem instanceProof {α : Type} [Inhabited α] : Nonempty α := by
  exact Nonempty.intro default

end ReplaySmoke
