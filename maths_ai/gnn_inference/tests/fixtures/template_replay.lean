import Init

namespace TemplateSmoke

theorem rwGoal (a b : Nat) (h : a = b) : a = b := by
  rw [h]

theorem rwBack (a b : Nat) (h : a = b) (hb : b = b) : True := by
  rw [← h] at hb
  trivial

theorem rwForwardAt (a b : Nat) (h : a = b) (ha : a = a) : True := by
  rw [h] at ha
  trivial

theorem simpOnly (n : Nat) : n + 0 = n := by
  simp only [Nat.add_zero]

theorem simpAt (P : Prop) (h : P) (hx : P) : P := by
  simp only [h] at hx
  exact h

theorem simpBoth (P : Prop) (h : P) (hx : P) : P := by
  simp only [h] at hx ⊢

theorem simpPlain (n : Nat) : n = n := by
  simp

theorem simpEmpty (n : Nat) : n = n := by
  simp only []

theorem rwApply (a : Nat) : a + 0 = a := by
  rw [Nat.add_zero a]

theorem rwTwo (a b c : Nat) (h1 : a = b) (h2 : c = b) (hx : a = a) : True := by
  rw [h1, ← h2] at hx
  trivial

theorem rwThree (a b c d : Nat) (h1 : a = b) (h2 : b = c) (h3 : c = d)
    (hx : a = a) : True := by
  rw [h1, h2, h3] at hx
  trivial

theorem rwRepeat (a b : Nat) (h : a = b) (hx : a = a) : True := by
  rw [h, ← h] at hx
  trivial

theorem rwTrailing (a b : Nat) (h : a = b) : a = b := by
  rw [h,]

theorem rwAscii (a b : Nat) (h : a = b) : a = b := by
  rw [<- h]

theorem simpStar (P : Prop) (h : P) : P := by
  simp [*]

end TemplateSmoke
