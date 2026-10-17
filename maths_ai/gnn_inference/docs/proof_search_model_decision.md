# Proof-search action model: team decision brief

**Status:** architecture decision pending. Goal is Lean-checked proof search, not
just reproducing the next tactic in an offline dataset.

## Why this decision matters

Search needs complete executable actions: choose a tactic, its syntax, local
hypotheses and/or library lemmas, apply it to the *same* live Lean state, then
explore resulting goals. Tactic-class accuracy and lemma Recall@K measure
different prerequisites; neither measures whether an action runs or a theorem
is proved. A syntax-limited decoder can miss valid proof paths even with a
correct tactic classifier and retriever.

## What we have measured

- Normalized-S-expression graphs cover 218,727 train / 4,157 validation rows.
  The new GNN baseline's best validation **tactic top-1 is 34.22%** (epoch 87);
  the normalized local-target audit found **4,880/4,880 selectable** traced
  local references. No new-graph joint local/library reranker is trained.
- Current flat format represents only **61,853/218,727 train (28.3%)** and
  **1,207/4,157 validation (29.0%)** recorded actions, including no-argument
  actions. These are *target coverage*, not model accuracy or proof success.
- V3 annotated source-syntax traces compiled for **217,896 train** and
  **4,148 validation** rows during relabeling. The existing compiler produces
  tree-shape operations with typed reference leaves, but also permits missing
  or unannotated leaves. It discards some original spelling/spacing; no
  operation-to-Lean renderer or round-trip execution check exists. Trace
  availability is **not** demonstrated full-action generation coverage.
- The full text retriever reached **68.49% any-citation Recall@200** at epoch
  5 on the *older* 2,501-labeled-step cohort; its recall on the new normalized
  cohort is unknown. More text training does not resolve action syntax.

## Approaches worth comparing

1. **Retrieved-premise-conditioned tactic text.** Fine-tune a pretrained
   encoder-decoder on cleaned complete tactic strings; execute candidates in
   Lean. [LeanDojo/ReProver](https://arxiv.org/html/2306.15626v2#S5) provides
   a Lean-specific precedent. Requires input-leakage checks, accessible
   premises, compute/storage budget, and evaluation of local-name errors.
2. **Structured syntax plus reference selection.** Predict tactic structure
   and fill local/library slots from typed candidates. Coq
   [ASTactic](https://arxiv.org/html/1905.09381v1#S5) demonstrates a restricted
   grammar, **not** full Lean syntax or superiority here. Our v3 targets first
   need a faithful renderer, length/leaf audits, and Lean round-trip tests;
   arbitrary terms and fresh names remain separate generation problems.
3. **Hybrid**, only if measured: structured decoding for reliable cases,
   text generation for unsupported syntax. Compare added complexity against
   actual Lean-checked gains. Existing GNN tactic scores may be an optional
   prior, not a hard gate or automatically required serving component.

## Evidence needed before committing to an architecture

Audit v3 operation lengths, missing/unannotated leaves, and render/reparse
success on held-out validation; check live normalized-graph parity and which
library declarations were accessible at each proof step. Compare text and
structured approaches on **the same validation states, candidate policy, and
search budget**: report syntax validity, Lean step execution, complete-action
match and completed checked proofs separately. Select using validation, not
test metrics already emitted by the baseline trainer. Pantograph supports
Lean interaction ([paper](https://arxiv.org/html/2410.16429v2#S4)); Lean's
[language reference](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/)
distinguishes source syntax, elaboration and kernel checking.

**Team decision:** whether to fund a small pretrained tactic-text baseline
first, investigate a faithful structured renderer first, or compare both in
parallel; set compute/storage and proof-search evaluation budgets before
training. No approach has yet shown better proof success for this repository.
