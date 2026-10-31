# Unsupported-action census

The full prepared V1 coverage scan found **151,823 train / 2,835 val**
unsupported-or-invalid rows. These are not Lean execution failures: most
have no supported executable rendering. Existing flat/template/scope
counters overlap, so they do not explain that population exclusively.

This census assigns one **primary blocker** to each such row, keeps
overlapping secondary syntax flags, and samples original tactics plus
annotated syntax by family/blocker. It reuses the completed coverage report
without reloading graphs, retraining, changing renderer support or running
Lean. Choose subsequent work from training counts and inspected examples;
validation is a fixed comparison, not a grammar-selection source.

## Server command

Pull published code, then run from repository root in `atp`:

```sh
python -B -u -m maths_ai.gnn_inference.scripts.audit_unsupported_actions \
  --coverage-dir maths_ai/gnn_inference/runs/template_v1_coverage \
  --output-dir maths_ai/gnn_inference/runs/unsupported_action_census_v1
```

Input must be a complete `template_coverage_v1` train/val report with its
original `rows.jsonl`. The runner verifies per-split row digests, counts,
statuses and uniqueness, then checks all recorded cache-file hashes for
each unsupported row before inspecting its source. A changed snapshot
requires a fresh coverage audit rather than silently changing the census.
Renderable rows retain their original support status without cache rereads.

Output must be a fresh directory outside input roots. Progress checkpoints
are marked incomplete; no resume is supported. The command is CPU-only
and needs no Lean backend. It reads no official test split.

## Primary categories and precedence

`unsupported_census_v1` uses this fixed operational order:

1. Invalid/missing cache or invalid source/context trace.
2. Missing syntax, then unresolved identifier annotations.
3. New/scoped bindings.
4. Non-single root tactic wrappers, then nested `by` proofs, then tactic
   control (`<;>`, sequencing, alternatives, repetition, goal dispatch,
   case blocks and focus), recognized from pinned constructors/atoms.
5. Known derived/alias/local/private-reference resolution failures.
6. Specific V1 rejections: applied/parenthesized/other compound rules,
   rule budgets, simp modifiers, locations, config, unsupported roots/shapes.
7. Flat rejections: fresh binding syntax, modifiers/locations, expression
   syntax, reference budget and unsupported tactic names.
8. Explicit `unclassified` residual for unknown rejection reasons.

Primary means the first observed blocker under this policy, **not** a
proven sole root cause or a promise that fixing it makes the row executable.
For example, a tactic can contain unresolved identifiers, scoped names and
a nested proof; the first is primary and all three remain secondary flags.
Unannotated names need inspection: not every one is an extraction bug or a
premise reference. Derived-reference spelling mismatches can also be aliases,
so the census does not pretend to distinguish them without native resolution.

Some known buckets remain coarse (`other_*`, unsupported root/shape).
`needs_subclassification_steps` counts these plus `unclassified_steps`.
A zero unclassified count alone does not mean every cause is understood.
Alignment is an independent dimension, not a substitute for syntax diagnosis.

## Outputs and next decision

- `report.json`: split totals, exclusive `primary_counts`, `by_family`,
  `by_primary`, secondary flags and counts of rows containing each syntax
  kind. Legacy-label exclusions and the eight train flat-only cases are
  counted separately inside each group.
- `train_ranked_blockers`: descending train counts, exact-aligned counts,
  work-area hints and flags for coarse categories. Counts are not estimated
  recoverable coverage; examples may reveal multiple necessary changes.
- `diagnostics.jsonl`: one row per unsupported-or-invalid action, with
  primary blocker, secondary flags, root kind, alignment and unknown local
  context indices.
- `examples.jsonl`: up to three examples per split/family/blocker, selected
  by lowest SHA256 of split/row ID. Includes dataset tactic, state preview,
  rejection details, local context preview and bounded annotated syntax.
  Truncation is marked; full traces remain in the hash-bound source caches.
  Use `--samples-per-family-blocker` to change the sample limit.

Required invariant for each split:

```text
sum(primary_counts) = unsupported_or_invalid_steps
selected_steps = renderable_steps + unsupported_or_invalid_steps
```

Expected current counts: **218,727 = 66,904 + 151,823 train** and
**4,157 = 1,322 + 2,835 val**. Review the largest train buckets, coarse or
unclassified residuals, and original examples before choosing a bounded
grammar, annotation, reference-resolution or data-quality change.

Verification: **11 new census tests pass**; combined focused checks verified
**66 non-native tests**. Three optional native tests are unavailable on this
rerun because the temporary pinned backend was removed; they were skipped
after correcting the stale environment setting. Census tests cover exact
accounting with overlapping flags, invalid context references, source-only
alignment, deterministic examples, unchanged inputs, stale snapshot/report
rejection and explicit unclassified reasons. Full server census is pending.
