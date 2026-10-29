# miniF2F validation development cohort

Move beyond the ten curated smoke tasks: corrected automation solved all ten,
while the model solved nine, so that set demonstrates plumbing but no added
model coverage. This run measures additional checked proofs on harder tasks.

## Frozen selection

Source: `yangky11/miniF2F-lean4`, commit
`5746b7d6c47855ce1294bed87329618ff7f1bc31`, **validation only**.
Choose five each from IMO, AIME, algebra, induction and number-theory prefixes
by ascending SHA-256 of `42:name`: **25 tasks**. Names are fixed before reading
proofs, checking compatibility or seeing model results. No test-source fetch
or test-split option. miniF2F pretraining exposure is unknown; this is not a
pretraining-held-out evaluation or an official leaderboard score.

Original source hashes, exact headers/import/open context and revision are
saved in `cohort.json` and `sources/`. Original proof bodies are not passed
to model or checker. Every selected task stays in denominator, including
ingestion/elaboration failures. No replacement tasks are sampled.

Upstream uses Lean **4.24.0** and Mathlib
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. Existing server source project uses
Lean **4.10.0-rc1**. `--allow-environment-port` explicitly labels a development
port, keeping statements unchanged; version/syntax incompatibility remains
an input failure, not a model failure. Exact-upstream mode requires both
original toolchain and Mathlib pins. No environment rebuild is automatic.

## CPU phase: source compatibility and automation

From server repository root in `atp`:

```sh
git pull --ff-only jajos gnn-issue-fixes
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_valid25_dev \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --per-group 5 --seed 42 --allow-environment-port --baseline-only
```

Only small pinned source files are downloaded. No GPU/model/cache access.
The original header is elaborated into a frozen proposition before candidate
code, preserving binders, instances and coercions without reconstructing
them from pretty-printed goals. A host-only stub supplies the type, never
a permitted proof premise: final axiom checks reject any use of that stub
or `sorryAx`. Existing imported target theorem names are rejected too.

Same closure-requiring automation portfolio is tried on compatible tasks.
Read `compatible_tasks`, `input_failures`, `baseline_checked_tasks`,
`automation_solved`, `model_eligible_tasks`. `completed_tasks` can be below
25 in this phase because unsolved compatible tasks await model attempts.
If many inputs fail compatibility, inspect those errors before inference;
do not silently rewrite hypotheses or present port results as official ones.

## Model phase: only automation-unsolved compatible tasks

Resume the same report without `--baseline-only`:

```sh
LD_PRELOAD= python -B -u -m maths_ai.gnn_inference.scripts.run_minif2f_dev \
  --output-dir maths_ai/gnn_inference/runs/minif2f_valid25_dev \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --per-group 5 --seed 42 --allow-environment-port --resume \
  --device cuda:0 --dtype bfloat16 \
  --context-tokens 8192 --max-new-tokens 4096 --max-attempts 2
```

Budgets above are also defaults bound in the CPU phase. No model weights
are downloaded: use cached official reference revision from backend control,
one model load and the same tokenizer-fidelity checks. At most one repair
per attempted task. Fixed original statement is never truncated; marked
failure-history compaction can fit repair context. Final target/axiom audit
supports equivalent binder layouts and rejects extra assumptions.

Inspect `combined_solved` over all 25 selected tasks, `model_only_solved`
(additional solves), `model_eligible_tasks` (conditional denominator), calls,
timeouts/truncation and input failures. Baseline-solved tasks do not consume
model calls. A reported additional solve is a checked theorem under recorded
environment/budget, not evidence of novel mathematics or causal repair gains.
Code/source/environment/budget changes require a new directory; interrupted
runs checkpoint and resume. No model training or retriever change.

## Verification

**48 focused tests passed / 1 optional full-Mathlib integration skipped**.
Includes deterministic valid-only selection, source/header preservation,
hidden proof exclusion, baseline-first skipping, CPU-to-model resume and
real Lean **4.10.0-rc1** tests of frozen binder types, extra-assumption
rejection and rejection of proofs using the specification stub.
Actual pinned 25-source preparation completed locally: **25/25 parsed**.
Full server compatibility/baseline/model results have not been measured yet.
