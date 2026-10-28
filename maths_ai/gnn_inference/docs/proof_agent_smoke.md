# Ten-task local-model proof smoke

Fixed curated compatibility set: two logic, two natural-number arithmetic,
one integer inequality, two real arithmetic, two list and one set theorem.
This is not a held-out benchmark, not the 4,157-row imitation evaluation,
and not evidence of novel theorem solving. Pretraining exposure is unknown.

## Execution

- Use the already-cached official DeepSeek-Prover-V2-7B revision from the
  backend control. Cached-only loading is enforced; no model download.
- Load one frozen BF16 model on one CUDA device, reuse across all tasks.
- Context 4,096; output cap 2,048; initial attempt plus at most one repair.
  Temperature 0.6, top-p 0.95, top-k 50, repetition penalty 1.0.
- Fresh conversation per task. Repair uses the prior final code and actual
  verifier diagnostics, not the long reasoning transcript. Elision is marked;
  input overflow fails explicitly rather than truncating the statement.
- Baseline runs the same fixed automation portfolio on all ten tasks; it
  does not skip model attempts or supply baseline proofs to the model.
- Generated declarations compile in isolated namespaces. The host freezes
  expected type before model code, then kernel-checks a theorem of that type.
  Equivalent binder arrangements are supported; extra assumptions are not.
  Axiom checks reject sorry/added axioms. Each candidate uses a fresh compiler.
- Save after raw generation, validation and task completion. Resume reuses
  stored output awaiting validation and does not repeat completed tasks.
  Tasks, model revision, code, packages, Lean pins and budgets are bound.

## Server command

From repository root in `atp`, pull first. The existing compiled source
Mathlib project used by native replay can supply Lean; no sudo/Ollama/new
environment required. This project uses Lean 4.10.0-rc1. Local reference
fixture tests used 4.29.1; actual run toolchain/manifest are recorded.

```sh
git pull --ff-only jajos gnn-issue-fixes
python -B -u -m maths_ai.gnn_inference.scripts.run_proof_smoke \
  --output-dir maths_ai/gnn_inference/runs/deepseek_ten_task_smoke \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4 \
  --device cuda:0 --dtype bfloat16 \
  --context-tokens 4096 --max-new-tokens 2048 --max-attempts 2 --seed 42
```

The runner verifies complete cached checkpoint shards and only requires
1 GiB free cache disk for reports/overhead, rather than the first-download
20 GiB guard. It checks CUDA and a real Lean preflight before inference.
Use `--hf-cache` only if the official control used a non-default cache.
Repeat identical command with `--resume` after interruption; changing
budgets or pulling code changes requires a fresh output directory.

## Results

Per-attempt progress prints task/status/token counts. `OUTPUT_DIR/report.json`
retains all ten statements, prompts, seeds, raw responses and compiler feedback.
Final summary includes:

- `selected_tasks`, `completed_tasks`: fixed denominator 10.
- `first_try_solved`, `solved_with_repairs`, `newly_solved_by_repair`.
- `automation_solved`, `model_only_solved`, `automation_only_solved`.
- `model_calls`: completed generations; `generation_requests_started` also
  records interrupted/failed request invocations across explicit resumes.

Failed or truncated tasks remain in denominator. Repair outcomes do not
establish causal feedback benefit without a matched resampling control.
No training, retrieval, source-native evaluation or agent-search claim.

## Verification

**41 focused smoke/backend/joint/local checks passed**, including real Lean
compilation of trusted reference proofs for all ten tasks, extra-assumption
rejection and equivalent-binder acceptance. Reference proofs live only in
tests; the runtime task manifest contains statements and categories only.

## First server result

Cached BF16 model loaded once in **72.11 s**. All ten tasks completed using
**12 generation requests**, with no hidden retries:

| Measure | Result |
|---|---:|
| Model first try | 8/10 |
| Model with at most one repair | 9/10 |
| Fixed automation portfolio | 9/10 |
| Model-only solved | 1 |
| Automation-only solved | 1 |
| Post-hoc union of verified results | 10/10 |

Model `nat_cancel` was Lean-rejected on both attempts. `set_distrib` hit the
2,048-token cap first, then passed its feedback retry. That retry recovered
truncation, not a demonstrated mathematical compiler-error correction.
Other eight tasks passed first try. Baseline-only task must be `nat_cancel`;
the model-only task identity and automation diagnostic need the detailed
report and must not be guessed from aggregate counts.

Both systems solve nine tasks but different ones. Their union is observed
coverage, **not a separately executed automation-first combined policy**.
No general 90% theorem-solving claim or superiority over automation follows.
This smoke establishes usable proof-completion/validation plumbing beyond
the single published example. Next inspect the failed cancellation proofs
and model-only case before changing prompts, training or orchestration.

## Failure audit and automation-baseline correction

`nat_cancel` invented `Nat.eq_of_add_eq_add_right` and repeated the same
unknown constant on repair. This is a model error; `Nat.add_left_cancel`
exists in the pinned Lean 4.10 core. `real_nonneg` is the reported model-only
task: its proof supplies `sq_nonneg x` to arithmetic automation.

However, the original automation portfolio used `first`, which accepts a
branch that merely makes progress. `ring` normalized the inequality without
closing it, preventing the later `nlinarith` branch. Thus the original 9/10
automation result is not a fair evaluation of the intended solver portfolio.
Model proofs/results remain valid, but model-only/union comparisons need
the corrected baseline. Portfolio now uses `solve`, which requires closure.
No theorem-specific rule or extra fact was added.

Recheck only automation, preserving original report/model outputs, in a new
directory (no GPU, checkpoint loading, downloads or inference):

```sh
git pull --ff-only jajos gnn-issue-fixes
python -B -u -m maths_ai.gnn_inference.scripts.run_proof_smoke \
  --recheck-automation maths_ai/gnn_inference/runs/deepseek_ten_task_smoke/report.json \
  --output-dir maths_ai/gnn_inference/runs/deepseek_ten_task_automation_recheck \
  --lean-project maths_ai/_support_files/sexpr_environment/mathlib4
```

Recheck enforces the same task list, Lean toolchain, Lake manifest and
validator time budget. It retains old automation records, copies unchanged
model attempts, checkpoints each new baseline result and supports `--resume`.
Final `model_calls` remains the original count; `new_model_calls` is zero.

Checks: **23 unit tests passed**. Focused real Lean checks verified closure
after a partial branch and the corrected nonnegativity portfolio (5.72 s),
plus `solve` and cancellation-name availability in Lean 4.10. The expanded
full integration invocation exceeded its aggregate local time limit; it is
not reported as passing. Corrected server baseline result is pending.
