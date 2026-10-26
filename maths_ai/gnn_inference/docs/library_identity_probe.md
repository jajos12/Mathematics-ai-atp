# Independent library-query probe in the joint decoder

## Audit and scope

Read-only full-val audit covered 4,157 states / 1,207 supported actions.
There are 802 gold library positions in supported actions, but only **408**
positions in actions whose *entire* gold sequence is retrieved/selectable.
On those 408 positions, frozen text, original joint V2 and epoch-8 local
checkpoint all give **62 top-1 / 131 top-5**. MRR: **0.24420 / 0.24420 /
0.24433**, respectively. The local checkpoint changed only the shared
`argument_selector.out_proj` weight/bias, with zero library top-1 changes.

| Argument slot | Matched library positions | Text top-1 | Text top-5 |
|---|---:|---:|---:|
| First | 324 | 57 | 116 |
| Second | 74 | 5 | 13 |
| Third | 10 | 0 | 2 |

Current residual multiplier is **-0.010771**. Residual/prior score-spread
ratio is only **0.001514 median / 0.002608 p95** (joint), **0.002575 /
0.004358** (local checkpoint). Context-only ranks put gold first **2/408**
and **1/408**, respectively. Prefix context changes logits slightly on 84
later positions but **never** changes the combined top-1. Predicted identity
is a previous library reference on **11/408** teacher-forced positions;
gold repeats on **6/408**. These are conditional identity diagnostics, not
greedy multi-action repeat rates; no blanket no-repeat rule follows.

## Controlled modification

This is an opt-in component **inside `JointSourceDecoder`**, not a standalone
retriever/reranker model. `enable_library_identity()` adds one independent
linear map `library_query: hidden_dim -> hidden_dim`, initialized to zero.
It scores frozen adapted text keys with the existing tactic/prefix GRU
hidden, adding the scaled dot product to unchanged legacy library logits.
Unlike the old residual, its gradient is not multiplied by the tiny legacy
scalar. Separate library query protects the finalized local projection.

The new correction is added *after* computing source-gate evidence, which
keeps original source logits at the same state/prefix. Changed greedy
library choices can still affect later GRU states and source decisions.
Graph, local query, tactic embedding, count/source/stop heads, GRU, text
adapter, legacy residual and text temperature are frozen. No candidate
masking or repeat ban is added. Old V2 checkpoints remain unchanged when
the component is not enabled; new checkpoints need it enabled before
strict loading.

Library training covers supported rows with library references. A gold
identity gets cross-entropy only if present in frozen top-k. A missing
earlier lemma may supply its full-index vector as an oracle *prefix*, never
as an added candidate; subsequent retrieved identities can still train.
Validation separates all 802 gold positions, individually retrieved target
positions, and the historical **408 matched** positions. Do not compare
individually-retrieved accuracy with the 408-position baseline.
Local cues are unnecessary for gold-prefix training; final greedy
evaluation must explicitly restore the finalized four-family settings.

Focused checks: **25 passed**, including zero-correction baseline equality,
library rank changes with unchanged local/source scores, missing-prefix
supervision without candidate injection, all-missing identity masking,
strict checkpoint reload, resume-policy rejection and exact interrupted-
versus-uninterrupted query training.

## Server pilot

Sync edited code and the new training script first. Check disk/GPU space;
no new retrieval caches or embeddings are generated. Use a new run directory.
Three epochs with a 500-batch/epoch cap is the initial probe, not full-data
training. The cap is applied to library-containing rows, so it is not the
same exposure as 500 arbitrary state batches in the original joint pilot.

```sh
python -B -u -m maths_ai.gnn_inference.scripts.train_library_identity \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_available_labels_v2 \
  --joint-checkpoint maths_ai/gnn_inference/runs/sexpr_joint_source_flat_epoch_5_count_pilot/best.pt \
  --local-checkpoint maths_ai/gnn_inference/runs/sexpr_local_identity_full_frozen_v2/best.pt \
  --text-checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_5_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_5 \
  --candidate-dir maths_ai/gnn_inference/runs/sexpr_text_library_epoch_5 \
  --run-dir maths_ai/gnn_inference/runs/sexpr_library_identity_query_pilot \
  --device cuda:0 --epochs 3 --batch-size 32 --max-train-batches 500 \
  --learning-rate 0.001 --k 200 \
  --expected-val-library 802 --expected-val-matched 408 --expected-baseline-top1 62
```

Epoch-zero validation must reproduce 802 total / 408 matched positions and
62 matched hits before training. Preserve `metrics_epoch_0.json` and initial
`best.pt`. The runner checkpoints every 50 training batches and at phase
boundaries; repeat identical options with `--resume` after interruption.
Only `--epochs` may increase; source data, checkpoint hashes, learning rate,
batch cap and candidate policy are resume-bound. `best.pt` selection uses
matched top-1, then MRR, then top-5; the epoch-zero checkpoint stays eligible.
No test split is loaded or used for selection.

Read `val_metrics_epoch_N.json`: `matched_positions` (408),
`matched_top1_sum`, `matched_top5_sum`, `matched_mrr`, and text baseline
counts; compare train metrics and per-slot retrieved counts separately.
If top-1 does not beat 62/408, do not claim improved library identity from
a minor MRR tie-break. If it does, perform paired full-validation greedy
evaluation using finalized local cues: reference local **169/277**, oracle
nonempty **75/759**, predicted nonempty **27/759**. Report library gains and
harms plus complete-action/source/local regressions before any serving use.
This probe does not address retrieval misses, live accessibility or Lean
proof success. No server training result has been measured yet.
