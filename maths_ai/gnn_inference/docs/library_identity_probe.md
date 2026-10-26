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

## First pilot command (completed)

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
proof success.

## First pilot outcome: not accepted

Epoch-zero baseline reproduced. Three capped epochs gave matched top-1
**62→63→65→63/408**; top-5 **131→150→156→156/408**. Best checkpoint is
epoch 2, MRR **0.27044** versus baseline **0.24433**. Paired full-val check:

| Metric | Baseline | Epoch 2 | Rescued / harmed |
|---|---:|---:|---:|
| Matched local | 169/277 | 169/277 | 0 / 0 |
| Matched library | 62/408 | 65/408 | 14 / 11 |
| Matched library top-5 | 131/408 | 156/408 | 40 / 15 |
| Teacher-forced source | 551/685 | 551/685 | 0 / 0 |
| Oracle nonempty exact | 75/759 | 74/759 | 6 / 7 |
| Predicted nonempty exact | 27/759 | 26/759 | 1 / 2 |

Keep previous baseline. Zero-/two-/three-argument oracle outcomes are
unchanged; one-argument exact drops **75→74/475**. Rank improvement alone
does not justify replacing the accepted action baseline.

## Next controlled ablation: original text keys

Test whether frozen GRU-adapter keys constrain library identity. This is a
hypothesis, not an established cause. `--library-key-space text` changes
only the independent query's output width and scoring keys: project GRU
hidden directly to the original text width (384 for current index) and
score unit-normalized frozen text vectors. Their normalized coordinate
scale makes an additional `1/sqrt(text_dim)` unnecessary. Original adapter
continues supplying GRU inputs; local/source scores and frozen dual encoder
stay unchanged. Default `adapted` mode retains the original pilot behavior.

Pull updated code, then reuse the command above with:

```sh
--library-key-space text \
--run-dir maths_ai/gnn_inference/runs/sexpr_library_identity_text_keys_pilot
```

Keep all other settings identical: three epochs, 500 library-row batches,
learning rate 0.001, seed 42 and same candidate/epoch-zero guards. Use a new
directory; do not resume the adapted-key run into this policy. Resume binds
key space, and checkpoint loading requires:

```python
model.enable_library_identity(
    key_space=checkpoint["binding"].get("library_key_space", "adapted"),
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
```

Both modes have zero correction at epoch zero. Focused checks: **28 passed**,
including frozen-component invariance and rejecting cross-mode resume or
in-place query replacement. Original-text-key training is pending; no new
rank or action gain is claimed.
