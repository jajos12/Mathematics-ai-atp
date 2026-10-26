# Frozen-graph local identity probe

Purpose: distinguish poor task-specific local ranking from a deficient graph
representation, **without** changing the text retriever, GNN encoder, source
gate, GRU or lemma adapter. Initialize from normalized joint V2 epoch-3
`best.pt`; train **only** its local pointer output projection on all corrected
flat-supported train rows with a local reference (14,171 rows / 16,821 target
positions before any oversized-graph exclusions). Gold tactic and previous
arguments are teacher-forced. A preceding library lemma uses its frozen text
index vector for GRU context even if it missed the serving top-200 shortlist.

Validation reports all **322** supported local positions and the **277** in
complete actions with all required library lemmas in the frozen top-200 pool.
The latter is the matched baseline: original joint V2 local identity
**130/277 on GPU** (CPU ties gave 129/277). No gold is injected into the
serving candidate pool; this is conditional identity accuracy, not a complete
action or Lean proof-success metric. The original local pool remains unchanged;
the FV-only diagnostic rescued just 3/277 targets.

Server repo root, after pulling the code; check disk and GPU 0 first:

```sh
python -u -m maths_ai.gnn_inference.scripts.train_local_identity \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_available_labels_v2 \
  --joint-checkpoint maths_ai/gnn_inference/runs/sexpr_joint_source_flat_epoch_5_count_pilot/best.pt \
  --text-checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_5_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_5 \
  --candidate-dir maths_ai/gnn_inference/runs/sexpr_text_library_epoch_5 \
  --run-dir maths_ai/gnn_inference/runs/sexpr_local_identity_full_frozen_v2 \
  --device cuda:0 --epochs 3 --batch-size 32 \
  --expected-val-local 322 --expected-val-reachable 277
```

The script first evaluates **epoch 0**, saves it as the initial `best.pt`,
and refuses a changed matched cohort. It checkpoints every 50 train batches;
repeat the identical command with `--resume` after interruption. A completed
run can extend total `--epochs` with `--resume`. Selection uses matched local
top-1, then all-local top-1 and matched top-5; never test-set metrics.

Inspect every `metrics_epoch_N.json` for both denominator counts, top-1 and
top-5, plus excluded oversized train local positions printed at startup.
If full local training cannot beat its own epoch-0 matched baseline, do not
claim graph information is absent: try a task-specific unfrozen encoder or
inspect ambiguous local references/Lean-valid alternative hypotheses first.
The experiment does not train library ranking or full tactic syntax.
The pointer projection is shared with the joint decoder's contextual library
residual: **do not swap these weights into serving** without rechecking
library ranking, source choice and complete actions on the matched cohort.

## Experiment-cycle outcome

Eight-epoch server run trained on **16,816/16,821** supported local train
positions (five oversized exclusions). Best epoch 8 improved matched val
local top-1 **130→152/277**, all-supported local **155→182/322**, with graph,
tactic embedding, GRU and text adapter frozen. Optional first-slot
`exact=8, rw=8, simp=4` cues then gave **167/277** matched local identity,
**73/759** oracle-tactic nonempty exact flat actions and **27/759**
predicted-tactic nonempty exact. These are offline imitation metrics.

Final paired comparison added train-selected first-slot `subst=8`; its
standalone rank test gives **390→426/623 train**, **6→8/13 val**, with
21 train harms and no observed val harms. Full validation reproduced the
three-family reference, then improved matched local identity **167→169/277**
and oracle nonempty exact **73→75/759** (both 2 rescues / 0 harms).
Predicted nonempty exact stayed **27/759**, supported predicted exact
**206/1,207**. Retain the four-family configuration as optional, default-off.

**Offline experiment cycle complete.** Relative to the same epoch-8 scorer
without cues: local identity **152→169/277**, oracle nonempty exact
**64→75/759**, predicted nonempty exact **19→27/759**. There are still
108 wrong matched local identities; later-operand/type-unification work and
live Lean checking are follow-ups, not proof-success claims.
See [pilot results and final settings](joint_source_flat_pilot.md).
