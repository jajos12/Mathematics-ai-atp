# Normalized flat-action joint-source pilot

This is an **offline argument experiment**, not full Lean tactic syntax or a
publishable bundle. The normalized GNN provides local-node vectors; frozen
MiniLM epoch 5 provides library IDs, cosine scores and 384-D lemma vectors.
A learned adapter feeds lemma vectors to the 128-D GRU. Source choice precedes
within-source identity at **every** argument position, including mixed actions.
Library ordering starts as frozen text ranking, with a zero-initialized
GRU-conditioned residual for later argument positions; no text-parser GNN
lemma vectors are mixed with normalized local nodes.

After the first pilot stopped after at most one argument, the revised decoder
predicts **0–3 argument slots explicitly** from state and tactic. It trains
this count on *all* supported actions, even when a cited lemma is absent from
top 200. The old binary stop head remains an auxiliary diagnostic, not a
greedy-length decision. Length classes receive smoothed inverse-frequency
weights from training rows; local-source errors receive a training-only
source weight. On retrieval misses, a reachable gold prefix can still teach
its source, but missing identity and subsequent GRU/stop targets are masked.

The corrected flat-action cohort contains 16,821 train / 322 val local and
42,116 train / 802 val library positions (5,301 / 97 mixed-source actions).
Unsupported syntax receives no argument or stop supervision. Retrieval-absent
gold cannot receive fabricated stop or candidate supervision. Validation never
injects gold; source/local/library ranks are teacher-forced diagnostics, while
complete-action counts use predicted choices. No Lean proof success is implied.

Commands below run on server from repository root after syncing code. Confirm
disk space and available GPU first. Candidate preparation may take substantial
time for 218,727 train states; `--resume` reuses committed row batches. The
table stores IDs/scores only (~350 MB for train top-200), not per-row text
vectors. FAISS index stays in `index_epoch_5`.

```sh
python -u -m maths_ai.gnn_inference.scripts.prepare_text_library_candidates \
  --checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_5_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_5 \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_available_labels_v2 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --output-dir maths_ai/gnn_inference/runs/sexpr_text_library_epoch_5 \
  --device cuda:0 --batch-size 32 --faiss-threads 4
```

Review `val_metrics.json`: any/all-citation@200 and micro recall are distinct.
The earlier 69.26% is any-citation retrieval on 2,765 cited validation rows;
this cache must reproduce that cohort before training. New train/val rows are
bound to their corrected action and citation labels and the exact retriever.

```sh
python -u -m maths_ai.gnn_inference.scripts.train_joint_source_decoder \
  --prepared-root maths_ai/_support_files/artifacts/prepared/sexpr_available_labels_v2 \
  --candidate-dir maths_ai/gnn_inference/runs/sexpr_text_library_epoch_5 \
  --text-checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_5_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_5 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --baseline-checkpoint maths_ai/gnn_inference/runs/sexpr_available_old_settings/run_20261015_084134/best.pt \
  --pointer-config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --run-dir maths_ai/gnn_inference/runs/sexpr_joint_source_flat_epoch_5_count_pilot \
  --device cuda:0 --epochs 1 --batch-size 32 --max-train-batches 500
```

Use this **new** run directory: the previous `joint_source_flat_decoder_v1`
checkpoint has no count head and is not resumable as the revised v2 model.
This pilot freezes graph backbone and text retriever, training tactic
embeddings, local pointer head, count head, auxiliary stop head, source gate
and lemma adapter.
It resumes within an epoch using `--resume` with identical settings. To
extend a completed run, keep every setting but increase `--epochs` and pass
`--resume`. Its `best.pt` is selected by nonempty oracle-tactic action match,
then count macro recall and reachable loss, **not** by stop-only matches.
Report retrieval-absent rows, both-sources-available source accuracy,
auxiliary stop diagnostics, **per-length count recall**, local/library source
recall and candidate ranks, nonempty actions, and all supported actions
separately. Compare library top-1 to frozen text ordering on the same
reachable argument positions; reject a learned rerank if it worsens that baseline.
Training does not greedily decode actions, so exact-action scores are only
reported for validation. Compare this **same-500-batch** run against v1 before
starting a separate larger-data run (for example, 2,000 batches).

For a **controlled count-loss ablation** after V2's three epochs, repeat the
same 500-batch/epoch command and seed with `--epochs 3`, add
`--count-weight-exponent 1.0`, and use a **new** run directory such as
`runs/sexpr_joint_source_flat_epoch_5_count_inverse_pilot`. Default `0.5`
remains the original square-root inverse-frequency policy; `1.0` weights
rare lengths more strongly from *training* counts only. It cannot resume the
old V2 run because that would switch objectives mid-run. Compare count-2/3
recall, zero/one losses, oracle/predicted nonempty exact, and source/identity
on the same validation cohort; weight changes alone are not a proved fix.

The library pool currently searches the full corpus: declaration accessibility
at the historical proof position remains unverified. Serving integration,
full syntax generation, same-live-state Pantograph parity and Lean-checked
proof search are separate gates before a composite bundle can be exported.

### Optional local type cue (offline experiment only)

`local_type_signals` compares normalized goal/hypothesis S-expressions: literal
type equality for `exact`, and Eq/Iff/HEq side-in-goal for the **first** `rw`
argument. A soft bonus of **8**, selected on training rows, improved local
first-choice ranks on a small `exact`/`rw` validation subset; it is not a
general Lean-validity rule. `SourceCandidates.local_type_signal` carries the
per-local features; set `JointSourceDecoder.local_type_bonus_strength = 8`
to test them. Default **0** preserves existing V2 checkpoints and scoring.
The source gate receives the *original* local maxima, so the bonus changes
within-local ranking only at position zero; later `rw` positions may use a
different goal. Compare full-cohort local/complete-action outcomes using
predicted tactic names where appropriate before integration. These offline
sidecars have not been verified against live Pantograph state fields (#42).
