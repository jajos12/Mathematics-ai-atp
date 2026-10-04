# Premise retriever comparison (pilot)

Two **experimental** dual encoders compete against text-query BM25 on the
same 4,260 validation states, of which 2,501 have a resolved cited external
declaration. The primary comparison is any-cited-lemma Recall@1/10/50/200 and
MRR **among those 2,501 steps**. BM25's R@200 is 41.30% in
`text_query_diagnostics.json`; this is not proof success or all-argument recall.
Do not use validation gold names in either query. Both experiments obtain
training labels from the prepared version-3 tactic traces. For the text model,
the original upstream `state` is joined by split, row index, theorem, tactic,
and dataset identity; only the goal and hypothesis text enters the query.

Commands run from the repository root on the server in the `atp` environment.
If `transformers`/`datasets` are missing for the text experiment, install
`transformers>=4.44` and `datasets>=2.20` in that environment. Both models need the same
existing prepared root and 318,953-declaration corpus:

```sh
python -u -m maths_ai.gnn_inference.scripts.train_text_retriever \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --baseline-json maths_ai/gnn_inference/runs/premise_retriever/run_20261001_122611/text_query_diagnostics.json \
  --run-dir maths_ai/gnn_inference/runs/premise_retriever_text_joint \
  --epochs 3 --batch-size 8 --max-train-batches 1000
```

Text towers start independently from
`sentence-transformers/all-MiniLM-L6-v2`. Goal text is placed before the
hypotheses to survive 192-token truncation. Lemma text is `name : type`.
Both towers receive multi-positive contrastive gradients; additional negatives
include random declarations and top BM25 false positives. They search the
full declaration corpus, not a topic-filtered subset.

```sh
python -u -m maths_ai.gnn_inference.scripts.train_joint_graph_retriever \
  --config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --checkpoint maths_ai/gnn_inference/runs/pointer_gnn/run_20260910_131534/best.pt \
  --index-path maths_ai/gnn_inference/runs/lemma_index_v1 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --baseline-json maths_ai/gnn_inference/runs/premise_retriever/run_20261001_122611/text_query_diagnostics.json \
  --run-dir maths_ai/gnn_inference/runs/premise_retriever_graph_joint \
  --epochs 3 --batch-size 16 --max-train-batches 800
```

Graph towers start from pointer backbone copies; **both** are trainable.
The state tower consumes prepared normalized S-expression graphs; the lemma
tower consumes graphs parsed from pretty-printed declaration types. Their
separate encoders can specialize, but the representation difference remains
an experimental limitation. Training positives and negatives are encoded
through the live lemma tower, rather than gathered from the old index. The old
index is used only to mine negatives for epoch 1; subsequent mining uses the
previous epoch's index. Each epoch **re-encodes all 318,953 declarations**
before exact validation. Never evaluate a trainable tower against stale index
vectors.

Both defaults are **small pilots**, not fully trained final models. They save
optimizer/model state every 100 training batches and a `train_done` checkpoint
before the costly index refresh. Index construction writes per-batch progress
and resumes after interruption. Rerun the identical command with `--resume`
after a pod respawn. Increase training steps only after inspecting learning
curves, GPU memory and completion time; changed negative counts, batch sizes,
dataset, model or corpus cannot silently resume an existing run.

Each run writes `metrics_epoch_N.json` beside `last.pt`, `best.pt`,
`best_index.json`, and versioned `index_epoch_N/`. Compare best validation
R@200, R@10, MRR, label/positive counts and wall time to the **same** baseline.
Best checkpoints are selected by validation R@200 (MRR breaks ties).
Select settings on validation; retain the test split for later evaluation.
These experimental checkpoints/indices cannot be swapped into the current
pointer/scorer serving bundle without implementing compatible loaders and
training the scorer against the chosen candidate distribution. Closed-loop
Lean evaluation still requires live graph parity (#42).
