# Continue retrieval training while measuring complete actions

Full-data text epoch 3 reached validation R@200 64.81%, R@10 26.43%, MRR
0.14187. It can continue to epoch 5 while downstream experiments use an
immutable epoch-3 snapshot. The held-out test split stays untouched until
retrieval/model settings are selected on validation.

Text vectors are 384-dimensional, while the action decoder expects the
pointer GNN's 128-dimensional vectors. Therefore **text selects declaration
IDs**, but the scorer uses the selected IDs' original pointer-bound lemma
vectors, alongside local graph-node vectors. Do not replace the pointer index
with the text index. This measures whether improved candidate selection helps
the existing action architecture; it does not train a text-to-graph projection.

## 1. Freeze epoch 3 before starting more training

From server repository root, after pulling `gnn-issue-fixes`:

```sh
python - <<'PY'
from pathlib import Path
import torch

run = Path("maths_ai/gnn_inference/runs/premise_retriever_text_full")
snapshot = run / "epoch_3_snapshot.pt"
if snapshot.exists():
    saved = torch.load(snapshot, map_location="cpu", weights_only=False)
else:
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 3 and saved["phase"] == "complete", "Need completed epoch-3 checkpoint"
    saved.pop("optimizer_state_dict", None)
    torch.save(saved, snapshot)
assert saved["epoch"] == 3 and saved["phase"] == "complete"
print("Frozen:", snapshot)
PY
```

The preparation command verifies the snapshot's lemma encoder and tokenizer
against `index_epoch_3`. It will refuse an epoch-4/5 substitution.

## 2. Continue text training in one server shell

```sh
python -u -m maths_ai.gnn_inference.scripts.train_text_retriever \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --baseline-json maths_ai/gnn_inference/runs/premise_retriever/run_20261001_122611/text_query_diagnostics.json \
  --run-dir maths_ai/gnn_inference/runs/premise_retriever_text_full \
  --epochs 5 --batch-size 8 --max-train-batches 0 --resume
```

## 3. Prepare fixed candidates in a second shell (CPU)

This script retrieves train/validation candidates from the frozen text model.
Only proof-state text enters the query. It also reports **all-known-citations
coverage** and citation-level recall alongside the original any-hit Recall@K.
"All known" covers resolved trace labels, not missing or implicit premises.

```sh
python -u -m maths_ai.gnn_inference.scripts.prepare_text_candidates \
  --checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_3_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_3 \
  --pointer-checkpoint maths_ai/gnn_inference/runs/pointer_gnn/run_20260910_131534/best.pt \
  --pointer-index maths_ai/gnn_inference/runs/lemma_index_v1 \
  --config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --output-dir maths_ai/gnn_inference/runs/text_candidates_epoch_3 \
  --device cpu --batch-size 32
```

Candidate generation resumes with the same command plus `--resume`. It writes
source-row and file hashes, frozen retriever/index provenance, and table IDs
and scores. No trained text tower is resident on the scorer GPU afterward.
CPU preparation avoids GPU contention with ongoing retrieval training; actual
runtime and host-memory availability still need measurement.

## 4. Train one scorer epoch after tables finish

Run this on a free GPU (e.g. after epoch 5 finishes); do not launch two GPU
training jobs concurrently without checking available devices/memory first.

```sh
python -u -m maths_ai.gnn_inference.scripts.train_text_candidate_scorer \
  --config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --prepared-root maths_ai/_support_files/artifacts/prepared/v1 \
  --checkpoint maths_ai/gnn_inference/runs/pointer_gnn/run_20260910_131534/best.pt \
  --index-path maths_ai/gnn_inference/runs/lemma_index_v1 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --candidate-dir maths_ai/gnn_inference/runs/text_candidates_epoch_3 \
  --run-dir maths_ai/gnn_inference/runs/scorer_text_epoch_3 \
  --epochs 1 --batch-size 16 --k 200
```

Scorer saves every 50 training batches and saves `train_done` **before**
validation. Add `--resume` after interruption; a `train_done` run repeats
validation without retraining. Validation does not inject gold citations.
Reports include predicted- and oracle-tactic exact complete-action match,
action Recall@5, source accuracy, and unresolved target counts. Model bundles
trained with this cached text policy refuse legacy serving loaders without
the same candidate-policy provenance; live text-retrieval serving integration
and Lean closed-loop proof evaluation remain separate next steps.

## 5. Print downstream reports

```sh
python - <<'PY'
import json
from pathlib import Path

root = Path("maths_ai/gnn_inference/runs")
coverage = root / "text_candidates_epoch_3/val_metrics.json"
if coverage.exists():
    print("VALIDATION RETRIEVAL COVERAGE:")
    print(json.dumps(json.loads(coverage.read_text()), indent=2))
for path in sorted((root / "scorer_text_epoch_3").glob("metrics_epoch_*.json")):
    m = json.loads(path.read_text())
    print(path.name)
    for key in ("complete_action_exact_match_predicted_tactic",
                "complete_action_exact_match_oracle_tactic",
                "action_recall_at_5_predicted_tactic",
                "candidate_source_accuracy_predicted_tactic",
                "reranking_target_coverage", "evaluated_count"):
        print(key, m[key])
PY
```
