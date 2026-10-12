# Repair cached argument supervision before retraining scorer

The first scorer baseline used unversioned regex-derived argument arrays.
LeanDojo citation tags became phantom `a` arguments, and Unicode names such
as `h₁` were split. Updating a parser does not repair those arrays. Old cached
rows retain their original positional interpretation until this migration;
the running text retriever can finish its existing experiment consistently.

## New label semantics

- Strip only citation `<a>` tags; a genuine variable named `a` is retained.
- Preserve Unicode identifiers and qualified names.
- Prefer digest-bound `action_trace_v3` global names/local context indices.
- Without a trace, bare names resolve against the corpus or hypothesis graph
  structure; locals point to shared `FV` nodes, not masked display-name nodes.
- Compound terms, syntax modifiers/locations, fresh bindings, unknown
  references and over-budget sequences are explicit unsupported actions for
  the current flat-pointer decoder. They receive no argument/stop supervision
  and count as misses in overall complete-action evaluation. Conditional
  metrics on supported actions are labeled separately.
- Known external citation IDs are stored independently, including citations
  from actions this decoder cannot generate. Unsupported syntax is never
  relabeled as a valid zero-argument action.

This is a conservative flat-action dataset, not a full Lean syntax generator.
Graphs, vocabulary IDs, tactic labels, source coordinates and row population
stay the same. Test labels undergo the same deterministic migration; no model
is evaluated on test while settings are selected on validation.

## 1. Create separate packed label version (CPU)

Run from the server repository root after pulling `gnn-issue-fixes`:

```sh
python -u -m maths_ai.gnn_inference.scripts.relabel_prepared_arguments \
  --source-root maths_ai/_support_files/artifacts/prepared/v1 \
  --output-root maths_ai/_support_files/artifacts/prepared/argument_labels_v2 \
  --index-path maths_ai/gnn_inference/runs/lemma_index_v1 \
  --max-args 3
```

Append `--resume` after interruption. Source and completed output chunks are
hash-checked before reuse. No Lean recompilation or GNN/text index rebuild is
needed. Do not run ordinary preprocessing over this repaired root: that would
replace resolved/reference-aware labels with the generic lexical fallback.

Print the train/validation audit:

```sh
python - <<'PY'
import json
from pathlib import Path

root = Path("maths_ai/_support_files/artifacts/prepared/argument_labels_v2")
report = json.loads((root / "reports/argument_relabeling.json").read_text())
for split in ("train", "val"):
    print(split)
    print(json.dumps(report["splits"][split], indent=2))
PY
```

Inspect supported versus unsupported reasons and how many rows use resolved
source traces versus clean-name fallback **before launching more training**.
Counts can change because fake slots are removed and previously missed real
references are recovered; retain the old benchmark as a historical baseline.

## 2. Rebind existing text rankings to repaired rows

The frozen text model's rankings do not depend on argument labels. Reuse the
epoch-3 cache instead of re-encoding 250k query states. This verifies that old
source identities and all scorer graph inputs are unchanged, then copies
candidate IDs/scores into a new provenance-bound directory and recomputes
known-citation coverage against the repaired labels.

```sh
python -u -m maths_ai.gnn_inference.scripts.prepare_text_candidates \
  --checkpoint maths_ai/gnn_inference/runs/premise_retriever_text_full/epoch_3_snapshot.pt \
  --text-index maths_ai/gnn_inference/runs/premise_retriever_text_full/index_epoch_3 \
  --pointer-checkpoint maths_ai/gnn_inference/runs/pointer_gnn/run_20260910_131534/best.pt \
  --pointer-index maths_ai/gnn_inference/runs/lemma_index_v1 \
  --config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --prepared-root maths_ai/_support_files/artifacts/prepared/argument_labels_v2 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --reuse-candidate-dir maths_ai/gnn_inference/runs/text_candidates_epoch_3 \
  --output-dir maths_ai/gnn_inference/runs/text_candidates_epoch_3_labels_v2 \
  --device cpu
```

Append `--resume` if interrupted. A different snapshot or changed graph/source
identity is rejected. The lookup uses 128-D pointer-index vectors; text
embeddings are not passed into the pointer decoder.

## 3. Retrain/fine-tune scorer after checking label coverage

Use a new run, not `--resume` on the contaminated scorer checkpoint. The
existing pointer model is a warm start; the corrected labels and fixed
candidate table now determine argument/stop supervision.

```sh
python -u -m maths_ai.gnn_inference.scripts.train_text_candidate_scorer \
  --config maths_ai/gnn_inference/configs/pointer_gat_state_mean_attention_pretrained.json \
  --prepared-root maths_ai/_support_files/artifacts/prepared/argument_labels_v2 \
  --checkpoint maths_ai/gnn_inference/runs/pointer_gnn/run_20260910_131534/best.pt \
  --index-path maths_ai/gnn_inference/runs/lemma_index_v1 \
  --corpus-path maths_ai/_support_files/artifacts/lemmas/v2/corpus/lemmas.jsonl \
  --candidate-dir maths_ai/gnn_inference/runs/text_candidates_epoch_3_labels_v2 \
  --run-dir maths_ai/gnn_inference/runs/scorer_text_labels_v2 \
  --epochs 1 --batch-size 16 --k 200
```

Report overall and supported-only action scores with support counts. Overall
evaluation retains all validation rows and counts unsupported rows as misses;
training excludes their argument supervision. Validation never injects gold
lemmas. Expect new denominators and report them rather than interpreting a
coverage increase as improved model predictions. Lean proof success remains
unmeasured.
