# Accuracy Analysis: Mathematics-ai-atp GNN Pipeline

**Date**: 2026-06-29  
**Status**: P0 bugs fixed — retrain required

---

## Executive Summary

The GNN pipeline had **3 critical bugs** (P0), all now fixed. **5 significant architectural limitations** (P1) and **5 moderate issues** (P2) remain. The model needs retraining with the fixed parser and the corrected inference pipeline.

---

## P0 — Critical Bugs (FIXED)

### Bug 1: Inference uses forward-only edges; training uses bidirectional ✅

**Files**: `inference.py:162`, `pyg.py:41`, `training.py:411-422`

**What happened**: Training used bidirectional edges (`transform_edge_index(edge_mode="bidirectional")`), but inference used `dag_to_pyg()` with default `add_reverse_edges=False`. The GNN saw a fundamentally different graph topology at inference.

**Fix applied**: Added `data.edge_index = transform_edge_index(data.edge_index, edge_mode="bidirectional")` in `inference.py` after batch creation.

**Before**: `intro` probability = 0.999 (overconfident, wrong edges)  
**After**: `intro` probability = 0.209 (realistic, correct edges)

---

### Bug 2: Premise mask overridden to ALL nodes during argument selection ✅

**Files**: `argument_selector.py:243-247`, `pyg.py:89-107`

**What happened**: `build_premise_mask()` filtered structural nodes, but `argument_selector.py` overrode it with `(node_types >= 0) & (node_types <= 5)` which is always True. Every node became a valid candidate.

**Fix applied**: Removed the override. Now uses `data.premise_mask` directly.

**Before**: `rw` selected random library lemmas (structural nodes included)  
**After**: `rw` selects `:` (local node, correctly filtered)

---

### Bug 3: Score index mismatch for local-only tactics ✅

**File**: `inference.py:259-277`

**What happened**: `local_scores[local_indices.tolist().index(idx)]` used the wrong index space — position in sorted output vs position in unsorted filtered set.

**Fix applied**: Changed to `local_scores[local_sorted[rank]]` with `enumerate`.

**Before**: Confidence scores were scrambled (wrong values)  
**After**: Confidence scores are correct (e.g., `rw` score = 1.5080)

---

## P1 — Significant Architectural Limitations

### Issue 4: No residual connections in 4-layer GraphSAGE

**File**: `model.py:75-80`

```python
for index, conv in enumerate(self.convs):
    x = conv(x, data.edge_index)
    x = F.relu(x)
```

With `num_layers=4`, deep GraphSAGE without residuals suffers from **oversmoothing**: all node embeddings converge to similar values.

**Impact**: GNN cannot distinguish structurally similar but semantically different nodes.

---

### Issue 5: Readout is single State node only

**File**: `model.py:82-99`

```python
def readout(self, node_embeddings, data):
    return node_embeddings.index_select(0, state_node_index)
```

Only the `State` node embedding is used. After 4 GNN layers, only nodes within 4 hops contribute.

**Impact**: Information loss for complex proof states with deep expression trees.

---

### Issue 6: Single pool built for all tactic candidates

**File**: `inference.py:190-198`

```python
pools = build_unified_pools(state_emb, node_embeddings, batch.premise_mask, batch.batch, ...)
pool = pools[0]  # ONE pool used for ALL tactic candidates
```

Different tactics need different premises (`rw` needs rewrite lemmas, `cases` needs inductive hypotheses, etc.). One pool scores identically for all tactics.

**Impact**: Suboptimal argument selection for each tactic type.

---

### Issue 7: Argument ground-truth uses first-match label heuristic

**File**: `preprocess.py:107-119`

```python
label_to_id = {}
for node in dag.nodes:
    if node.label not in label_to_id:
        label_to_id[node.label] = node.id  # FIRST match wins
```

In a hash-consed DAG, multiple nodes share labels. The ground-truth may reference a different node with the same label.

**Impact**: Noisy training signal for argument selection.

---

### Issue 8: Inference bypasses TacticWithArgsClassifier.forward

**File**: `inference.py:178-182`

Inference manually calls `backbone.encode_nodes()`, `backbone.readout()`, `backbone.classifier()` — bypassing `TacticWithArgsClassifier.forward` which has the autoregressive argument selection logic.

**Impact**: Code duplication; improvements to `forward` don't benefit inference.

---

## P2 — Moderate Impact

### Issue 9: No edge-type encoding

**File**: `pyg.py:37-81`

All edges treated identically. DAG has semantically distinct edge types (parent-child, hypothesis name→type, goal→expression, binder→variable).

---

### Issue 10: No hard negative mining in premise ranking

**File**: `premise_scoring.py:202-298`

Cross-entropy over entire pool with no negative mining. Easy negatives dominate the loss.

---

### Issue 11: Missing tactics from arity registry

**File**: `labels.py:46-112`

Missing: `split`, `exists`, `injection`, `clear`, `revert`, `rename`, `swap`, `rotate`, `done`, `sorry`, `iterate`, `repeat`, `all_goals`, `any_goals`, `first`, `try`, `rwa`. Default arity is 1, wrong for 0-arg tactics.

---

### Issue 12: No proof history or next-state context

**File**: `preparation.py:14-19`

Only stores current proof state. Missing: tactic history, first-tactic flag, success/failure status, depth in proof tree.

---

### Issue 13: Training data prepared with old parser

Dataset at `artifacts/prepared/v1/` was prepared before the parser comma fix. Models trained on old DAG topology need retraining.

---

## Recommended Fix Priority

### Done:

1. ✅ Fix Bug 1: Bidirectional edges in inference
2. ✅ Fix Bug 2: Premise mask override removed
3. ✅ Fix Bug 3: Score index mapping fixed

### Next (retrain):

4. Re-run `--stages prepare,baseline,pointer` with fixed parser

### Later (architecture):

5. Add residual connections to GraphSAGE
6. Add multi-head attention readout
7. Build tactic-aware candidate pools
8. Add edge-type encoding
9. Add hard negative mining
10. Complete arity registry
11. Add proof history context
12. Unify inference with model.forward
