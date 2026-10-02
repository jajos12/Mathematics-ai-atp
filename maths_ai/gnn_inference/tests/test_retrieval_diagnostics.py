"""Verify lexical retrieval against known token and ranking expectations."""

import torch
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord
from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import retrieval_metrics
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_diagnostics import (
    PremiseBM25,
    lexical_tokens,
)
from maths_ai.gnn_inference.scripts.diagnose_retriever import _oov_stats


def test_bm25_ranks_matching_names_and_types_on_original_ids() -> None:
    records = [
        LemmaRecord(17, "Nat.add_comm", "Nat.add a b = Nat.add b a", "Nat", "Mathlib"),
        LemmaRecord(99, "Set.union_subset", "Set.union A B ⊆ C", "Set", "Mathlib"),
        LemmaRecord(41, "Nat.mul_comm", "Nat.mul a b = Nat.mul b a", "Nat", "Mathlib"),
    ]
    baseline = PremiseBM25(records)
    assert lexical_tokens("Nat.add_comm") == ["nat", "add", "comm"]
    assert baseline.search(lexical_tokens("Nat.add_comm"), k=3)[0] == 17
    assert retrieval_metrics([baseline.search(["add"], k=2)], [[17]]).as_dict()["recall_at_200"] == 1.0
    assert baseline.search(["notincorpus"], k=10) == []


def test_oov_stats_reports_unknown_node_and_graph_rates() -> None:
    stats = _oov_stats(
        [Data(x=torch.tensor([0, 0, 1])), Data(x=torch.tensor([1]))],
        unknown_id=0,
    )
    assert stats == {
        "graphs": 2,
        "nodes": 4,
        "unknown_nodes": 2,
        "unknown_fraction": 0.5,
        "graphs_with_unknown": 1,
    }
