"""Verify lexical retrieval against known token and ranking expectations."""

import torch
from torch_geometric.data import Batch, Data
from pathlib import Path
import json
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord
from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import retrieval_metrics
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_diagnostics import (
    PremiseBM25,
    TextMissAnalysis,
    hit_overlap,
    hybrid_top_k,
    lexical_tokens,
    load_cached_state_text,
    reciprocal_rank_fusion,
    text_graph_hit_overlap,
    text_miss_category,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import load_pointer_config
from maths_ai.gnn_inference.scripts.diagnose_retriever import (
    _diagnostic_config,
    _load_text_states,
    _oov_stats,
    _original_row_index,
)


def test_bm25_ranks_matching_names_and_types_on_original_ids() -> None:
    records = [
        LemmaRecord(17, "Nat.add_comm", "Nat.add a b = Nat.add b a", "Nat", "Mathlib"),
        LemmaRecord(99, "Set.union_subset", "Set.union A B ⊆ C", "Set", "Mathlib"),
        LemmaRecord(41, "Nat.mul_comm", "Nat.mul a b = Nat.mul b a", "Nat", "Mathlib"),
    ]
    baseline = PremiseBM25(records)
    assert lexical_tokens("Nat.add_comm") == ["nat", "add", "comm"]
    assert baseline.search(lexical_tokens("Nat.add_comm"), k=3)[0] == 17
    top_ids, rank = baseline.search_with_gold_rank(
        lexical_tokens("Nat.add_comm"), [17], k=2
    )
    assert top_ids == baseline.search(lexical_tokens("Nat.add_comm"), k=2)
    assert rank == 1
    assert baseline.search_with_gold_rank(["missing"], [17], k=2) == ([], None)
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


def test_diagnostic_preserves_packed_cache_for_graph_budget() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pointer_gat_state_mean_attention_pretrained.json"
    config = load_pointer_config(config_path)
    diagnostic = _diagnostic_config(config, "/tmp/prepared")
    assert diagnostic.training.max_batch_nodes == config.training.max_batch_nodes
    assert diagnostic.training.max_batch_edges == config.training.max_batch_edges
    assert diagnostic.training.cache_in_memory is True
    assert diagnostic.training.num_workers == 0


def test_fixed_budget_hybrid_retains_distinct_gnn_candidates() -> None:
    assert hybrid_top_k([1, 2, 3, 4], [2, 9, 10, 11], k=4, bm25_quota=2) == [1, 2, 9, 10]
    assert hybrid_top_k([1, 2, 3], [1, 2], k=4, bm25_quota=2) == [1, 2, 3]
    assert len(hybrid_top_k(list(range(200)), list(range(100, 300)), k=200, bm25_quota=150)) == 200


def test_rank_fusion_and_hit_overlap_use_same_labeled_rows() -> None:
    assert reciprocal_rank_fusion([1, 2], [2, 3], k=3)[0] == 2
    assert hit_overlap(
        [[1], [2], [3], [4], [5]],
        [[1], [8], [9], [7], [5]],
        [[1], [2], [9], [6], []],
    ) == {"both": 1, "bm25_only": 1, "gnn_only": 1, "neither": 1}


def test_gold_rank_matches_full_bm25_order_with_ties_and_multiple_positives() -> None:
    records = [
        LemmaRecord(index, f"lemma_{index}", "common" if index < 300 else "rare", "", "")
        for index in range(320)
    ]
    bm25 = PremiseBM25(records)
    for positives in ([275], [310], [275, 310], [999]):
        top, rank = bm25.search_with_gold_rank(["common"], positives, k=10)
        full = bm25.search(["common"], k=320)
        assert top == full[:10]
        expected = next(
            (i for i, lemma_id in enumerate(full, start=1) if lemma_id in positives),
            None,
        )
        assert rank == expected
    assert bm25.search_with_gold_rank(["rare"], [315], k=10)[1] == 16


def test_raw_text_join_validates_original_row_identity(tmp_path) -> None:
    data = SimpleNamespace(
        row_index=7, theorem="Nat.demo", tactic_raw="rw [Nat.add_comm]",
        dataset_name="demo/dataset",
    )
    cache = tmp_path / "val" / "sexpr"
    cache.mkdir(parents=True)
    path = cache / "000000007.json"
    payload = {
        "schema_version": 4, "dataset": data.dataset_name,
        "split": "val", "row_index": 7, "theorem": data.theorem,
        "tactic": data.tactic_raw, "text_state": "n : Nat\n⊢ n + 0 = n",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_cached_state_text(tmp_path, data) == payload["text_state"]
    args = SimpleNamespace(text_source="raw-cache", prepared_root=str(tmp_path))
    assert _load_text_states(args, [data]) == {7: payload["text_state"]}
    payload["tactic"] = "exact Nat.add_comm"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs at tactic"):
        load_cached_state_text(tmp_path, data)


def test_missing_raw_text_cache_fails_with_upstream_fallback(tmp_path) -> None:
    data = SimpleNamespace(row_index=2)
    with pytest.raises(FileNotFoundError, match="--text-source upstream"):
        load_cached_state_text(tmp_path, data)


def test_upstream_text_join_checks_row_identity(monkeypatch) -> None:
    from maths_ai.gnn_inference.scripts import diagnose_retriever

    data = SimpleNamespace(
        row_index=7, theorem="Nat.demo", tactic_raw="rw [Nat.add_comm]",
        dataset_name="demo/dataset",
    )
    row = SimpleNamespace(
        row_index=7, theorem=data.theorem, tactic=data.tactic_raw,
        dataset_name=data.dataset_name, state="⊢ Nat.add n m = Nat.add m n",
    )
    monkeypatch.setattr(
        diagnose_retriever, "iter_dataset_rows",
        lambda **kwargs: iter([SimpleNamespace(row_index=0), row]),
    )
    args = SimpleNamespace(text_source="upstream", dataset_name="demo/dataset")
    assert _load_text_states(args, [data]) == {7: row.state}
    row.tactic = "exact Nat.add_comm"
    with pytest.raises(ValueError, match="does not match prepared graph"):
        _load_text_states(args, [data])


def test_original_row_index_undoes_pyg_node_offsets() -> None:
    graphs = [Data(x=torch.zeros(size, dtype=torch.long)) for size in (3, 5, 7)]
    for graph, original in zip(graphs, (33, 34, 150)):
        graph.row_index = original
    batch = Batch.from_data_list(graphs)
    assert batch.row_index.tolist() == [33, 37, 158]
    assert [_original_row_index(batch, row) for row in range(3)] == [33, 34, 150]


def test_text_graph_overlap_names_which_query_hit() -> None:
    assert text_graph_hit_overlap(
        [[1], [2], [3], [4]],
        [[1], [8], [9], [7]],
        [[1], [2], [9], [6]],
    ) == {"both": 1, "text_only": 1, "graph_only": 1, "neither": 1}


def test_text_failure_categories_preserve_multi_positive_term_overlap() -> None:
    records = [
        LemmaRecord(1, "Set.union", "Nat.lt x y", "", ""),
        LemmaRecord(2, "Nat.add_comm", "List.append a b", "", ""),
    ]
    category = text_miss_category(201, {"nat", "append"}, records)
    assert category == {
        "rank_bucket": "rank_201_to_1000",
        "query_token_count": 2,
        "shared_name_terms": ["nat"],
        "shared_type_terms": ["append", "nat"],
    }
    assert text_miss_category(None, {"unrelated"}, records)["rank_bucket"] == "no_lexical_overlap"
    with pytest.raises(ValueError, match="miss at K=200"):
        text_miss_category(200, {"nat"}, records)


def test_text_failure_analysis_counts_labeled_rows_and_samples_by_bucket() -> None:
    record = LemmaRecord(1, "Nat.add_comm", "Nat.add x y", "Nat", "Mathlib")
    analysis = TextMissAnalysis(seed=42, examples_per_bucket=1)
    for row, rank in enumerate((1, 201, 1000, 1001, 5000, 5001, None, None)):
        analysis.record(
            rank=rank, query_tokens={"nat"} if rank is not None else {"unrelated"},
            gold_records=[record], row_index=row, theorem=f"Nat.test{row}",
            tactic_name="rw", tactic_raw="rw [Nat.add_comm]",
            state_text="⊢ Nat.add x y", top_names=["Nat.succ_eq_add_one"],
        )
    report = analysis.as_dict()
    assert report["labeled_query_count"] == 8
    assert report["miss_at_200_count"] == 7
    assert report["rank_buckets"] == {
        "rank_201_to_1000": 2,
        "rank_1001_to_5000": 2,
        "rank_above_5000": 1,
        "no_lexical_overlap": 2,
    }
    assert report["miss_shared_term_counts"] == {
        "shared_name": 5, "shared_type": 5, "neither": 2,
    }
    assert report["per_tactic"]["rw"] == {"labeled": 8, "missed_at_200": 7}
    assert all(len(examples) == 1 for examples in report["examples_by_bucket"].values())
