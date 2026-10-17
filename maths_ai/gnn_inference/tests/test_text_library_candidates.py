"""Frozen text retrieval remains bound to labels without old graph vectors."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import source_rows_digest
from maths_ai.gnn_inference.atp_lean_gnn.text_library_candidates import (
    SCHEMA, CachedTextLibraryCandidates,
)
from maths_ai.gnn_inference.scripts.prepare_text_library_candidates import _open_arrays


def _graph(row, *, nodes=3):
    graph = Data(
        x=torch.ones(nodes, dtype=torch.long),
        y=torch.tensor([1]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        premise_mask=torch.tensor([False] + [True] * (nodes - 1)),
        arg_node_indices=torch.tensor([-1]),
        arg_lemma_ids=torch.tensor([101]),
        citation_lemma_ids=torch.tensor([101]),
    )
    graph.row_index = row
    graph.split = "val"
    graph.dataset_name = "demo"
    graph.theorem = f"Demo.{row}"
    graph.tactic_raw = "exact Demo.b"
    graph.arg_count = graph.citation_count = 1
    graph.action_target_supported = True
    graph.arg_label_policy = "flat_reference_v2"
    return graph


def _cache(tmp_path):
    import faiss

    vectors = np.eye(3, dtype=np.float32)
    faiss_index = faiss.IndexFlatIP(3)
    faiss_index.add(vectors)
    index = LemmaIndex(
        faiss_index, [100, 101, 102], vectors,
        lemma_names=["Demo.a", "Demo.b", "Demo.c"],
        manifest={"model_type": "text", "encoder": "frozen"},
    )
    graphs = [_graph(7), _graph(9, nodes=5)]
    arrays = {
        "rows": np.asarray([7, 9], dtype=np.int64),
        "ids": np.asarray([[101, 100], [100, 102]], dtype=np.int32),
        "scores": np.asarray([[0.8, 0.5], [0.7, 0.4]], dtype=np.float32),
    }
    split = {"count": 2, "source_rows_sha256": source_rows_digest(graphs)}
    for field, values in arrays.items():
        path = tmp_path / f"val_{field}.npy"
        np.save(path, values)
        split[f"{field}_sha256"] = file_sha256(path)
    checkpoint = tmp_path / "snapshot.pt"
    checkpoint.write_bytes(b"frozen checkpoint")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "k": 2, "text_index_manifest": index.manifest,
        "text_checkpoint_sha256": file_sha256(checkpoint),
        "query_policy": "goal_first_v1", "selection_policy": "exact_cosine_no_gold_injection",
        "splits": {"val": split},
    }))
    return index, checkpoint, graphs


def test_text_only_candidates_bind_rows_and_gather_frozen_text_vectors(tmp_path):
    index, checkpoint, graphs = _cache(tmp_path)
    cache = CachedTextLibraryCandidates(tmp_path, index, checkpoint_path=checkpoint)
    cache.validate_dataset(graphs, split="val")
    batch = Batch.from_data_list(graphs)
    assert batch.row_index.tolist() == [7, 12]  # PyG increments by node count.
    ids, vectors, scores = cache.search(batch, k=2)
    assert ids == [[101, 100], [100, 102]]
    np.testing.assert_array_equal(vectors[0], index.lemma_vectors[[1, 0]])
    np.testing.assert_allclose(scores, [[0.8, 0.5], [0.7, 0.4]])
    assert cache.search(batch, k=1)[0] == [[101], [100]]
    with pytest.raises(ValueError, match="absent"):
        cache.search(Batch.from_data_list([_graph(8)]), k=1)


def test_text_only_candidates_reject_stale_labels_index_and_tables(tmp_path):
    index, checkpoint, graphs = _cache(tmp_path)
    cache = CachedTextLibraryCandidates(tmp_path, index, checkpoint_path=checkpoint)
    graphs[0].citation_lemma_ids = torch.tensor([100])
    with pytest.raises(ValueError, match="supervision differs"):
        cache.validate_dataset(graphs, split="val")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        CachedTextLibraryCandidates(tmp_path, index, checkpoint_path=checkpoint)
    index.manifest["encoder"] = "changed"
    with pytest.raises(ValueError, match="different text lemma index"):
        CachedTextLibraryCandidates(tmp_path, index)
    index.manifest["encoder"] = "frozen"
    np.save(tmp_path / "val_ids.npy", np.asarray([[102, 100], [100, 102]], dtype=np.int32))
    with pytest.raises(ValueError, match="file changed"):
        CachedTextLibraryCandidates(tmp_path, index)


def test_partial_arrays_resume_only_matching_shapes(tmp_path):
    arrays = _open_arrays(tmp_path, "val", count=2, k=3, resume=False)
    arrays["ids"][0] = [7, 8, 9]
    arrays["ids"].flush()
    resumed = _open_arrays(tmp_path, "val", count=2, k=3, resume=True)
    assert resumed["ids"][0].tolist() == [7, 8, 9]
    with pytest.raises(ValueError, match="shape or dtype changed"):
        _open_arrays(tmp_path, "val", count=2, k=4, resume=True)


def test_builder_caches_text_only_ranks_and_resumes_without_reencoding(tmp_path, monkeypatch):
    from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
    from maths_ai.gnn_inference.scripts import prepare_text_library_candidates as runner

    index, _, graphs = _cache(tmp_path)
    prepared = tmp_path / "prepared"
    (prepared / "packed/bidirectional").mkdir(parents=True)
    (prepared / "vocab").mkdir()
    for name in ("argument_labels.json", "packed/bidirectional/manifest.json",
                 "vocab/node_vocab.json", "vocab/tactic_vocab.json"):
        (prepared / name).write_text("{}")
    checkpoint = tmp_path / "text.pt"
    checkpoint.write_bytes(b"fixed epoch 5")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("frozen corpus")
    for graph in graphs:
        graph.dataset_name = "demo"
        graph.split = "val"

    class Rows(list):
        packed_cache_loaded = True

    class Model:
        def encode_states(self, tokens):
            return tokens["input_ids"]

    def text_rows(**kwargs):
        return iter(DatasetRow(
            state="p : Prop\n⊢ p", theorem=graph.theorem, tactic=graph.tactic_raw,
            split="val", row_index=graph.row_index, dataset_name="demo",
        ) for graph in graphs)

    monkeypatch.setattr(runner, "load_frozen_text_retriever", lambda *a, **k: (
        Model(), object(), index,
        {"epoch": 5, "dataset_name": "demo", "corpus_sha256": "fixed", "max_length": 16},
    ))
    monkeypatch.setattr(runner, "load_prepared_metadata", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner, "PreparedGraphDataset", lambda *a, **k: Rows(graphs))
    monkeypatch.setattr(runner, "iter_dataset_rows", text_rows)
    monkeypatch.setattr(runner, "tokenize_texts", lambda _, texts, **kwargs: {
        "input_ids": torch.tensor([[0.0, 1.0, 0.0]] * len(texts)),
    })
    output = tmp_path / "new_cache"
    argv = [
        "--checkpoint", str(checkpoint), "--text-index", str(tmp_path),
        "--prepared-root", str(prepared), "--corpus-path", str(corpus),
        "--output-dir", str(output), "--dataset-name", "demo", "--splits", "val", "--k", "2",
    ]
    assert runner.main(argv) == 0
    cache = CachedTextLibraryCandidates(output, index, checkpoint_path=checkpoint)
    cache.validate_dataset(graphs, split="val")
    assert cache.search(Batch.from_data_list(graphs), k=1)[0] == [[101], [101]]
    assert json.loads((output / "val_metrics.json").read_text())["recall_at_1"] == 1.0

    monkeypatch.setattr(runner, "tokenize_texts", lambda *a, **k: pytest.fail("completed rows must be reused"))
    assert runner.main([*argv, "--resume"]) == 0
