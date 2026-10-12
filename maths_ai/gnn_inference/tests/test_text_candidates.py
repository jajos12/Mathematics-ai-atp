"""Frozen text IDs must preserve source coordinates and pointer-space vectors."""

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex, state_dict_sha256
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import (
    CachedTextCandidates, citation_coverage, load_frozen_text_retriever, source_rows_digest,
)
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord
from maths_ai.gnn_inference.atp_lean_gnn.retriever_experiments import encode_corpus_index
from maths_ai.gnn_inference.atp_lean_gnn.text_retriever import TextDualEncoder
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.premise_training import (
    evaluate_model_with_premises, train_one_epoch_with_premises,
)
from maths_ai.gnn_inference.atp_lean_gnn.unified_reranking import verify_unified_reranker_dependencies


def _graph(row, nodes=3):
    data = Data(
        x=torch.ones(nodes, dtype=torch.long), node_type=torch.zeros(nodes, dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]]), y=torch.tensor([1]),
        premise_mask=torch.tensor([False] + [True] * (nodes - 1)),
        state_node_index=torch.tensor([0]),
        arg_node_indices=torch.tensor([-1]), arg_lemma_ids=torch.tensor([101]),
    )
    data.row_index = row
    data.split = "val"
    data.dataset_name = "demo"
    data.theorem = f"Demo.{row}"
    data.tactic_raw = "exact Demo.b"
    data.tactic_name = "exact"
    data.arg_count = 1
    return data


def _provider(tmp_path, *, selected=None):
    import faiss

    vectors = np.eye(8, dtype=np.float32)[:2]
    index = faiss.IndexFlatIP(8)
    index.add(vectors)
    pointer = LemmaIndex(
        index, [100, 101], vectors, lemma_names=["Demo.a", "Demo.b"],
        normalize_queries=True, manifest={"encoder_state_sha256": "pointer"},
    )
    graphs = [_graph(7), _graph(9, nodes=5)]
    selected = selected if selected is not None else [[101], [100]]
    arrays = {"rows": np.asarray([7, 9]), "ids": np.asarray(selected),
              "scores": np.ones((2, 1), dtype=np.float32)}
    metadata = {"source_rows_sha256": source_rows_digest(graphs), "count": 2}
    for name, array in arrays.items():
        path = tmp_path / f"val_{name}.npy"
        np.save(path, array)
        metadata[f"{name}_sha256"] = file_sha256(path)
    manifest = {
        "schema": "text_candidates_v1", "k": 1,
        "pointer_index_manifest": pointer.manifest,
        "text_checkpoint_sha256": "snapshot", "splits": {"val": metadata},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return CachedTextCandidates(tmp_path, pointer), pointer, graphs


def test_all_known_citation_coverage_is_not_any_positive_recall():
    metrics = citation_coverage([[10, 11], [11, 12], [13]], [[10, 12], [11], []], ks=(1, 2))
    assert metrics["labeled_query_count"] == 2
    assert metrics["recall_at_2"] == 1.0
    assert metrics["all_known_citations_at_2"] == 0.5
    assert metrics["citation_micro_recall_at_2"] == pytest.approx(2 / 3)


def test_cached_text_ids_gather_pointer_vectors_and_recover_pyg_rows(tmp_path):
    provider, pointer, graphs = _provider(tmp_path)
    provider.validate_dataset(graphs, split="val")
    batch = Batch.from_data_list(graphs)
    assert batch.row_index.tolist() == [7, 12]
    ids, vectors, _ = provider.search(batch, k=1)
    assert ids == [[101], [100]]
    assert vectors.shape == (2, 1, 8)
    np.testing.assert_array_equal(vectors[0, 0], pointer.lemma_vectors[1])
    graphs[0].tactic_raw = "exact Demo.a"
    with pytest.raises(ValueError, match="rows differ"):
        provider.validate_dataset(graphs, split="val")


def test_cached_text_candidates_reject_modified_table_or_index(tmp_path):
    provider, pointer, _ = _provider(tmp_path)
    pointer.manifest["encoder_state_sha256"] = "wrong"
    with pytest.raises(ValueError, match="different pointer-vector"):
        CachedTextCandidates(tmp_path, pointer)
    pointer.manifest["encoder_state_sha256"] = "pointer"
    np.save(tmp_path / "val_ids.npy", np.asarray([[100], [100]]))
    with pytest.raises(ValueError, match="file changed"):
        CachedTextCandidates(tmp_path, pointer)


def test_text_scorer_uses_cached_pool_with_no_gold_injection_during_validation(tmp_path):
    provider, pointer, graphs = _provider(tmp_path, selected=[[100], [100]])
    model = TacticWithArgsClassifier(
        num_node_labels=4, num_tactics=4, hidden_dim=8, num_layers=1, max_args=2,
    )
    model.backbone.requires_grad_(False)
    scorer = PremiseScorer(8)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    pointer.search = lambda *args, **kwargs: pytest.fail("pointer must not retrieve candidates")
    loader = DataLoader(graphs, batch_size=2)
    trained = train_one_epoch_with_premises(
        model, scorer, loader, pointer, candidate_provider=provider,
        optimizer=optimizer, grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        device=torch.device("cpu"), grad_clip=1.0, unknown_tactic_id=0,
        arg_loss_weight=0.5, premise_loss_weight=1.0, k=1,
        epoch=1, total_epochs=1, log_every_batches=1, use_amp=False,
    )
    assert trained["reranking_target_coverage"] == 1.0
    evaluated = evaluate_model_with_premises(
        model, scorer, loader, pointer, candidate_provider=provider,
        device=torch.device("cpu"), unknown_tactic_id=0,
        arg_loss_weight=0.5, premise_loss_weight=1.0, k=1,
        tactic_vocab={"<UNK>": 0, "exact": 1, "apply": 2, "rw": 3},
    )
    assert evaluated["reranking_target_coverage"] == 0.0
    assert evaluated["complete_action_exact_match_oracle_tactic"] == 0.0
    checkpoint = {"lemma_index_manifest": pointer.manifest,
                  "candidate_manifest": provider.manifest, "retriever_state_sha256": None}
    with pytest.raises(ValueError, match="requires its frozen text"):
        verify_unified_reranker_dependencies(checkpoint, index_manifest=pointer.manifest, retriever=None)
    verify_unified_reranker_dependencies(
        checkpoint, index_manifest=pointer.manifest, retriever=None, candidate_provider=provider,
    )


def test_text_snapshot_rejects_stale_lemma_index(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    vocab = {"nat": 1}
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: SimpleNamespace(
        get_vocab=lambda: vocab,
    ))
    monkeypatch.setattr(TextDualEncoder, "from_pretrained", lambda *a, **k: TextDualEncoder(
        torch.nn.Linear(8, 8),
    ))
    model = TextDualEncoder(torch.nn.Linear(8, 8))
    corpus = tmp_path / "lemmas.jsonl"
    corpus.write_text("fixture corpus")
    tokenizer_sha = hashlib.sha256(json.dumps(vocab, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    saved = {
        "model_type": "pretrained_text_dual_encoder", "phase": "complete",
        "model_name": "tiny", "epoch": 3,
        "corpus_sha256": file_sha256(corpus), "tokenizer_sha256": tokenizer_sha,
        "state_text_policy": "goal_first_v1", "max_length": 32,
        "model_state_dict": model.state_dict(),
    }
    snapshot = tmp_path / "snapshot.pt"
    torch.save(saved, snapshot)
    manifest = {
        "model_type": "pretrained_text_dual_encoder_index_v1", "model_name": "tiny",
        "corpus_sha256": file_sha256(corpus), "tokenizer_sha256": tokenizer_sha,
        "state_text_policy": "goal_first_v1", "max_length": 32, "normalize": True,
        "lemma_encoder_sha256": state_dict_sha256(model.lemma_encoder.state_dict()),
    }
    index_dir = tmp_path / "index"
    encode_corpus_index(
        [LemmaRecord(100, "Demo.a", "a", "", "")],
        lambda ids: torch.ones(len(ids), 8), batch_size=1,
        manifest=manifest, output_dir=index_dir,
    )
    loaded, _, _, _ = load_frozen_text_retriever(snapshot, index_dir, corpus, device=torch.device("cpu"))
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    with torch.no_grad():
        model.lemma_encoder.weight.add_(1)
    saved["model_state_dict"] = model.state_dict()
    torch.save(saved, snapshot)
    with pytest.raises(ValueError, match="does not match frozen"):
        load_frozen_text_retriever(snapshot, index_dir, corpus, device=torch.device("cpu"))


def test_scorer_cli_resumes_validation_without_retraining_completed_epoch(tmp_path, monkeypatch):
    from maths_ai.gnn_inference.scripts import train_text_candidate_scorer as script
    from maths_ai.gnn_inference.atp_lean_gnn.training import PointerConfig

    provider, pointer, val_graphs = _provider(tmp_path)
    train_graphs = [_graph(7), _graph(9, nodes=5)]
    for data in train_graphs:
        data.split = "train"
    manifest = provider.manifest
    train_meta = {"count": 2, "source_rows_sha256": source_rows_digest(train_graphs)}
    for field in ("rows", "ids", "scores"):
        path = tmp_path / f"train_{field}.npy"
        np.save(path, np.load(tmp_path / f"val_{field}.npy"))
        train_meta[f"{field}_sha256"] = file_sha256(path)
    manifest["splits"]["train"] = train_meta
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    config = PointerConfig.from_dict({
        "prepared_root": str(tmp_path), "device": "cpu", "max_args": 2,
        "model": {"hidden_dim": 8, "num_layers": 1},
    })
    model = TacticWithArgsClassifier(
        num_node_labels=4, num_tactics=4, hidden_dim=8, num_layers=1, max_args=2,
    )
    checkpoint = tmp_path / "pointer.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    monkeypatch.setattr(script, "load_pointer_config", lambda *a, **k: config)
    monkeypatch.setattr(script, "build_pointer_model", lambda *a: model)
    metadata = SimpleNamespace(
        node_vocab={"<UNK>": 0, "State": 1, "A": 2, "B": 3},
        tactic_vocab={"<UNK>": 0, "exact": 1, "apply": 2, "rw": 3}, unknown_tactic_id=0,
    )
    monkeypatch.setattr(script, "load_prepared_metadata", lambda *a: metadata)
    monkeypatch.setattr(script, "load_index_for_encoder", lambda *a, **k: pointer)
    datasets = {"train": train_graphs, "val": val_graphs}
    monkeypatch.setattr(script, "build_dataloaders", lambda *a, **k: (
        datasets, {split: DataLoader(rows, batch_size=2) for split, rows in datasets.items()},
    ))
    evaluate = script.evaluate_model_with_premises
    monkeypatch.setattr(script, "evaluate_model_with_premises", lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    run = tmp_path / "run"
    argv = [
        "--config", str(tmp_path / "config.json"), "--prepared-root", str(tmp_path),
        "--checkpoint", str(checkpoint), "--index-path", str(tmp_path),
        "--corpus-path", str(tmp_path / "corpus.jsonl"), "--candidate-dir", str(tmp_path),
        "--run-dir", str(run), "--device", "cpu", "--batch-size", "2", "--k", "1",
    ]
    with pytest.raises(KeyboardInterrupt):
        script.main(argv)
    assert torch.load(run / "last.pt", weights_only=False)["phase"] == "train_done"
    monkeypatch.setattr(script, "evaluate_model_with_premises", evaluate)
    monkeypatch.setattr(script, "train_one_epoch_with_premises", lambda *a, **k: pytest.fail("must not repeat training"))
    script.main([*argv, "--resume"])
    assert torch.load(run / "last.pt", weights_only=False)["phase"] == "complete"
    assert (run / "metrics_epoch_1.json").is_file()


def test_candidate_preparation_cli_resumes_mid_split(tmp_path, monkeypatch):
    import faiss
    from maths_ai.gnn_inference.scripts import prepare_text_candidates as script
    from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow

    vectors = np.random.default_rng(2).normal(size=(300, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatIP(8)
    faiss_index.add(vectors)
    index = LemmaIndex(
        faiss_index, list(range(300)), vectors,
        lemma_names=[f"Demo.{i}" for i in range(300)], normalize_queries=True,
        manifest={"normalize": True},
    )
    checkpoint = tmp_path / "text.pt"
    checkpoint.write_bytes(b"frozen snapshot")
    pointer = tmp_path / "pointer.pt"
    torch.save({"model_state_dict": {}}, pointer)
    model = SimpleNamespace(encode_states=lambda tokens: torch.ones(len(tokens), 8) / np.sqrt(8))
    saved = {"epoch": 3, "max_length": 32, "dataset_name": "demo", "corpus_sha256": "corpus"}
    monkeypatch.setattr(script, "load_frozen_text_retriever", lambda *a, **k: (model, None, index, saved))
    monkeypatch.setattr(script, "load_index_for_encoder", lambda *a, **k: index)
    monkeypatch.setattr(script, "load_pointer_config", lambda *a: SimpleNamespace(edge_mode="bidirectional"))
    monkeypatch.setattr(script, "load_prepared_metadata", lambda path: SimpleNamespace(
        root=Path(path), node_vocab={}, tactic_vocab={},
    ))

    class Prepared(list):
        def __init__(self, metadata, *, split, **kwargs):
            super().__init__([_graph(7), _graph(9)])
            for data in self:
                data.split = split
                if metadata.root.name == "repaired":
                    data.arg_label_policy = "flat_reference_v2"
                    data.action_target_supported = True
                    data.citation_lemma_ids = torch.tensor([101])
                    data.citation_count = 1

    monkeypatch.setattr(script, "PreparedGraphDataset", Prepared)
    monkeypatch.setattr(script, "iter_dataset_rows", lambda *, dataset_name, split: iter([
        DatasetRow(
            state="⊢ P", theorem=f"Demo.{row}", tactic="exact Demo.b",
            split=split, row_index=row, dataset_name="demo",
        ) for row in (7, 9)
    ]))
    calls = []

    def tokenize(tokenizer, texts, **kwargs):
        calls.extend(texts)
        if len(calls) == 2:
            raise RuntimeError("pod restarted")
        return texts

    monkeypatch.setattr(script, "tokenize_texts", tokenize)
    output = tmp_path / "candidates"
    argv = [
        "--checkpoint", str(checkpoint), "--text-index", str(tmp_path),
        "--pointer-checkpoint", str(pointer), "--pointer-index", str(tmp_path),
        "--config", str(tmp_path / "config"), "--prepared-root", str(tmp_path),
        "--corpus-path", str(tmp_path / "corpus"), "--output-dir", str(output),
        "--dataset-name", "demo", "--batch-size", "1",
    ]
    with pytest.raises(RuntimeError, match="pod restarted"):
        script.main(argv)
    assert json.loads((output / "val_progress.json").read_text())["next_row"] == 1
    assert not (output / "manifest.json").exists()
    script.main([*argv, "--resume"])
    assert len(calls) == 5  # failed batch + its retry, not replay of completed batch
    provider = CachedTextCandidates(output, index)
    assert provider.manifest["splits"]["val"]["retrieval_metrics"]["labeled_query_count"] == 2
    assert provider.search(Batch.from_data_list([_graph(9)]), k=200)[1].shape == (1, 200, 8)
    repaired = tmp_path / "repaired"
    repaired.mkdir()
    (repaired / "argument_labels.json").write_text(json.dumps({"source_root": str(tmp_path)}))
    reused_argv = argv.copy()
    reused_argv[reused_argv.index("--prepared-root") + 1] = str(repaired)
    reused_argv[reused_argv.index("--output-dir") + 1] = str(tmp_path / "rebound")
    script.main([*reused_argv, "--reuse-candidate-dir", str(output)])
    assert len(calls) == 5  # ranks are reused, with no model query encoding
    rebound = CachedTextCandidates(tmp_path / "rebound", index)
    rebound.validate_dataset(Prepared(SimpleNamespace(root=repaired), split="val"), split="val")
    np.testing.assert_array_equal(rebound.tables["val"]["ids"], provider.tables["val"]["ids"])


def test_unsupported_rows_are_skipped_in_training_and_counted_as_validation_misses(tmp_path):
    provider, pointer, graphs = _provider(tmp_path)
    for row, data in enumerate(graphs):
        data.action_target_supported = row == 1
        data.arg_count = 0
        data.arg_node_indices = torch.empty(0, dtype=torch.long)
        data.arg_lemma_ids = torch.empty(0, dtype=torch.long)
    model = TacticWithArgsClassifier(
        num_node_labels=4, num_tactics=4, hidden_dim=8, num_layers=1, max_args=2,
    )
    model.backbone.requires_grad_(False)
    scorer = PremiseScorer(8)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    trained = train_one_epoch_with_premises(
        model, scorer, DataLoader(graphs, batch_size=2), pointer, candidate_provider=provider,
        optimizer=optimizer, grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        device=torch.device("cpu"), grad_clip=1.0, unknown_tactic_id=0,
        arg_loss_weight=0.5, premise_loss_weight=1.0, k=1,
        epoch=1, total_epochs=1, log_every_batches=1, use_amp=False,
    )
    assert trained["example_count"] == 1
    assert trained["unsupported_example_count"] == 1
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(20)
        model.backbone.classifier.weight.zero_()
        model.backbone.classifier.bias.fill_(-20)
        model.backbone.classifier.bias[1] = 20
    metrics = evaluate_model_with_premises(
        model, scorer, DataLoader(graphs, batch_size=2), pointer, candidate_provider=provider,
        device=torch.device("cpu"), unknown_tactic_id=0,
        arg_loss_weight=0.5, premise_loss_weight=1.0, k=1,
        tactic_vocab={"<UNK>": 0, "exact": 1, "apply": 2, "rw": 3},
    )
    assert metrics["evaluated_count"] == 2
    assert metrics["complete_action_exact_match_predicted_tactic"] == 0.5
    assert metrics["supported_action_exact_match_predicted_tactic"] == 1.0
    assert metrics["action_target_unsupported_count"] == 1
