"""Mixed-source arguments supervise stop and GRU with actual lemma embeddings."""

import numpy as np
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    DecodedArgument, JointSourceDecoder, SourceCandidates,
    candidate_groups_from_batch, gold_arguments_from_batch,
)
from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidateSource
from maths_ai.gnn_inference.scripts.train_joint_source_decoder import _run_epoch


def _model():
    pointer = TacticWithArgsClassifier(
        num_node_labels=8, num_tactics=3, hidden_dim=8, num_layers=1, max_args=3,
    )
    return JointSourceDecoder(pointer, text_dim=4)


def _candidates():
    return SourceCandidates(
        (1, 2), torch.randn(2, 8),
        (100, 101), torch.randn(2, 4), torch.tensor([0.9, 0.4]),
    )


def test_mixed_teacher_forcing_uses_library_adapter_for_next_local_choice():
    torch.manual_seed(42)
    model = _model()
    state = torch.randn(8)
    candidates = _candidates()
    gold = [DecodedArgument(CandidateSource.LIBRARY, 101),
            DecodedArgument(CandidateSource.LOCAL, 2)]
    captured = []
    original = model.pointer.argument_selector.gru.forward

    def record_gru(value, hidden):
        captured.append(value)
        return original(value, hidden)

    model.pointer.argument_selector.gru.forward = record_gru
    loss, metrics = model.supervised_row(state, 1, candidates, gold)
    loss.backward()
    assert metrics["positions"] == 2
    assert metrics["local_positions"] == metrics["library_positions"] == 1
    torch.testing.assert_close(captured[0], model.lemma_adapter(candidates.lemma_vectors[1:2]))
    assert torch.equal(captured[1], candidates.local_vectors[1:2])
    assert model.lemma_adapter[0].weight.grad.abs().sum() > 0
    assert model.source_head.weight.grad.abs().sum() > 0
    # Learned text temperature changes probabilities, not frozen text order.
    _, _, library = model.score_step(model.initial_state(state, 1), candidates)
    assert library.argmax().item() == 0


def test_library_rank_can_depend_on_previous_gru_arguments():
    torch.manual_seed(14)
    model = _model()
    candidates = _candidates()
    hidden = model.initial_state(torch.randn(8), 1)
    _, _, frozen = model.score_step(hidden, candidates)
    assert frozen.argmax().item() == candidates.lemma_scores.argmax().item()
    with torch.no_grad():
        model.library_residual_scale.fill_(1.0)
    _, _, first = model.score_step(hidden, candidates)
    after_local = model.pointer.argument_selector.gru(candidates.local_vectors[0:1], hidden)
    _, _, second = model.score_step(after_local, candidates)
    assert not torch.allclose(first - second, torch.zeros_like(first))


def test_unretrieved_library_target_never_becomes_zero_argument_stop():
    model = _model()
    state = torch.randn(8)
    pool = _candidates()
    missing = [DecodedArgument(CandidateSource.LIBRARY, 999)]
    assert model.supervised_row(state, 1, pool, missing) is None
    zero_loss, metrics = model.supervised_row(state, 1, pool, [])
    assert zero_loss.isfinite() and metrics["positions"] == 0
    with pytest.raises(ValueError, match="budget"):
        model.supervised_row(state, 1, pool, missing * 4)


def _graph(row, *, local, lemma):
    graph = Data(
        x=torch.tensor([1, 2, 3]), node_type=torch.tensor([3, 0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        state_node_index=torch.tensor([0]), y=torch.tensor([1]),
        premise_mask=torch.tensor([False, True, True]),
        arg_node_indices=torch.tensor([local]), arg_lemma_ids=torch.tensor([lemma]),
    )
    graph.arg_count = 1
    graph.action_target_supported = True
    graph.row_index = row
    graph.split = "val"
    return graph


def test_pyg_local_offsets_and_ordered_mixed_sources():
    first = _graph(7, local=1, lemma=-1)
    first.arg_node_indices = torch.tensor([1, -1])
    first.arg_lemma_ids = torch.tensor([-1, 101])
    first.arg_count = 2
    second = _graph(9, local=-1, lemma=100)
    batch = Batch.from_data_list([first, second])
    groups = candidate_groups_from_batch(
        batch, torch.randn(batch.x.numel(), 8), [[100, 101], [101, 100]],
        np.random.default_rng(0).normal(size=(2, 2, 4)).astype(np.float32),
        np.asarray([[0.9, 0.8], [0.8, 0.6]], dtype=np.float32),
    )
    assert groups[0].local_ids == groups[1].local_ids == (1, 2)
    assert gold_arguments_from_batch(batch) == [
        [DecodedArgument(CandidateSource.LOCAL, 1), DecodedArgument(CandidateSource.LIBRARY, 101)],
        [DecodedArgument(CandidateSource.LIBRARY, 100)],
    ]
    assert groups[1].lemma_ids == (101, 100)
    assert groups[1].lemma_vectors.shape == (2, 4)


def test_train_and_validation_do_not_inject_missing_library(tmp_path):
    torch.manual_seed(42)
    model = _model()
    model.pointer.backbone.requires_grad_(False)
    graphs = [_graph(7, local=1, lemma=-1), _graph(9, local=-1, lemma=999)]

    class FrozenProvider:
        def search(self, batch, *, k):
            assert k == 2
            return (
                [[100, 101]] * int(batch.y.numel()),
                np.ones((int(batch.y.numel()), 2, 4), dtype=np.float32),
                np.asarray([[0.9, 0.4]] * int(batch.y.numel()), dtype=np.float32),
            )

    loader = DataLoader(graphs, batch_size=2)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    trained = _run_epoch(model, loader, FrozenProvider(), split="train", device=torch.device("cpu"),
                         k=2, optimizer=optimizer)
    assert trained["supported_rows"] == 2
    assert trained["reachable_rows"] == 1
    assert trained["missing_library_rows"] == 1
    evaluated = _run_epoch(model, loader, FrozenProvider(), split="val", device=torch.device("cpu"), k=2)
    assert evaluated["reachable_local_positions"] == 1
    assert evaluated["reachable_library_positions"] == 0
    assert evaluated["supported_rows"] == 2


def test_joint_runner_resumes_with_same_candidates_and_preserves_source_binding(tmp_path, monkeypatch):
    import faiss
    from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
    from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
    from maths_ai.gnn_inference.scripts import train_joint_source_decoder as runner

    prepared = tmp_path / "prepared"
    (prepared / "packed/bidirectional").mkdir(parents=True)
    (prepared / "vocab").mkdir()
    bindings = {
        "argument_label_binding_sha256": prepared / "argument_labels.json",
        "prepared_packed_manifest_sha256": prepared / "packed/bidirectional/manifest.json",
        "node_vocab_sha256": prepared / "vocab/node_vocab.json",
        "tactic_vocab_sha256": prepared / "vocab/tactic_vocab.json",
    }
    for path in bindings.values():
        path.write_text("{}")
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "manifest.json").write_text("{}")
    snapshot = tmp_path / "text.pt"
    snapshot.write_bytes(b"epoch 5")
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"new normalized baseline")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("test corpus")
    graphs = [_graph(7, local=1, lemma=-1), _graph(9, local=-1, lemma=101)]
    for graph in graphs:
        graph.split = "val"
    vectors = np.eye(4, dtype=np.float32)[:2]
    faiss_index = faiss.IndexFlatIP(4)
    faiss_index.add(vectors)
    index = LemmaIndex(faiss_index, [100, 101], vectors, manifest={"epoch": 5})
    metadata = SimpleNamespace(unknown_tactic_id=0)
    monkeypatch.setattr(runner, "load_prepared_metadata", lambda *a, **k: metadata)
    monkeypatch.setattr(runner, "load_frozen_text_retriever", lambda *a, **k: (
        None, None, index, {"epoch": 5, "corpus_sha256": "sha"},
    ))
    bound_files = {name: file_sha256(path) for name, path in bindings.items()}

    class Provider:
        def __init__(self, directory, frozen_index, *, checkpoint_path):
            assert frozen_index is index and checkpoint_path == snapshot
            self.manifest = {
                "prepared_root": str(prepared.resolve()), "k": 2, "text_epoch": 5,
                "corpus_sha256": "sha", "splits": {},
                **bound_files,
            }
            self.tables = {"train": None, "val": None}

        def validate_dataset(self, dataset, *, split):
            assert len(dataset) == 2

        def search(self, batch, *, k):
            assert k == 2
            size = int(batch.y.numel())
            return [[100, 101]] * size, np.tile(vectors, (size, 1, 1)), np.tile([0.8, 0.4], (size, 1))

    monkeypatch.setattr(runner, "CachedTextLibraryCandidates", Provider)
    monkeypatch.setattr(runner, "build_dataloaders", lambda *a, **k: (
        {split: graphs for split in ("train", "val")},
        {split: DataLoader(graphs, batch_size=2) for split in ("train", "val")},
    ))
    monkeypatch.setattr(runner, "build_pointer_model", lambda *a, **k: _model().pointer)
    monkeypatch.setattr(runner, "initialize_pointer_from_baseline_checkpoint", lambda *a, **k: {
        "source_checkpoint_sha256": file_sha256(baseline),
    })
    output = tmp_path / "joint"
    config = Path(__file__).parents[1] / "configs/pointer_gat_state_mean_attention_pretrained.json"
    argv = [
        "--prepared-root", str(prepared), "--candidate-dir", str(candidates),
        "--text-checkpoint", str(snapshot), "--text-index", str(tmp_path),
        "--corpus-path", str(corpus), "--baseline-checkpoint", str(baseline),
        "--pointer-config", str(config), "--run-dir", str(output),
        "--device", "cpu", "--epochs", "1", "--batch-size", "2", "--k", "2",
    ]
    assert runner.main(argv) == 0
    first = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1 and first["phase"] == "complete"
    assert first["val_metrics"]["reachable_library_positions"] == 1
    assert first["val_metrics"]["reachable_local_positions"] == 1
    assert first["val_metrics"]["supported_rows"] == 2
    assert (output / "best.pt").exists()
    assert runner.main([*argv[:argv.index("--epochs")], "--epochs", "2", *argv[argv.index("--batch-size"):], "--resume"]) == 0
    resumed = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    assert resumed["epoch"] == 2
    assert resumed["binding"] == first["binding"]
    with pytest.raises(ValueError, match="different data or training policy"):
        runner.main([*argv[:argv.index("--epochs")], "--epochs", "3", "--batch-size", "2",
                     "--k", "1", "--resume"])
    bindings["node_vocab_sha256"].write_text('{"FV0": 1}')
    with pytest.raises(ValueError, match="candidate source binding changed"):
        runner.main([*argv, "--resume"])
