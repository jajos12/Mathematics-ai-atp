"""Library-only updates preserve local weights and never inject missing gold."""

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import JointSourceDecoder
from maths_ai.gnn_inference.atp_lean_gnn.training import GraphBudgetBatchSampler, PointerConfig
from maths_ai.gnn_inference.scripts import train_library_identity as runner


def _graph(row, locals_, libraries, *, split="val"):
    graph = Data(
        x=torch.tensor([1, 2, 3]), node_type=torch.tensor([3, 0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 2]]), y=torch.tensor([1]),
        state_node_index=torch.tensor([0]), premise_mask=torch.tensor([False, True, True]),
        arg_node_indices=torch.tensor(locals_, dtype=torch.long),
        arg_lemma_ids=torch.tensor(libraries, dtype=torch.long),
    )
    graph.row_index = row
    graph.arg_count = len(locals_)
    graph.action_target_supported = True
    graph.split = split
    return graph


def _model():
    return JointSourceDecoder(TacticWithArgsClassifier(
        num_node_labels=8, num_tactics=3, hidden_dim=8, num_layers=1, max_args=3,
    ), text_dim=4)


def _provider():
    class Provider:
        manifest = {"k": 2}

        def search(self, batch, *, k):
            assert k == 2
            size = int(batch.y.numel())
            return ([[100, 101]] * size,
                    np.tile(np.eye(4, dtype=np.float32)[:2], (size, 1, 1)),
                    np.tile(np.array([0.8, 0.4], dtype=np.float32), (size, 1)))
    return Provider()


def test_missing_library_prefix_teaches_later_retrieved_identity_without_injection():
    torch.manual_seed(215)
    graphs = [_graph(7, [-1, -1], [999, 100]),
              _graph(9, [2, -1], [-1, 101]), _graph(11, [-1], [999])]
    assert runner._library_rows(graphs) == [0, 1, 2]
    sampler = GraphBudgetBatchSampler([(3, 2)] * 3, max_graphs=3)
    loader = DataLoader(graphs, batch_sampler=sampler)
    index = SimpleNamespace(id_to_position={100: 0, 101: 1, 999: 2},
                            lemma_vectors=np.eye(4, dtype=np.float32)[:3])
    model = _model()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    model.enable_library_identity()
    model.requires_grad_(False)
    model.library_query.requires_grad_(True)
    captured = []
    score_step = model.score_step

    def capture(hidden, pool, **kwargs):
        captured.append(pool.lemma_ids)
        return score_step(hidden, pool, **kwargs)

    model.score_step = capture
    optimizer = torch.optim.AdamW(model.library_query.parameters(), lr=0.01)
    metrics = runner._run_epoch(
        model, loader, sampler, _provider(), index, device=torch.device("cpu"),
        epoch=1, split="train", k=2, optimizer=optimizer,
    )
    assert captured == [(100, 101), (100, 101)]
    assert metrics["all_library_positions"] == 4
    assert metrics["missing_library_positions"] == 2
    assert metrics["retrieved_positions"] == 2
    assert metrics["matched_positions"] == 1
    assert metrics["slot_2_retrieved_positions"] == 2
    assert metrics["mean_library_loss"] > 0
    assert model.library_query.weight.abs().sum() > 0
    for key, value in original.items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


def test_missing_gold_has_no_identity_loss_and_unsupported_rows_are_excluded():
    graphs = [_graph(7, [-1], [999]), _graph(9, [-1], [100])]
    graphs[1].action_target_supported = False
    assert runner._library_rows(graphs) == [0]
    sampler = GraphBudgetBatchSampler([(3, 2)], max_graphs=1)
    model = _model()
    model.enable_library_identity()
    model.requires_grad_(False)
    model.library_query.requires_grad_(True)
    before = model.library_query.weight.detach().clone()
    metrics = runner._run_epoch(
        model, DataLoader(graphs[:1], batch_sampler=sampler), sampler, _provider(),
        SimpleNamespace(id_to_position={999: 0}, lemma_vectors=np.eye(4, dtype=np.float32)[:1]),
        device=torch.device("cpu"), epoch=1, split="train", k=2,
        optimizer=torch.optim.AdamW(model.library_query.parameters()),
    )
    assert metrics["missing_library_positions"] == 1
    assert metrics["mean_library_loss"] == 0
    torch.testing.assert_close(before, model.library_query.weight, rtol=0, atol=0)


def test_mid_epoch_resume_matches_uninterrupted_query_training():
    torch.manual_seed(217)
    graphs = [_graph(i, [-1], [100 if i % 2 else 101], split="train") for i in range(51)]
    sampler = GraphBudgetBatchSampler([(3, 2)] * len(graphs), max_graphs=1,
                                      shuffle=True, seed=42)
    loader = DataLoader(graphs, batch_sampler=sampler)
    index = SimpleNamespace(id_to_position={100: 0, 101: 1},
                            lemma_vectors=np.eye(4, dtype=np.float32)[:2])
    original = _model().state_dict()

    def initialize():
        model = _model()
        model.load_state_dict(original, strict=True)
        model.enable_library_identity()
        model.requires_grad_(False)
        model.library_query.requires_grad_(True)
        return model, torch.optim.AdamW(model.library_query.parameters(), lr=0.01)

    model, optimizer = initialize()
    expected = runner._run_epoch(model, loader, sampler, _provider(), index,
                                device=torch.device("cpu"), epoch=1, split="train",
                                k=2, optimizer=optimizer)
    expected_state = deepcopy(model.state_dict())
    model, optimizer = initialize()
    checkpoint = {}

    def interrupt(next_batch, counts):
        checkpoint.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
                          next_batch=next_batch, counts=counts)
        raise InterruptedError("test checkpoint interruption")

    with pytest.raises(InterruptedError):
        runner._run_epoch(model, loader, sampler, _provider(), index,
                          device=torch.device("cpu"), epoch=1, split="train",
                          k=2, optimizer=optimizer, on_progress=interrupt)
    restored, restored_optimizer = initialize()
    restored.load_state_dict(checkpoint["model"], strict=True)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    actual = runner._run_epoch(
        restored, loader, sampler, _provider(), index, device=torch.device("cpu"),
        epoch=1, split="train", k=2, optimizer=restored_optimizer,
        start_batch=checkpoint["next_batch"], initial_counts=checkpoint["counts"],
    )
    assert actual == expected
    for key, value in expected_state.items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_runner_preserves_epoch_zero_and_resume_binding(tmp_path, monkeypatch):
    torch.manual_seed(216)
    prepared = tmp_path / "prepared"
    paths = {
        "argument_label_binding_sha256": prepared / "argument_labels.json",
        "prepared_packed_manifest_sha256": prepared / "packed/bidirectional/manifest.json",
        "node_vocab_sha256": prepared / "vocab/node_vocab.json",
        "tactic_vocab_sha256": prepared / "vocab/tactic_vocab.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    text = tmp_path / "text.pt"
    text.write_bytes(b"epoch 5")
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "manifest.json").write_text("{}")
    index = SimpleNamespace(manifest={"epoch": 5}, id_to_position={100: 0, 101: 1, 999: 2},
                            lemma_vectors=np.eye(4, dtype=np.float32)[:3])
    config = PointerConfig.from_dict({
        "prepared_root": str(prepared), "device": "cpu", "max_args": 3,
        "model": {"hidden_dim": 8, "num_layers": 1},
    })
    binding = {
        "prepared_root": str(prepared.resolve()), "text_index_manifest": index.manifest,
        "text_checkpoint_sha256": file_sha256(text),
        "candidate_manifest_sha256": file_sha256(candidates / "manifest.json"),
    }
    source = _model()
    joint, local = tmp_path / "joint.pt", tmp_path / "local.pt"
    torch.save({"model_type": "joint_source_flat_decoder_v2", "binding": binding,
                "config": config.to_dict(), "model_state_dict": source.state_dict()}, joint)
    torch.save({"model_type": "frozen_graph_local_identity_v1",
                "binding": {**binding, "joint_checkpoint_sha256": file_sha256(joint)},
                "model_state_dict": source.state_dict()}, local)

    class Rows(list):
        def graph_sizes(self):
            return [(3, 2)] * len(self)

    datasets = {split: Rows([_graph(7, [-1, -1], [999, 100], split=split),
                             _graph(9, [2, -1], [-1, 101], split=split)])
                for split in ("train", "val")}

    class Provider:
        def __init__(self, *args, **kwargs):
            self.manifest = {"k": 2, **{key: file_sha256(path) for key, path in paths.items()}}

        def validate_dataset(self, dataset, *, split):
            assert dataset is datasets[split]

        search = staticmethod(_provider().search)

    monkeypatch.setattr(runner, "LemmaIndex", SimpleNamespace(load=lambda _: index))
    monkeypatch.setattr(runner, "CachedTextLibraryCandidates", Provider)
    monkeypatch.setattr(runner, "load_prepared_metadata", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner, "PreparedGraphDataset", lambda *a, split, **k: datasets[split])
    monkeypatch.setattr(runner, "build_pointer_model", lambda *a, **k: _model().pointer)
    run = tmp_path / "run"
    argv = [
        "--prepared-root", str(prepared), "--joint-checkpoint", str(joint),
        "--local-checkpoint", str(local), "--text-checkpoint", str(text),
        "--text-index", str(tmp_path), "--candidate-dir", str(candidates),
        "--run-dir", str(run), "--device", "cpu", "--epochs", "1",
        "--batch-size", "2", "--k", "2", "--expected-val-library", "3",
        "--expected-val-matched", "1",
    ]
    assert runner.main(argv) == 0
    baseline = json.loads((run / "metrics_epoch_0.json").read_text())
    assert baseline["all_library_positions"] == 3
    assert baseline["retrieved_positions"] == 2
    assert baseline["matched_positions"] == 1
    first = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 1 and first["phase"] == "complete"
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, first["model_state_dict"][key], rtol=0, atol=0)
    assert runner.main([*argv[:argv.index("--epochs")], "--epochs", "2",
                        *argv[argv.index("--batch-size"):], "--resume"]) == 0
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2 and saved["binding"] == first["binding"]
    best = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    assert best["val_metrics"]["matched_top1"] >= baseline["matched_top1"]
    with pytest.raises(ValueError, match="changed"):
        runner.main([*argv, "--learning-rate", "0.1", "--resume"])
    with pytest.raises(ValueError, match="cohort changed"):
        runner.main([*argv, "--expected-val-matched", "9", "--resume"])
    paths["node_vocab_sha256"].write_text('{"different": 1}')
    # Snapshot manifest is bound, not recomputed from changed sources.
    previous_provider = Provider.__init__

    def original_manifest(self, *args, **kwargs):
        previous_provider(self, *args, **kwargs)
        self.manifest["node_vocab_sha256"] = "stale"

    monkeypatch.setattr(Provider, "__init__", original_manifest)
    with pytest.raises(ValueError, match="candidate source binding changed"):
        runner.main([*argv, "--resume"])
