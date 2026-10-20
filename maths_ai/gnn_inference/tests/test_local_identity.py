"""Local identity learning retains mixed-prefix supervision and fixed cohort."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import JointSourceDecoder
from maths_ai.gnn_inference.atp_lean_gnn.training import GraphBudgetBatchSampler
from maths_ai.gnn_inference.scripts.train_local_identity import _local_rows, _run_epoch


def _graph(row, local_ids, library_ids):
    graph = Data(
        x=torch.tensor([1, 2, 3]), node_type=torch.tensor([3, 0, 0]),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        y=torch.tensor([1]), state_node_index=torch.tensor([0]),
        premise_mask=torch.tensor([False, True, True]),
        arg_node_indices=torch.tensor(local_ids),
        arg_lemma_ids=torch.tensor(library_ids),
    )
    graph.row_index = row
    graph.tactic_name = "rw"
    graph.arg_count = len(local_ids)
    graph.action_target_supported = True
    graph.split = "val"
    return graph


def _model():
    pointer = TacticWithArgsClassifier(
        num_node_labels=8, num_tactics=3,
        hidden_dim=8, num_layers=1, max_args=3,
    )
    model = JointSourceDecoder(pointer, text_dim=4)
    model.requires_grad_(False)
    model.pointer.argument_selector.out_proj.requires_grad_(True)
    return model


def test_full_local_supervision_with_missing_text_pool_lemma_and_matched_val():
    torch.manual_seed(17)
    graphs = [_graph(7, [-1, 1], [999, -1]), _graph(9, [2], [-1])]
    assert _local_rows(graphs) == [0, 1]
    # The earlier lemma is a valid text-index declaration but absent from the
    # frozen top-200 pool. Its gold vector still provides a teacher-forced GRU
    # prefix; it does not turn that row into a reachable serving action.
    index = SimpleNamespace(
        lemma_vectors=np.eye(4, dtype=np.float32)[:3],
        id_to_position={100: 0, 101: 1, 999: 2},
    )
    val_candidates = {
        "rows": np.asarray([7, 9], dtype=np.int64),
        "ids": np.asarray([[100, 101], [100, 101]], dtype=np.int32),
    }
    sampler = GraphBudgetBatchSampler(
        [(3, 2), (3, 2)], max_graphs=2, max_nodes=100, max_edges=100,
        oversize_policy="skip", seed=42,
    )
    loader = DataLoader(graphs, batch_sampler=sampler)
    model = _model()
    before = model.pointer.argument_selector.out_proj.weight.detach().clone()
    backbone_before = model.pointer.backbone.label_embedding.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.pointer.argument_selector.out_proj.parameters(), lr=0.01)
    train = _run_epoch(model, loader, sampler, index, device=torch.device("cpu"),
                       epoch=1, split="train", optimizer=optimizer)
    assert train["rows_with_local"] == 2
    assert train["all_local_positions"] == 2
    assert train["mean_local_loss"] > 0
    assert not torch.equal(before, model.pointer.argument_selector.out_proj.weight)
    torch.testing.assert_close(backbone_before, model.pointer.backbone.label_embedding.weight)

    val = _run_epoch(model, loader, sampler, index, device=torch.device("cpu"),
                     epoch=1, split="val", candidate_table=val_candidates)
    assert val["all_local_positions"] == 2
    assert val["text_pool_reachable_rows"] == 1
    assert val["text_pool_reachable_local_positions"] == 1
    assert 0 <= val["text_pool_reachable_top1"] <= 1


def test_unsupported_rows_do_not_enter_local_training_population():
    graphs = [_graph(7, [1], [-1]), _graph(9, [2], [-1])]
    graphs[1].action_target_supported = False
    assert _local_rows(graphs) == [0]


def test_local_runner_keeps_epoch_zero_baseline_and_resumes(tmp_path, monkeypatch):
    import faiss

    from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
    from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
    from maths_ai.gnn_inference.atp_lean_gnn.training import PointerConfig
    from maths_ai.gnn_inference.scripts import train_local_identity as runner

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    snapshot = tmp_path / "text.pt"
    snapshot.write_bytes(b"epoch 5")
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "manifest.json").write_text("{}")
    vectors = np.eye(4, dtype=np.float32)[:3]
    faiss_index = faiss.IndexFlatIP(4)
    faiss_index.add(vectors)
    index = LemmaIndex(faiss_index, [100, 101, 999], vectors, manifest={"text_epoch": 5})
    val = [_graph(7, [-1, 1], [999, -1]), _graph(9, [2], [-1])]
    train = [_graph(7, [-1, 1], [999, -1]), _graph(9, [2], [-1])]
    for graph in train:
        graph.split = "train"

    class Rows(list):
        def graph_sizes(self):
            return [(int(graph.num_nodes), int(graph.edge_index.size(1))) for graph in self]

    datasets = {"train": Rows(train), "val": Rows(val)}
    source = _model()
    config = PointerConfig.from_dict({
        "prepared_root": str(prepared), "device": "cpu", "max_args": 3,
        "model": {"hidden_dim": 8, "num_layers": 1},
    })
    joint = tmp_path / "joint.pt"
    torch.save({
        "model_type": "joint_source_flat_decoder_v2",
        "binding": {
            "prepared_root": str(prepared.resolve()),
            "text_index_manifest": index.manifest,
            "text_checkpoint_sha256": file_sha256(snapshot),
            "candidate_manifest_sha256": file_sha256(candidates / "manifest.json"),
        },
        "config": config.to_dict(), "model_state_dict": source.state_dict(),
    }, joint)

    class Provider:
        def __init__(self, *args, **kwargs):
            self.tables = {"val": {
                "rows": np.asarray([7, 9]),
                "ids": np.asarray([[100, 101], [100, 101]]),
            }}

        def validate_dataset(self, dataset, *, split):
            assert split == "val" and len(dataset) == 2

    monkeypatch.setattr(runner, "LemmaIndex", SimpleNamespace(load=lambda _: index))
    monkeypatch.setattr(runner, "CachedTextLibraryCandidates", Provider)
    monkeypatch.setattr(runner, "load_prepared_metadata", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(runner, "PreparedGraphDataset", lambda *a, split, **k: datasets[split])
    monkeypatch.setattr(runner, "build_pointer_model", lambda *a, **k: _model().pointer)
    run = tmp_path / "local_run"
    argv = [
        "--prepared-root", str(prepared), "--joint-checkpoint", str(joint),
        "--text-checkpoint", str(snapshot), "--text-index", str(tmp_path),
        "--candidate-dir", str(candidates), "--run-dir", str(run),
        "--device", "cpu", "--epochs", "1", "--batch-size", "2",
        "--expected-val-local", "2", "--expected-val-reachable", "1",
    ]
    assert runner.main(argv) == 0
    baseline = json.loads((run / "metrics_epoch_0.json").read_text())
    assert baseline["all_local_positions"] == 2
    assert baseline["text_pool_reachable_local_positions"] == 1
    assert runner.main([*argv[:argv.index("--epochs")], "--epochs", "2",
                        *argv[argv.index("--batch-size"):], "--resume"]) == 0
    saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2 and saved["phase"] == "complete"
    assert saved["binding"]["joint_checkpoint_sha256"] == file_sha256(joint)
    with pytest.raises(ValueError, match="changed"):
        runner.main([*argv[:argv.index("--expected-val-reachable")],
                     "--expected-val-reachable", "3", "--resume"])
