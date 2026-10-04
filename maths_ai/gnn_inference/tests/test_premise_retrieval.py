"""Tests for first-stage dual-encoder premise retrieval."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import (
    DualEncoderRetriever,
    build_positive_mask,
    mine_hard_negative_ids,
    multi_positive_info_nce,
    retrieval_metrics,
    load_retriever_checkpoint,
)
from maths_ai.gnn_inference.atp_lean_gnn.premise_retriever_training import (
    LemmaGraphStore,
    combine_retrieval_candidates,
    evaluate_retriever,
    extract_external_positive_ids,
    load_or_build_hnsw_index,
    sample_accessible_unused_ids,
    train_retriever_epoch,
)
from maths_ai.gnn_inference.atp_lean_gnn.retriever_experiments import (
    encode_corpus_index, evaluate_index,
)
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex


class _ToyEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim)

    def encode_nodes(self, graph_batch) -> torch.Tensor:
        return self.projection(graph_batch.features)

    def readout(self, node_embeddings, graph_batch) -> torch.Tensor:
        return node_embeddings


class _PriorIndex:
    lemma_ids = [10, 11, 12, 13, 14, 15]

    def search(self, query, *, k):
        rows = [[10, 11, 12, 13, 14, 15][:k] for _ in range(len(query))]
        vectors = np.zeros((len(rows), k, 3), dtype=np.float32)
        scores = np.zeros((len(rows), k), dtype=np.float32)
        return rows, vectors, scores


class _NormalizedPriorIndex(_PriorIndex):
    normalize_queries = True
    id_to_position = {
        lemma_id: position
        for position, lemma_id in enumerate(_PriorIndex.lemma_ids)
    }
    lemma_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5, 0.5],
        ],
        dtype=np.float32,
    )


class _TrainingBatch(SimpleNamespace):
    def to(self, device):
        for key, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self


class _FailingLemmaStore:
    def batch(self, lemma_ids, *, device):
        raise AssertionError("frozen lemma graphs must not be re-encoded")


class _ToyLemmaStore:
    def batch(self, lemma_ids, *, device):
        return SimpleNamespace(
            features=torch.eye(4, device=device)[
                torch.tensor([int(lemma_id) % 4 for lemma_id in lemma_ids], device=device)
            ]
        )


def test_trainable_lemma_tower_receives_gradients() -> None:
    model = DualEncoderRetriever(_ToyEncoder(4), _ToyEncoder(4), hidden_dim=4)
    batch = _TrainingBatch(
        features=torch.eye(4)[:2], y=torch.tensor([1, 1]),
        arg_count=torch.tensor([1, 1]),
        arg_lemma_ids=torch.tensor([10, 11]),
        arg_node_indices=torch.tensor([-1, -1]),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_retriever_epoch(
        model, [batch], _ToyLemmaStore(), _NormalizedPriorIndex(),
        optimizer=optimizer, device=torch.device("cpu"),
        accessible_negative_count=1, hard_negative_count=0,
        grad_clip=1.0, rng=random.Random(42),
        trainable_lemma_tower=True, max_batches=1,
    )
    assert model.lemma_encoder.projection.weight.grad is not None
    assert model.lemma_projection.weight.grad is not None


def test_interrupted_graph_epoch_resumes_without_replaying_batches() -> None:
    torch.manual_seed(3)
    initial = DualEncoderRetriever(_ToyEncoder(4), _ToyEncoder(4), hidden_dim=4)

    def batches():
        return [
            _TrainingBatch(
                features=torch.eye(4)[i:i + 1], y=torch.tensor([1]),
                arg_count=torch.tensor([1]),
                arg_lemma_ids=torch.tensor([10 + i]),
                arg_node_indices=torch.tensor([-1]),
            ) for i in range(2)
        ]

    def train(model, optimizer, rng, *, max_batches, start_batch=0, progress=None):
        return train_retriever_epoch(
            model, batches(), _ToyLemmaStore(), _NormalizedPriorIndex(),
            optimizer=optimizer, device=torch.device("cpu"),
            accessible_negative_count=1, hard_negative_count=0,
            grad_clip=1.0, rng=rng, trainable_lemma_tower=True,
            max_batches=max_batches, start_batch=start_batch,
            progress=progress, progress_every=1,
        )

    full = copy.deepcopy(initial)
    full_optimizer = torch.optim.AdamW(full.parameters(), lr=1e-3)
    train(full, full_optimizer, random.Random(42), max_batches=2)

    paused = copy.deepcopy(initial)
    paused_optimizer = torch.optim.AdamW(paused.parameters(), lr=1e-3)
    progress = []
    train(
        paused, paused_optimizer, random.Random(42), max_batches=1,
        progress=lambda next_batch, rng: progress.append((next_batch, rng.getstate())),
    )
    resumed = copy.deepcopy(initial)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed.load_state_dict(paused.state_dict())
    resumed_optimizer.load_state_dict(paused_optimizer.state_dict())
    rng = random.Random(42)
    rng.setstate(progress[0][1])
    assert progress[0][0] == 1
    train(resumed, resumed_optimizer, rng, max_batches=2, start_batch=1)
    for key, tensor in full.state_dict().items():
        assert torch.allclose(tensor, resumed.state_dict()[key], atol=1e-6)


def test_rebuilt_corpus_index_reflects_updated_lemma_vectors(tmp_path) -> None:
    records = [
        LemmaRecord(10, "A", "A : Prop", "", ""),
        LemmaRecord(11, "B", "B : Prop", "", ""),
    ]
    columns = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    def encode(ids):
        return torch.stack([columns[0 if lemma_id == 10 else 1] for lemma_id in ids])
    first = encode_corpus_index(
        records, encode, batch_size=1,
        manifest={"model_type": "joint_graph_retriever_index_v1", "normalize": True},
        output_dir=tmp_path / "epoch_1",
    )
    assert first.search(torch.tensor([[1.0, 0.0]]), k=1)[0] == [[10]]
    columns.reverse()
    second = encode_corpus_index(
        records, encode, batch_size=1,
        manifest={"model_type": "joint_graph_retriever_index_v1", "normalize": True},
        output_dir=tmp_path / "epoch_2",
    )
    assert second.search(torch.tensor([[1.0, 0.0]]), k=1)[0] == [[11]]
    assert LemmaIndex.load(tmp_path / "epoch_2").lemma_ids == [10, 11]


def test_full_corpus_encoding_resumes_without_reencoding_finished_batches(tmp_path) -> None:
    records = [LemmaRecord(i, str(i), "x : Nat", "", "") for i in (10, 11, 12)]
    calls = []

    def interrupted(ids):
        calls.extend(ids)
        if 12 in ids:
            raise RuntimeError("session restarted")
        return torch.eye(3)[torch.tensor([i - 10 for i in ids])]

    path = tmp_path / "unfinished"
    with pytest.raises(RuntimeError, match="session restarted"):
        encode_corpus_index(
            records, interrupted, batch_size=1,
            manifest={"encoder_state_sha256": "original"}, output_dir=path,
        )
    assert calls == [10, 11, 12]
    assert not (path / "manifest.json").exists()
    with pytest.raises(ValueError, match="different lemma tower"):
        encode_corpus_index(
            records, interrupted, batch_size=1,
            manifest={"encoder_state_sha256": "wrong"}, output_dir=path,
        )
    def resumed(ids):
        calls.extend(ids)
        return torch.eye(3)[torch.tensor([i - 10 for i in ids])]

    result = encode_corpus_index(
        records, resumed, batch_size=1,
        manifest={"encoder_state_sha256": "original"}, output_dir=path,
    )
    assert calls == [10, 11, 12, 12]
    assert result.search(torch.tensor([[0.0, 0.0, 1.0]]), k=1)[0] == [[12]]
    assert (path / "manifest.json").exists()
    assert not (path / "progress.json").exists()


def test_experiment_evaluation_uses_same_labeled_step_denominator() -> None:
    records = [LemmaRecord(i, str(i), "x : Nat", "", "") for i in (10, 11, 12)]
    index = encode_corpus_index(
        records,
        lambda ids: torch.eye(3)[torch.tensor([i - 10 for i in ids])],
        batch_size=2, manifest={"normalize": True},
    )
    batches = [
        (torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), [[10], [12]]),
        (torch.tensor([[0.0, 0.0, 1.0]]), [[]]),
    ]
    metrics = evaluate_index(
        index, batches, encode_queries=lambda batch: batch[0],
        extract_positives=lambda batch: batch[1], ks=(1, 2),
    )
    assert metrics["query_count"] == 3
    assert metrics["labeled_query_count"] == 2
    assert metrics["label_coverage"] == pytest.approx(2 / 3)
    assert metrics["recall_at_1"] == 0.5


def test_dual_encoder_uses_separate_towers_and_normalizes() -> None:
    state_encoder = _ToyEncoder(4)
    lemma_encoder = _ToyEncoder(4)
    model = DualEncoderRetriever(
        state_encoder,
        lemma_encoder,
        hidden_dim=4,
        projection_dim=3,
    )

    state_batch = SimpleNamespace(features=torch.randn(2, 4))
    lemma_batch = SimpleNamespace(features=torch.randn(3, 4))
    states = model.encode_states(state_batch)
    lemmas = model.encode_lemmas(lemma_batch)
    loss = model.similarity(states, lemmas).sum()
    loss.backward()

    assert states.shape == (2, 3)
    assert lemmas.shape == (3, 3)
    assert torch.allclose(states.norm(dim=1), torch.ones(2), atol=1e-6)
    assert torch.allclose(lemmas.norm(dim=1), torch.ones(3), atol=1e-6)
    assert state_encoder.projection.weight.grad is not None
    assert lemma_encoder.projection.weight.grad is not None
    assert state_encoder is not lemma_encoder


def test_frozen_lemma_tower_stays_in_eval_mode() -> None:
    model = DualEncoderRetriever(_ToyEncoder(4), _ToyEncoder(4), hidden_dim=4)
    model.freeze_lemma_tower()
    model.train()
    assert not model.lemma_encoder.training
    assert not model.lemma_projection.training
    assert not any(parameter.requires_grad for parameter in model.lemma_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.lemma_projection.parameters())
    assert torch.allclose(model.lemma_projection.weight, torch.eye(4))


def test_retriever_checkpoint_round_trip(tmp_path) -> None:
    pointer_backbone = _ToyEncoder(4)
    model = DualEncoderRetriever(
        copy.deepcopy(pointer_backbone),
        copy.deepcopy(pointer_backbone),
        hidden_dim=4,
        temperature=0.2,
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_type": "dual_encoder_retriever",
            "model_state_dict": model.state_dict(),
            "config": {"temperature": 0.2},
            "node_vocab": {"State": 1},
            "tactic_vocab": {"<UNK>": 0},
        },
        checkpoint_path,
    )
    loaded = load_retriever_checkpoint(
        checkpoint_path,
        pointer_backbone=pointer_backbone,
        hidden_dim=4,
        node_vocab={"State": 1},
        tactic_vocab={"<UNK>": 0},
        device=torch.device("cpu"),
    )
    assert loaded.temperature == 0.2
    assert not loaded.training
    assert not any(
        parameter.requires_grad for parameter in loaded.lemma_encoder.parameters()
    )


def test_retriever_checkpoint_rejects_different_vocab(tmp_path) -> None:
    model = DualEncoderRetriever(_ToyEncoder(4), _ToyEncoder(4), hidden_dim=4)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_type": "dual_encoder_retriever",
            "model_state_dict": model.state_dict(),
            "config": {"temperature": 0.07},
            "node_vocab": {"State": 1},
            "tactic_vocab": {"<UNK>": 0},
        },
        checkpoint_path,
    )
    with pytest.raises(ValueError, match="vocabularies"):
        load_retriever_checkpoint(
            checkpoint_path,
            pointer_backbone=_ToyEncoder(4),
            hidden_dim=4,
            node_vocab={"State": 1, "P": 2},
            tactic_vocab={"<UNK>": 0},
            device=torch.device("cpu"),
        )


def test_retriever_training_resume_restores_optimizer_and_overrides_lr(tmp_path) -> None:
    from maths_ai.gnn_inference.scripts.train_retriever import _restore_resume_state

    model = DualEncoderRetriever(_ToyEncoder(4), _ToyEncoder(4), hidden_dim=4)
    model.freeze_lemma_tower()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
        weight_decay=1e-4,
    )
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "epoch": 10,
            "model_type": "dual_encoder_retriever",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "node_vocab": {"State": 1},
            "tactic_vocab": {"<UNK>": 0},
        },
        checkpoint_path,
    )
    with torch.no_grad():
        model.state_projection.weight.zero_()
    restored = _restore_resume_state(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        node_vocab={"State": 1},
        tactic_vocab={"<UNK>": 0},
        learning_rate=5e-5,
        weight_decay=2e-4,
        device=torch.device("cpu"),
    )
    assert restored["epoch"] == 10
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[0]["weight_decay"] == 2e-4
    assert not torch.equal(
        model.state_projection.weight, torch.zeros_like(model.state_projection.weight)
    )


def test_multi_positive_info_nce_rewards_either_positive() -> None:
    positives = torch.tensor([[True, True, False], [False, False, False]])
    good_logits = torch.tensor([[4.0, 3.0, -2.0], [1.0, 2.0, 3.0]], requires_grad=True)
    bad_logits = torch.tensor([[-2.0, -1.0, 4.0], [1.0, 2.0, 3.0]])

    good_loss, metrics = multi_positive_info_nce(good_logits, positives)
    bad_loss, _ = multi_positive_info_nce(bad_logits, positives)
    good_loss.backward()

    assert good_loss < bad_loss
    assert good_logits.grad is not None
    assert metrics == {
        "query_count": 2,
        "labeled_query_count": 1,
        "positive_count": 2,
        "label_coverage": 0.5,
    }


def test_unlabeled_contrastive_batch_has_backward_safe_zero_loss() -> None:
    logits = torch.randn(2, 3, requires_grad=True)
    loss, metrics = multi_positive_info_nce(
        logits, torch.zeros_like(logits, dtype=torch.bool)
    )
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None
    assert metrics["label_coverage"] == 0.0


def test_positive_mask_supports_multiple_positives_and_duplicate_candidates() -> None:
    mask = build_positive_mask([5, 7, 5, 9], [{5, 9}, {7}, set()])
    assert mask.tolist() == [
        [True, False, True, True],
        [False, True, False, False],
        [False, False, False, False],
    ]


def test_hard_negative_mining_excludes_positives_and_inaccessible_ids() -> None:
    mined = mine_hard_negative_ids(
        _PriorIndex(),
        torch.randn(2, 3),
        [{10, 12}, {11}],
        k=2,
        accessible_ids_per_query=[{10, 11, 12, 14}, {11, 13, 15}],
    )
    assert mined == [[11, 14], [13, 15]]


def test_hard_negative_mining_validates_visibility_rows() -> None:
    with pytest.raises(ValueError, match="one row per query"):
        mine_hard_negative_ids(
            _PriorIndex(),
            torch.randn(2, 3),
            [{10}, {11}],
            k=2,
            accessible_ids_per_query=[{10}],
        )


def test_retrieval_metrics_report_coverage_recall_and_mrr_separately() -> None:
    metrics = retrieval_metrics(
        [
            [2, 8, 3, 4],  # first positive at rank 2
            [5, 6, 7, 9],  # positive absent
            [1, 2, 3, 4],  # unlabeled: excluded from retrieval denominator
        ],
        [{8, 9}, {10}, set()],
        ks=(1, 2, 4),
    ).as_dict()

    assert metrics["query_count"] == 3
    assert metrics["labeled_query_count"] == 2
    assert metrics["positive_count"] == 3
    assert metrics["label_coverage"] == pytest.approx(2 / 3)
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_2"] == 0.5
    assert metrics["recall_at_4"] == 0.5
    assert metrics["mrr"] == 0.25


def test_retrieval_metrics_default_acceptance_cutoffs() -> None:
    metrics = retrieval_metrics([[42]], [{42}]).as_dict()
    assert set(key for key in metrics if key.startswith("recall_at_")) == {
        "recall_at_1",
        "recall_at_10",
        "recall_at_50",
        "recall_at_200",
    }


def test_external_positive_extraction_preserves_multiple_citations() -> None:
    batch = SimpleNamespace(
        y=torch.tensor([0, 1, 2]),
        arg_count=torch.tensor([3, 1, 0]),
        arg_lemma_ids=torch.tensor([10, -1, 12, 11]),
    )
    assert extract_external_positive_ids(batch) == [[10, 12], [11], []]


def test_external_positive_extraction_recovers_legacy_missing_ids() -> None:
    batch = SimpleNamespace(
        y=torch.tensor([0]),
        arg_count=torch.tensor([2]),
        arg_lemma_ids=torch.tensor([-1, -1]),
        arg_node_indices=torch.tensor([3, -1]),
        tactic_raw=["rw [h, Nat.add_comm]"],
    )
    assert extract_external_positive_ids(
        batch, name_to_id={"Nat.add_comm": 42}
    ) == [[42]]


def test_lemma_graph_store_uses_structural_state_root() -> None:
    from maths_ai.gnn_inference.atp_lean_gnn.graph import (
        lemma_statement_to_dag,
        root_state_node_id,
    )
    from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord

    dag = lemma_statement_to_dag("State")
    assert sum(node.label == "State" for node in dag.nodes) == 2
    root_id = root_state_node_id(dag)
    store = LemmaGraphStore(
        [
            LemmaRecord(
                lemma_id=1,
                name="Demo.state",
                statement="State",
                namespace="Demo",
                module="Demo",
            )
        ],
        node_vocab={"<UNK>": 0, "State": 1, "Goal": 2},
    )
    graph = store.graph(1)
    assert graph.state_node_index.tolist() == [root_id]


def test_hard_negative_index_is_approximate_and_cached(tmp_path) -> None:
    source = _NormalizedPriorIndex()
    source.lemma_names = [str(value) for value in source.lemma_ids]
    source.manifest = {"normalize": True}
    cache_path = tmp_path / "hard.faiss"
    first = load_or_build_hnsw_index(source, cache_path, connections=4)
    second = load_or_build_hnsw_index(source, cache_path, connections=4)
    assert cache_path.exists()
    assert first.lemma_ids == source.lemma_ids
    assert second.index.ntotal == len(source.lemma_ids)
    assert second.normalize_queries


def test_accessible_sampling_excludes_citations_and_is_deterministic() -> None:
    import random

    first = sample_accessible_unused_ids(
        [[1, 2, 3, 4], [1, 2, 3]],
        [[2], [1, 3]],
        count=2,
        rng=random.Random(7),
    )
    second = sample_accessible_unused_ids(
        [[1, 2, 3, 4], [1, 2, 3]],
        [[2], [1, 3]],
        count=2,
        rng=random.Random(7),
    )
    assert first == second
    assert 2 not in first[0]
    assert first[1] == [2]


def test_candidate_counts_keep_negative_sources_separate() -> None:
    candidates = combine_retrieval_candidates(
        [[10, 11], [12]],
        [[20, 21], [22, 23]],
        [[30], [31]],
    )
    assert candidates.lemma_ids == [10, 11, 12, 20, 21, 22, 23, 30, 31]
    assert candidates.in_batch_negative_count == 3
    assert candidates.accessible_negative_count == 4
    assert candidates.hard_negative_count == 2


def test_training_epoch_encodes_external_candidates_and_reports_sources() -> None:
    import random

    torch.manual_seed(4)
    model = DualEncoderRetriever(_ToyEncoder(4), _ToyEncoder(4), hidden_dim=4)
    model.freeze_lemma_tower()
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    batch = _TrainingBatch(
        features=torch.eye(4)[:2],
        y=torch.tensor([0, 0]),
        arg_count=torch.tensor([1, 1]),
        arg_lemma_ids=torch.tensor([10, 11]),
    )
    metrics = train_retriever_epoch(
        model,
        [batch],
        _FailingLemmaStore(),
        _NormalizedPriorIndex(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        accessible_negative_count=1,
        hard_negative_count=1,
        grad_clip=1.0,
        rng=random.Random(0),
    )
    assert metrics["labeled_query_count"] == 2
    assert metrics["positive_count"] == 2
    assert metrics["label_coverage"] == 1.0
    assert metrics["in_batch_negative_count"] == 2
    assert metrics["accessible_negative_count"] == 2
    assert metrics["hard_negative_count"] == 2


def test_retriever_evaluation_reports_requested_corpus_metrics() -> None:
    model = DualEncoderRetriever(_ToyEncoder(4), _ToyEncoder(4), hidden_dim=4)
    batch = _TrainingBatch(
        features=torch.eye(4)[:2],
        y=torch.tensor([0, 0]),
        arg_count=torch.tensor([1, 1]),
        arg_lemma_ids=torch.tensor([10, 11]),
    )
    metrics = evaluate_retriever(
        model,
        [batch],
        _NormalizedPriorIndex(),
        device=torch.device("cpu"),
        ks=(1, 2),
    )
    assert metrics["label_coverage"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0
    assert metrics["mrr"] == 0.75
