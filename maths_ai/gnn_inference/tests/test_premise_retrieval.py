"""Tests for first-stage dual-encoder premise retrieval."""

from __future__ import annotations

import copy
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
    combine_retrieval_candidates,
    evaluate_retriever,
    extract_external_positive_ids,
    sample_accessible_unused_ids,
    train_retriever_epoch,
)


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


class _TrainingBatch(SimpleNamespace):
    def to(self, device):
        for key, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self


class _ToyLemmaStore:
    lemma_ids = (10, 11, 12, 13, 14, 15)

    def batch(self, lemma_ids, *, device):
        features = torch.tensor(
            [[float(value == basis) for basis in (10, 11, 12, 13)] for value in lemma_ids],
            dtype=torch.float32,
            device=device,
        )
        return SimpleNamespace(features=features)


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
        _ToyLemmaStore(),
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
