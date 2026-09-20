"""Acceptance tests for typed mixed-pool complete-action reranking."""

from __future__ import annotations

import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import (
    TacticWithArgsClassifier,
)
from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import (
    CandidatePool,
    CandidateRef,
    CandidateSource,
)
from maths_ai.gnn_inference.atp_lean_gnn.unified_reranking import (
    ActionTarget,
    RankedAction,
    complete_action_metrics,
    compute_unified_reranking_loss,
    load_unified_reranker_weights,
    rank_complete_actions,
    resolve_ordered_pool_targets,
    verify_unified_reranker_dependencies,
)


HIDDEN_DIM = 8


def _model() -> TacticWithArgsClassifier:
    torch.manual_seed(0)
    model = TacticWithArgsClassifier(
        num_node_labels=10,
        num_tactics=4,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        max_args=3,
    )
    model.eval()
    return model


def _pool(sources: list[CandidateSource], *, graph_id: int = 0) -> CandidatePool:
    candidates = []
    local_ids = []
    lemma_ids = []
    for position, source in enumerate(sources):
        candidate_id = position if source is CandidateSource.LOCAL else 100 + position
        if source is CandidateSource.LOCAL:
            local_ids.append(candidate_id)
            candidates.append(
                CandidateRef(
                    source=source,
                    stable_id=candidate_id,
                    graph_id=graph_id,
                    local_node_index=candidate_id,
                )
            )
        else:
            lemma_ids.append(candidate_id)
            candidates.append(CandidateRef(source=source, stable_id=candidate_id))
    return CandidatePool(
        candidate_vectors=torch.randn(len(candidates), HIDDEN_DIM),
        candidate_sources=[candidate.source.value for candidate in candidates],
        candidate_ids=[candidate.stable_id for candidate in candidates],
        local_node_ids=local_ids,
        lemma_ids=lemma_ids,
        candidates=candidates,
        graph_id=graph_id,
    )


@pytest.mark.parametrize(
    "sources",
    [
        [CandidateSource.LOCAL, CandidateSource.LOCAL],
        [CandidateSource.LIBRARY, CandidateSource.LIBRARY],
        [CandidateSource.LOCAL, CandidateSource.LIBRARY],
    ],
)
def test_complete_action_decoder_supports_local_external_and_mixed(sources) -> None:
    actions = rank_complete_actions(
        _model(),
        torch.randn(1, HIDDEN_DIM),
        torch.tensor([0.0, 4.0, 1.0, -1.0]),
        _pool(sources),
        {0: "<UNK>", 1: "rw", 2: "exact", 3: "apply"},
        top_k=5,
        tactic_k=2,
        beam_size=4,
    )
    assert actions
    assert all(isinstance(argument, CandidateRef) for action in actions for argument in action.arguments)
    assert actions == sorted(actions, key=lambda action: action.log_probability, reverse=True)


def test_empty_pool_returns_stop_only_actions() -> None:
    actions = rank_complete_actions(
        _model(),
        torch.randn(1, HIDDEN_DIM),
        torch.tensor([0.0, 3.0, 2.0, 1.0]),
        _pool([]),
        {0: "<UNK>", 1: "simp", 2: "exact", 3: "apply"},
        top_k=3,
        tactic_k=3,
    )
    assert len(actions) == 3
    assert all(action.arguments == () for action in actions)


def test_multi_argument_actions_never_repeat_stable_candidates() -> None:
    model = _model()
    with torch.no_grad():
        model.stop_head.weight.zero_()
        model.stop_head.bias.fill_(-3.0)
    actions = rank_complete_actions(
        model,
        torch.randn(1, HIDDEN_DIM),
        torch.tensor([0.0, 4.0, 1.0, -1.0]),
        _pool(
            [
                CandidateSource.LOCAL,
                CandidateSource.LIBRARY,
                CandidateSource.LIBRARY,
            ]
        ),
        {1: "rw"},
        top_k=10,
        tactic_k=1,
        beam_size=8,
    )
    assert any(len(action.arguments) > 1 for action in actions)
    for action in actions:
        keys = [argument.key for argument in action.arguments]
        assert len(keys) == len(set(keys))


def test_ordered_target_resolution_preserves_source_and_position() -> None:
    pool = _pool([CandidateSource.LOCAL, CandidateSource.LIBRARY])
    positions, metrics = resolve_ordered_pool_targets(
        [pool],
        torch.tensor([[0, -1]]),
        torch.tensor([[-1, 101]]),
        [2],
        max_args=3,
    )
    assert positions == [[0, 1]]
    assert metrics == {
        "local_target_count": 1,
        "library_target_count": 1,
        "unresolved_target_count": 0,
    }


def test_missing_retrieved_target_is_reported_not_silently_remapped() -> None:
    pool = _pool([CandidateSource.LOCAL])
    positions, metrics = resolve_ordered_pool_targets(
        [pool],
        torch.tensor([[-1]]),
        torch.tensor([[999]]),
        [1],
        max_args=3,
    )
    assert positions == [[-1]]
    assert metrics["unresolved_target_count"] == 1


def test_mixed_sequence_loss_trains_existing_pointer_and_stop_head() -> None:
    model = _model()
    model.train()
    pool = _pool([CandidateSource.LOCAL, CandidateSource.LIBRARY])
    loss, metrics = compute_unified_reranking_loss(
        model,
        torch.randn(1, HIDDEN_DIM),
        torch.tensor([1]),
        [pool],
        [[0, 1]],
    )
    loss.backward()
    assert metrics["target_coverage"] == 1.0
    assert metrics["scored_target_count"] == 2
    assert model.argument_selector.gru.weight_ih.grad is not None
    assert model.stop_head.weight.grad is not None


def test_unresolved_suffix_remains_in_coverage_denominator() -> None:
    model = _model()
    pool = _pool([CandidateSource.LOCAL, CandidateSource.LIBRARY])
    _, metrics = compute_unified_reranking_loss(
        model,
        torch.randn(1, HIDDEN_DIM),
        torch.tensor([1]),
        [pool],
        [[-1, 1]],
    )
    assert metrics["target_count"] == 2
    assert metrics["scored_target_count"] == 0
    assert metrics["unresolved_target_count"] == 2
    assert metrics["target_coverage"] == 0.0


def test_complete_action_metrics_include_source_recall_and_per_tactic() -> None:
    local = CandidateRef(
        source=CandidateSource.LOCAL,
        stable_id=0,
        graph_id=0,
        local_node_index=0,
    )
    library = CandidateRef(source=CandidateSource.LIBRARY, stable_id=100)
    target = ActionTarget(tactic_id=2, arguments=(library, local))
    wrong = RankedAction(1, "apply", (local,), -0.1)
    right = RankedAction(2, "rw", (library, local), -0.2)
    metrics = complete_action_metrics([[wrong, right]], [target], ks=(1, 2))
    assert metrics["complete_action_exact_match"] == 0.0
    assert metrics["action_recall_at_1"] == 0.0
    assert metrics["action_recall_at_2"] == 1.0
    assert metrics["candidate_source_accuracy"] == 0.0
    assert metrics["candidate_source_count"] == 2
    assert metrics["per_tactic"][2]["count"] == 1


def test_unresolved_action_is_counted_as_miss() -> None:
    target = ActionTarget(tactic_id=1, arguments=(), resolved=False, argument_count=1)
    action = RankedAction(1, "exact", (), -0.1)
    metrics = complete_action_metrics([[action]], [target], ks=(1,))
    assert metrics["complete_action_exact_match"] == 0.0
    assert metrics["action_recall_at_1"] == 0.0
    assert metrics["candidate_source_accuracy"] == 0.0


def test_unified_checkpoint_loads_trained_pointer_weights(tmp_path) -> None:
    trained = _model()
    with torch.no_grad():
        trained.stop_head.bias.fill_(7.0)
    node_vocab = {str(index): index for index in range(10)}
    tactic_vocab = {str(index): index for index in range(4)}
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_type": "unified_action_reranker",
            "config": {"edge_mode": "bidirectional"},
            "model_state_dict": trained.state_dict(),
            "node_vocab": node_vocab,
            "tactic_vocab": tactic_vocab,
        },
        checkpoint_path,
    )
    served = _model()
    checkpoint = load_unified_reranker_weights(
        checkpoint_path,
        model=served,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        expected_edge_mode="bidirectional",
        device=torch.device("cpu"),
    )
    assert checkpoint is not None
    assert torch.equal(served.stop_head.bias, trained.stop_head.bias)


def test_unified_checkpoint_rejects_different_index_binding() -> None:
    checkpoint = {
        "lemma_index_manifest": {
            "encoder_state_sha256": "trained",
            "node_vocab_sha256": "nodes",
            "tactic_vocab_sha256": "tactics",
            "corpus_sha256": "corpus",
            "edge_mode": "bidirectional",
            "normalize": True,
        },
        "retriever_state_sha256": None,
    }
    serving_manifest = dict(checkpoint["lemma_index_manifest"])
    serving_manifest["corpus_sha256"] = "different"
    with pytest.raises(ValueError, match="different lemma index"):
        verify_unified_reranker_dependencies(
            checkpoint,
            index_manifest=serving_manifest,
            retriever=None,
        )


def test_premise_evaluation_reports_predicted_and_oracle_complete_actions() -> None:
    import faiss

    from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
    from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
    from maths_ai.gnn_inference.atp_lean_gnn.premise_training import (
        evaluate_model_with_premises,
    )

    data = Data(
        x=torch.tensor([1, 2, 3]),
        node_type=torch.tensor([0, 0, 0]),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        y=torch.tensor([1]),
        premise_mask=torch.tensor([False, True, True]),
        state_node_index=torch.tensor([0]),
        arg_node_indices=torch.tensor([1, -1]),
        arg_lemma_ids=torch.tensor([-1, 100]),
    )
    data.arg_count = 2
    data.tactic_name = "rw"
    data.tactic_raw = "rw [h, Demo.lemma]"
    loader = DataLoader([data], batch_size=1)
    vectors = np.ones((1, HIDDEN_DIM), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatIP(HIDDEN_DIM)
    faiss_index.add(vectors)
    lemma_index = LemmaIndex(
        faiss_index,
        [100],
        vectors,
        lemma_names=["Demo.lemma"],
        normalize_queries=True,
    )
    model = _model()
    metrics = evaluate_model_with_premises(
        model,
        PremiseScorer(HIDDEN_DIM),
        loader,
        lemma_index,
        device=torch.device("cpu"),
        unknown_tactic_id=0,
        arg_loss_weight=0.5,
        premise_loss_weight=0.3,
        tactic_vocab={"<UNK>": 0, "rw": 1, "exact": 2, "apply": 3},
        k=1,
    )
    assert metrics["complete_action_labeled_count"] == 1
    assert "complete_action_exact_match_predicted_tactic" in metrics
    assert "complete_action_exact_match_oracle_tactic" in metrics
    assert "action_recall_at_5_predicted_tactic" in metrics
    assert "candidate_source_accuracy_predicted_tactic" in metrics
