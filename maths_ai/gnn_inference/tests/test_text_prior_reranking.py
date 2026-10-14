"""Text prior augments library ranks without changing local candidate logits."""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import TacticWithArgsClassifier
from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidateRef, CandidateSource
from maths_ai.gnn_inference.atp_lean_gnn.text_prior_reranking import (
    blend_library_scores, rank_of,
)
from maths_ai.gnn_inference.scripts.evaluate_text_prior_reranking import _collect, _metrics


def test_text_prior_rescues_library_target_without_modifying_local_score():
    candidates = [
        CandidateRef(source=CandidateSource.LOCAL, stable_id=0),
        CandidateRef(source=CandidateSource.LIBRARY, stable_id=100, retrieval_score=0.25),
        CandidateRef(source=CandidateSource.LIBRARY, stable_id=101, retrieval_score=0.75),
    ]
    pointer = np.asarray([2.0, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(blend_library_scores(pointer, candidates, alpha=0), pointer)
    new = blend_library_scores(pointer, candidates, alpha=3)
    assert new[0] == pointer[0]
    assert rank_of(pointer, 2, eligible=[1, 2]) == 2
    assert rank_of(new, 2, eligible=[1, 2]) == 1
    assert rank_of(new, 2) == 1


def test_text_prior_requires_finite_text_scores_and_valid_strength():
    candidate = CandidateRef(source=CandidateSource.LIBRARY, stable_id=1)
    with pytest.raises(ValueError, match="finite text retrieval score"):
        blend_library_scores(np.ones(1), [candidate], alpha=1)
    with pytest.raises(ValueError, match="nonnegative"):
        blend_library_scores(np.ones(1), [candidate], alpha=-1)


def test_step_metrics_compare_pointer_text_and_combined_on_same_gold():
    samples = [(
        np.asarray([0.0, 2.0, 1.0], dtype=np.float32),
        [("local", None), ("library", 0.1), ("library", 0.9)],
        2,
    )]
    original = _metrics(samples, 0)
    combined = _metrics(samples, 3)
    assert original["library_positions"] == combined["library_positions"] == 1
    assert original["library_only_top1"] == 0
    assert combined["library_only_top1"] == 1
    assert combined["text_only_library_top1"] == 1


def test_collection_uses_true_tactic_and_prefix_on_cached_library_candidates():
    data = Data(
        x=torch.tensor([1, 2, 3]), node_type=torch.zeros(3, dtype=torch.long),
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        premise_mask=torch.tensor([False, True, True]),
        state_node_index=torch.tensor([0]), y=torch.tensor([1]),
        arg_node_indices=torch.tensor([-1]), arg_lemma_ids=torch.tensor([101]),
    )
    data.arg_count = 1
    data.action_target_supported = True
    data.tactic_raw = "exact Demo.b"

    class Cached:
        def search(self, batch, *, k):
            assert int(batch.y.item()) == 1
            assert k == 200
            return (
                [[100, 101]],
                np.eye(8, dtype=np.float32)[:2][None],
                np.asarray([[0.1, 0.9]], dtype=np.float32),
            )

    class Index:
        name_to_id = {"Demo.b": 101}

    model = TacticWithArgsClassifier(
        num_node_labels=4, num_tactics=4, hidden_dim=8, num_layers=1, max_args=2,
    ).eval()
    rows = _collect(
        model, [data], Cached(), Index(), torch.device("cpu"),
        k=200, max_library_positions=1,
    )
    assert len(rows) == 1
    assert rows[0][1][rows[0][2]] == ("library", pytest.approx(0.9))
    assert _metrics(rows, 0)["library_positions"] == 1
    assert _metrics(rows, 3)["text_only_library_top1"] == 1
