"""End-to-end tests for the inference decode loop.

The decode loop in ``InferencePipeline._predict_from_dag`` replaced one-shot
pool ranking with sequential GRU decoding.  Nothing else drove that path, and a
shape mismatch between ``score_candidates`` (which keeps the batch dimension)
and the loop (which indexes flat positions) shipped twice.  These tests drive
the whole path with a fake candidate pool so every decode step runs.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from maths_ai.gnn_inference.atp_lean_gnn import build_vocab, proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn import inference as inference_module
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import (
    TacticWithArgsClassifier,
)
from maths_ai.gnn_inference.atp_lean_gnn.inference import InferencePipeline


STATE = "h1 : P\nh2 : Q\n⊢ P ∧ Q"
HIDDEN_DIM = 16


class _FakePool:
    """Stands in for the unified local+library pool build_unified_pools returns."""

    def __init__(self, dag) -> None:
        # Two local candidates (real DAG node ids) and two library candidates
        # (lemma ids resolved through the fake corpus).
        n_nodes = len(dag.nodes)
        self.candidate_ids = [min(1, n_nodes - 1), min(2, n_nodes - 1), 101, 102]
        self.candidate_sources = ["local", "local", "library", "library"]
        self.candidate_vectors = torch.randn(4, HIDDEN_DIM)


class _LemmaRecord:
    def __init__(self, name: str) -> None:
        self.name = name


class InferenceDecodeTests(unittest.TestCase):
    def _build_pipeline(self, *, stop_bias: float) -> tuple[InferencePipeline, _FakePool]:
        torch.manual_seed(0)
        dag = proof_state_to_dag(STATE)
        node_vocab = build_vocab([dag])
        tactic_vocab = {
            "<UNK>": 0, "exact": 1, "apply": 2, "rw": 3, "cases": 4, "intro": 5,
        }
        model = TacticWithArgsClassifier(
            num_node_labels=len(node_vocab),
            num_tactics=len(tactic_vocab),
            hidden_dim=HIDDEN_DIM,
            num_layers=2,
            max_args=3,
        )
        # A large negative bias makes the stop head never fire, so the decode
        # loop runs to max_args; a large positive one stops at step 0.
        with torch.no_grad():
            model.stop_head.bias.fill_(stop_bias)
        model.eval()

        pool = _FakePool(dag)
        lemma_corpus = {
            101: _LemmaRecord("Nat.add_comm"),
            102: _LemmaRecord("Nat.mul_comm"),
        }

        def fake_build_unified_pools(state_emb, node_embeddings, premise_mask, batch, *, lemma_index, k):
            return [pool]

        patcher = mock.patch.object(
            inference_module, "build_unified_pools", side_effect=fake_build_unified_pools
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        pipeline = InferencePipeline(
                model=model,
                scorer=SimpleNamespace(eval=lambda: None),
                lemma_index=object(),
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
                device=torch.device("cpu"),
                k=500,
                lemma_corpus=lemma_corpus,
            )
        return pipeline, pool

    def test_decode_selects_distinct_candidates_across_steps(self) -> None:
        """A never-firing stop head decodes max_args distinct pool positions.

        The second selection cannot repeat the first, so at least one selected
        position is >= 1: exactly the indexing that crashed when the loop read
        ``score_candidates``' [1, P] result as flat [P].
        """
        pipeline, pool = self._build_pipeline(stop_bias=-10.0)
        result = pipeline.predict_tactic_result(STATE, top_k=3)
        self.assertEqual(len(result.top_tactic_predictions), 3)

        for candidate in result.top_tactic_predictions:
            details = candidate["selected_argument_details"]
            self.assertEqual(len(details), pipeline.model.max_args)
            positions = [d.candidate_id for d in details]
            # The same pool candidate must never be selected twice in one action.
            self.assertEqual(len(positions), len(set(positions)))
            # Every decode step selected from the fake pool.
            for detail in details:
                self.assertIn(detail.candidate_id, pool.candidate_ids)

        # Library selections resolve to the corpus name, not a placeholder.
        library_labels = {
            d.label
            for c in result.top_tactic_predictions
            for d in c["selected_argument_details"]
            if d.source == "library"
        }
        self.assertTrue(library_labels.issubset({"Nat.add_comm", "Nat.mul_comm"}))
        self.assertNotIn("<lemma_101>", library_labels)

    def test_stop_head_halts_the_sequence_at_step_zero(self) -> None:
        """A firing stop head produces zero arguments, not a default arity."""
        pipeline, _ = self._build_pipeline(stop_bias=10.0)
        result = pipeline.predict_tactic_result(STATE, top_k=2)
        for candidate in result.top_tactic_predictions:
            self.assertEqual(candidate["selected_arguments"], [])
            self.assertEqual(candidate["selected_argument_details"], [])
        self.assertEqual(result.selected_arguments, [])


if __name__ == "__main__":
    unittest.main()
