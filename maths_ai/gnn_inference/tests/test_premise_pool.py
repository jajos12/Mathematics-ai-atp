from __future__ import annotations

import unittest

import numpy as np
import torch

from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import (
    CandidatePool,
    CandidateRef,
    CandidateSource,
    build_unified_pools,
    ensure_library_targets,
)


class _FakeLemmaIndex:
    def search(self, goal_vecs, *, k):
        batch_size = int(goal_vecs.size(0))
        dim = int(goal_vecs.size(1))
        lemma_ids = []
        lemma_vecs = []
        for b in range(batch_size):
            lemma_ids.append([100 + 2 * b, 100 + 2 * b + 1])
            lemma_vecs.append(
                np.full((2, dim), fill_value=float(b + 1), dtype=np.float32)
            )
        lemma_vecs = np.stack(lemma_vecs, axis=0)
        scores = np.zeros((batch_size, 2), dtype=np.float32)
        return lemma_ids, lemma_vecs, scores


class _PaddedLemmaIndex:
    def search(self, goal_vecs, *, k):
        dim = int(goal_vecs.size(1))
        return (
            [[100, -1]],
            np.stack([np.ones((2, dim), dtype=np.float32)]),
            np.array([[1.0, float("-inf")]], dtype=np.float32),
        )


class PremisePoolTests(unittest.TestCase):
    def test_injects_external_positive_missed_by_top_k(self) -> None:
        pool = CandidatePool(
            candidate_vectors=torch.randn(1, 4),
            candidate_sources=["local"],
            candidate_ids=[0],
            local_node_ids=[0],
            lemma_ids=[],
            graph_id=0,
        )
        index = type(
            "Index",
            (),
            {
                "id_to_position": {999: 0},
                "lemma_vectors": np.ones((1, 4), dtype=np.float32),
            },
        )()
        augmented = ensure_library_targets(
            [pool], index, torch.tensor([[999, -1]])
        )[0]
        self.assertEqual(augmented.candidate_ids, [0, 999])
        self.assertEqual(augmented.candidate_sources, ["local", "library"])
        self.assertTrue(augmented.candidates[-1].metadata["injected_positive"])

    def test_missing_positive_in_bound_index_fails(self) -> None:
        pool = CandidatePool(
            candidate_vectors=torch.empty(0, 4),
            candidate_sources=[],
            candidate_ids=[],
            local_node_ids=[],
            lemma_ids=[],
            graph_id=0,
        )
        index = type(
            "Index",
            (),
            {"id_to_position": {}, "lemma_vectors": np.empty((0, 4), dtype=np.float32)},
        )()
        with self.assertRaisesRegex(ValueError, "absent from the bound lemma index"):
            ensure_library_targets([pool], index, torch.tensor([[999]]))

    def test_filters_faiss_padding_from_typed_pool(self) -> None:
        pools = build_unified_pools(
            torch.randn(1, 4),
            torch.randn(2, 4),
            torch.tensor([False, False]),
            torch.tensor([0, 0]),
            lemma_index=_PaddedLemmaIndex(),
            k=2,
        )
        self.assertEqual(pools[0].candidate_ids, [100])
        self.assertEqual(pools[0].candidate_sources, ["library"])

    def test_rejects_cross_graph_local_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "different graph"):
            CandidatePool(
                candidate_vectors=torch.randn(1, 4),
                candidate_sources=["local"],
                candidate_ids=[2],
                local_node_ids=[2],
                lemma_ids=[],
                candidates=[
                    CandidateRef(
                        source=CandidateSource.LOCAL,
                        stable_id=2,
                        graph_id=1,
                        local_node_index=2,
                    )
                ],
                graph_id=0,
            )

    def test_builds_unified_pool(self) -> None:
        goal_vecs = torch.randn(2, 4)
        node_embeddings = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=torch.float,
        )
        premise_mask = torch.tensor([True, False, True, False, True, True])
        batch_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

        pools = build_unified_pools(
            goal_vecs,
            node_embeddings,
            premise_mask,
            batch_index,
            lemma_index=_FakeLemmaIndex(),
            k=2,
        )

        self.assertEqual(len(pools), 2)

        pool0 = pools[0]
        self.assertEqual(pool0.local_node_ids, [0, 2])
        self.assertEqual(pool0.lemma_ids, [100, 101])
        self.assertEqual(pool0.candidate_sources.count("local"), 2)
        self.assertEqual(pool0.candidate_sources.count("library"), 2)
        self.assertEqual(pool0.candidates[0].source, CandidateSource.LOCAL)
        self.assertEqual(pool0.candidates[-1].source, CandidateSource.LIBRARY)
        self.assertEqual(pool0.candidates[-1].stable_id, 101)
        self.assertEqual(pool0.candidate_vectors.shape[0], 4)

        pool1 = pools[1]
        self.assertEqual(pool1.local_node_ids, [1, 2])
        self.assertEqual(pool1.lemma_ids, [102, 103])
        self.assertEqual(pool1.candidate_sources.count("local"), 2)
        self.assertEqual(pool1.candidate_sources.count("library"), 2)
        self.assertEqual(pool1.candidate_vectors.shape[0], 4)
