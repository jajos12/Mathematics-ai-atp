"""Tests for lemma index to encoder binding.

An index is a pile of vectors in the space of the encoder that produced
them.  Nothing about a mismatched index raises at retrieval time -- the
dimensions still match, the scores still rank -- so the binding between an
index and its encoder must be recorded at build time and enforced at load
time.  These tests cover the hash machinery and the refusal paths.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def _stable_vocab_sha256(vocab: dict[str, int]) -> str:
    from maths_ai.gnn_inference.atp_lean_gnn.training import (
        _stable_vocab_sha256 as stable,
    )

    return stable(vocab)


def _encoder_state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "conv.weight": torch.randn(4, 8),
        "conv.bias": torch.randn(4),
        "readout.weight": torch.randn(4, 4),
    }


class StateDictHashTests(unittest.TestCase):
    def test_hash_is_deterministic_and_key_order_independent(self) -> None:
        from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import (
            state_dict_sha256,
        )

        sd = _encoder_state_dict()
        reordered = dict(reversed(list(sd.items())))
        self.assertEqual(state_dict_sha256(sd), state_dict_sha256(reordered))

    def test_same_encoder_in_baseline_and_pointer_files_hashes_equal(self) -> None:
        """The binding must be over the encoder, not the checkpoint file.

        An index is built from a baseline checkpoint but served through a
        pointer whose backbone holds those same weights under ``backbone.``.
        A file hash can never match that pair; the state hash must.
        """
        from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import (
            encoder_state_from_checkpoint,
            state_dict_sha256,
        )

        encoder = _encoder_state_dict()
        baseline_checkpoint = dict(encoder)
        pointer_checkpoint = {f"backbone.{k}": v for k, v in encoder.items()}
        pointer_checkpoint["argument_selector.gru.weight_ih"] = torch.randn(6, 6)
        pointer_checkpoint["stop_head.weight"] = torch.randn(1, 4)
        dataparallel_checkpoint = {
            f"module.backbone.{k}": v for k, v in encoder.items()
        }

        self.assertEqual(
            state_dict_sha256(encoder_state_from_checkpoint(baseline_checkpoint)),
            state_dict_sha256(encoder_state_from_checkpoint(pointer_checkpoint)),
        )
        self.assertEqual(
            state_dict_sha256(encoder_state_from_checkpoint(baseline_checkpoint)),
            state_dict_sha256(encoder_state_from_checkpoint(dataparallel_checkpoint)),
        )

    def test_retrained_encoder_hashes_differently(self) -> None:
        from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import (
            state_dict_sha256,
        )

        torch.manual_seed(1)
        retrained = {
            "conv.weight": torch.randn(4, 8),
            "conv.bias": torch.randn(4),
            "readout.weight": torch.randn(4, 4),
        }
        self.assertNotEqual(
            state_dict_sha256(_encoder_state_dict()),
            state_dict_sha256(retrained),
        )


def _write_index_dir(
    root: Path,
    *,
    encoder_state_dict: dict[str, torch.Tensor],
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    normalize: bool = False,
    dim: int = 4,
) -> Path:
    import faiss

    from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import state_dict_sha256

    index_dir = root / "lemma_index_v1"
    index_dir.mkdir(parents=True)
    vectors = np.random.RandomState(0).randn(2, dim).astype(np.float32)
    if normalize:
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(index_dir / "lemma_vectors.npy", vectors)
    (index_dir / "lemma_ids.json").write_text(json.dumps([7, 8]))
    (index_dir / "lemma_names.json").write_text(json.dumps(["Lemma.one", "Lemma.two"]))
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    manifest = {
        "encoder_state_sha256": state_dict_sha256(encoder_state_dict),
        "node_vocab_sha256": _stable_vocab_sha256(node_vocab),
        "tactic_vocab_sha256": _stable_vocab_sha256(tactic_vocab),
        "normalize": normalize,
    }
    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return index_dir


class LoadIndexForEncoderTests(unittest.TestCase):
    NODE_VOCAB = {"State": 1, "∀": 2, "P": 3}
    TACTIC_VOCAB = {"<UNK>": 0, "exact": 1, "apply": 2}

    def setUp(self) -> None:
        import faiss

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.encoder = _encoder_state_dict()
        self.index_dir = _write_index_dir(
            self.root,
            encoder_state_dict=self.encoder,
            node_vocab=self.NODE_VOCAB,
            tactic_vocab=self.TACTIC_VOCAB,
        )

    def _load(self, encoder_state_dict=None, node_vocab=None, tactic_vocab=None):
        from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import (
            load_index_for_encoder,
        )

        return load_index_for_encoder(
            self.index_dir,
            encoder_state_dict=encoder_state_dict or self.encoder,
            node_vocab=node_vocab or self.NODE_VOCAB,
            tactic_vocab=tactic_vocab or self.TACTIC_VOCAB,
        )

    def test_matching_encoder_loads(self) -> None:
        index = self._load()
        self.assertEqual(index.lemma_ids, [7, 8])

    def test_matching_encoder_loads_through_a_pointer_state_dict(self) -> None:
        pointer_state = {f"backbone.{k}": v for k, v in self.encoder.items()}
        pointer_state["stop_head.weight"] = torch.randn(1, 4)
        index = self._load(encoder_state_dict=pointer_state)
        self.assertEqual(index.lemma_ids, [7, 8])

    def test_retrained_encoder_is_refused(self) -> None:
        torch.manual_seed(1)
        retrained = {k: torch.randn_like(v) for k, v in self.encoder.items()}
        with self.assertRaisesRegex(ValueError, "different encoder"):
            self._load(encoder_state_dict=retrained)

    def test_vocabulary_mismatch_is_refused(self) -> None:
        other_vocab = {"State": 1, "∀": 2}
        with self.assertRaisesRegex(ValueError, "node"):
            self._load(node_vocab=other_vocab)

    def test_index_without_manifest_is_refused(self) -> None:
        (self.index_dir / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "no manifest"):
            self._load()

    def test_manifest_without_state_hash_is_refused(self) -> None:
        manifest_path = self.index_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("encoder_state_sha256")
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "predates index-to-encoder binding"):
            self._load()


class NormalizeRoundTripTests(unittest.TestCase):
    def test_build_time_normalize_reaches_query_time(self) -> None:
        """A normalized build must score normalized queries.

        The 8/31 finding: ``--normalize`` normalized the stored vectors but
        ``LemmaIndex.load`` defaulted ``normalize_queries=False`` and never
        read the manifest, so every caller scored ``q · (l/‖l‖)`` -- neither
        inner product nor cosine.
        """
        import faiss

        from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex

        with tempfile.TemporaryDirectory() as tmp:
            index_dir = _write_index_dir(
                Path(tmp),
                encoder_state_dict=_encoder_state_dict(),
                node_vocab={"State": 1},
                tactic_vocab={"<UNK>": 0},
                normalize=True,
            )
            # Load with the legacy default (normalize_queries=False); the
            # manifest's build-time decision must win.
            index = LemmaIndex.load(index_dir)
            self.assertTrue(index.normalize_queries)

    def test_unnormalized_build_stays_unnormalized(self) -> None:
        from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex

        with tempfile.TemporaryDirectory() as tmp:
            index_dir = _write_index_dir(
                Path(tmp),
                encoder_state_dict=_encoder_state_dict(),
                node_vocab={"State": 1},
                tactic_vocab={"<UNK>": 0},
                normalize=False,
            )
            index = LemmaIndex.load(index_dir)
            self.assertFalse(index.normalize_queries)


if __name__ == "__main__":
    unittest.main()
