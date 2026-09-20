from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LemmaIndexConfig:
    index_dir: Path
    k: int = 500
    normalize_queries: bool = False


class LemmaIndex:
    """Load and query a FAISS index of precomputed lemma embeddings."""

    def __init__(
        self,
        index: Any,
        lemma_ids: list[int],
        lemma_vectors: np.ndarray,
        *,
        lemma_names: list[str] | None = None,
        normalize_queries: bool = False,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.lemma_ids = lemma_ids
        self.lemma_vectors = lemma_vectors
        self.normalize_queries = normalize_queries
        self.manifest = dict(manifest or {})
        self.lemma_names = lemma_names or []
        self.id_to_position = {
            lemma_id: position for position, lemma_id in enumerate(self.lemma_ids)
        }
        self.name_to_id = {
            name: lemma_id for name, lemma_id in zip(self.lemma_names, self.lemma_ids)
        }

        if self.lemma_vectors.ndim != 2:
            raise ValueError("lemma_vectors must be 2D (num_lemmas, dim).")
        if len(self.lemma_ids) != self.lemma_vectors.shape[0]:
            raise ValueError("lemma_ids length must match lemma_vectors rows.")
        if hasattr(self.index, "d") and int(self.index.d) != int(self.lemma_vectors.shape[1]):
            raise ValueError("FAISS index dimension does not match lemma_vectors.")

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        *,
        normalize_queries: bool = False,
        manifest: dict[str, Any] | None = None,
    ) -> "LemmaIndex":
        """Load an index, applying the build-time decisions its manifest records.

        ``manifest`` may be passed directly (a loader that already read it, such
        as a bundle loader) or omitted, in which case ``manifest.json`` beside
        the index is read when present.  An index built with ``--normalize``
        must score queries the same way: taking ``normalize_queries`` from the
        caller used to silently produce ``q · (l/‖l‖)`` — neither inner product
        nor cosine — whenever the two disagreed.  The build decision wins
        unless the caller explicitly forces a value, which only the legacy
        no-manifest path can reach.
        """
        import faiss

        input_path = Path(index_dir)

        if input_path.is_file():
            index_path = input_path
            vectors_path = input_path.with_name("lemma_vectors.npy")
            ids_path = input_path.with_name("lemma_ids.json")
            names_path = input_path.with_name("lemma_names.json")
        else:
            index_path = input_path / "faiss.index"
            vectors_path = input_path / "lemma_vectors.npy"
            ids_path = input_path / "lemma_ids.json"
            names_path = input_path / "lemma_names.json"

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at '{index_path}'.")
        if not vectors_path.exists():
            raise FileNotFoundError(f"Lemma vectors not found at '{vectors_path}'.")
        if not ids_path.exists():
            raise FileNotFoundError(f"Lemma id map not found at '{ids_path}'.")

        if manifest is None:
            manifest_path = (
                input_path.parent / "manifest.json"
                if input_path.is_file() else input_path / "manifest.json"
            )
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest is not None and bool(manifest.get("normalize")):
            # The build normalized the stored vectors; queries must be too.
            normalize_queries = True

        lemma_vectors = np.load(vectors_path)
        lemma_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        if not isinstance(lemma_ids, list):
            raise ValueError("lemma_ids.json must contain a JSON list of ids.")

        lemma_names = (
            json.loads(names_path.read_text(encoding="utf-8"))
            if names_path.exists() else None
        )
        index = faiss.read_index(str(index_path))
        return cls(
            index,
            [int(x) for x in lemma_ids],
            lemma_vectors,
            lemma_names=lemma_names,
            normalize_queries=normalize_queries,
            manifest=manifest,
        )

    def search(
        self,
        state_vecs: np.ndarray | "torch.Tensor",
        *,
        k: int = 500,
    ) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        """Return (lemma_ids, lemma_vecs, scores) for each query."""
        query = self._to_numpy(state_vecs)
        if self.normalize_queries:
            query = _normalize_rows(query)

        scores, indices = self.index.search(query, k)
        
        num_lemmas = len(self.lemma_ids)
        lemma_ids = [
            [self.lemma_ids[int(idx)] if 0 <= int(idx) < num_lemmas else -1 for idx in row]
            for row in indices
        ]
        
        valid_mask = (indices >= 0) & (indices < num_lemmas)
        safe_indices = np.where(valid_mask, indices, 0)
        
        if num_lemmas > 0:
            lemma_vecs = self.lemma_vectors[safe_indices]
            lemma_vecs[~valid_mask] = 0.0
        else:
            lemma_vecs = np.zeros(
                (indices.shape[0], indices.shape[1], self.lemma_vectors.shape[1]),
                dtype=self.lemma_vectors.dtype
            )
            
        return lemma_ids, lemma_vecs, scores

    @staticmethod
    def _to_numpy(state_vecs: np.ndarray | "torch.Tensor") -> np.ndarray:
        if isinstance(state_vecs, np.ndarray):
            array = state_vecs
        else:
            import torch

            if not torch.is_tensor(state_vecs):
                raise TypeError("state_vecs must be a numpy array or torch Tensor.")
            array = state_vecs.detach().cpu().numpy()
        if array.dtype != np.float32:
            array = array.astype(np.float32)
        return array


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return array / norms


def read_index_manifest(index_dir: str | Path) -> dict[str, Any]:
    """Return the manifest recorded beside an index, or an empty mapping."""
    path = Path(index_dir)
    manifest_path = path.parent / "manifest.json" if path.is_file() else path / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Index manifest '{manifest_path}' is not a JSON object.")
    return manifest


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_bytes(tensor: Any) -> bytes:
    """Raw bytes of a tensor, whatever its dtype (bfloat16 has no numpy)."""
    import torch

    t = tensor.detach().cpu().contiguous()
    try:
        return t.numpy().tobytes()
    except (TypeError, RuntimeError):
        return t.view(torch.int16).numpy().tobytes() if t.element_size() == 2 else t.view(torch.uint8).numpy().tobytes()


def state_dict_sha256(state_dict: Any) -> str:
    """Hash an encoder's weights, independent of file layout and key order.

    The same encoder weights can sit in a bare baseline checkpoint or under
    ``backbone.`` in a pointer checkpoint (and under ``module.`` when a run
    used DataParallel); the index may legitimately be built from either.
    Hashing the sorted (key, dtype, shape, bytes) sequence of the tensors
    identifies the *encoder*, not the file it happened to be saved in.
    """
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def encoder_state_from_checkpoint(state_dict: Any) -> dict[str, Any]:
    """Extract the encoder tensors from a baseline or pointer state dict.

    A ``TacticWithArgsClassifier`` holds its encoder in ``self.backbone``, so
    a pointer checkpoint's encoder keys carry that prefix alongside head keys
    (``argument_selector.*``, ``stop_head.*``, ``tactic_embedding.*``) that
    are no part of the encoder; a baseline checkpoint is the encoder and
    nothing else.  DataParallel runs add ``module.``.  Keeping only the
    ``backbone.`` keys when any are present yields the same bare-encoder key
    set either way, which is what makes the state hash comparable -- an
    index built from the baseline must verify against the pointer serving
    that same backbone.
    """
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        clean = str(key)
        if clean.startswith("module."):
            clean = clean[len("module."):]
        cleaned[clean] = value

    if any(key.startswith("backbone.") for key in cleaned):
        return {
            key[len("backbone."):]: value
            for key, value in cleaned.items()
            if key.startswith("backbone.")
        }
    return cleaned


def load_index_for_encoder(
    index_dir: str | Path,
    *,
    encoder_state_dict: Any,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    corpus_path: str | Path | None = None,
    expected_edge_mode: str | None = None,
) -> LemmaIndex:
    """Load an index only when it was built by exactly this encoder.

    An index is a pile of vectors in the space of the encoder that produced
    them.  Retraining the same architecture at the same width leaves every
    vector in a different space: retrieval returns confident nonsense, the
    dimensions still match, and nothing raises.  The manifest written by
    ``build_lemma_index`` binds the index to its encoder by a hash over the
    encoder's weight tensors; this loader enforces that binding against the
    caller's encoder, however that encoder was checkpointed.

    Raises ``ValueError`` when the manifest is absent (the index predates the
    binding and cannot be trusted to match anything) or when any hash
    disagrees.
    """
    manifest = read_index_manifest(index_dir)
    if not manifest:
        raise ValueError(
            f"Lemma index '{index_dir}' has no manifest.json, so it cannot be "
            "verified against the encoder it was built from. Rebuild it with "
            "scripts/build_lemma_index.py --checkpoint <encoder best.pt>."
        )

    encoder_sha = manifest.get("encoder_state_sha256")
    if not encoder_sha:
        raise ValueError(
            f"Index manifest '{index_dir}' records no encoder_state_sha256; it "
            "predates index-to-encoder binding. Rebuild the index."
        )

    actual_encoder_sha = state_dict_sha256(encoder_state_from_checkpoint(encoder_state_dict))
    if str(encoder_sha) != actual_encoder_sha:
        raise ValueError(
            f"Lemma index '{index_dir}' was built by a different encoder: "
            f"manifest expects state sha256 {str(encoder_sha)[:12]}…, the "
            f"checkpoint being loaded hashes {actual_encoder_sha[:12]}…. "
            "Retrieval against a mismatched encoder returns confident "
            "nonsense, so this is refused rather than warned about. Rebuild "
            "the index from the checkpoint you are serving."
        )

    from .training import _stable_vocab_sha256 as stable_vocab_sha256

    expected_node = manifest.get("node_vocab_sha256")
    expected_tactic = manifest.get("tactic_vocab_sha256")
    if not expected_node or not expected_tactic:
        raise ValueError(
            f"Lemma index '{index_dir}' does not record both vocabulary hashes; "
            "rebuild it before use."
        )
    if str(expected_node) != stable_vocab_sha256(node_vocab):
        raise ValueError(
            f"Lemma index '{index_dir}' was built with a different node "
            "vocabulary. The lemma graphs were encoded against different "
            "labels than the running model uses."
        )
    if str(expected_tactic) != stable_vocab_sha256(tactic_vocab):
        raise ValueError(
            f"Lemma index '{index_dir}' was built with a different tactic "
            "vocabulary."
        )

    if corpus_path is not None:
        expected_corpus = manifest.get("corpus_sha256")
        if not expected_corpus:
            raise ValueError(
                f"Lemma index '{index_dir}' records no corpus_sha256; rebuild it."
            )
        actual_corpus = _file_sha256(corpus_path)
        if str(expected_corpus) != actual_corpus:
            raise ValueError(
                f"Lemma index '{index_dir}' was built from a different lemma corpus."
            )
    if expected_edge_mode is not None:
        index_edge_mode = manifest.get("edge_mode")
        if index_edge_mode != expected_edge_mode:
            raise ValueError(
                f"Lemma index '{index_dir}' uses edge_mode={index_edge_mode!r}, "
                f"but the model uses {expected_edge_mode!r}."
            )

    return LemmaIndex.load(index_dir, manifest=manifest)
