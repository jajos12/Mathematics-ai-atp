"""Frozen text retrieval for a prepared graph cohort, without graph lemma keys.

The cache stores IDs and cosine scores. Lemma text vectors remain in the
checkpoint-bound text index and are gathered on demand, never copied per row.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .bundle import file_sha256
from .lemma_index import LemmaIndex
from .text_candidates import source_rows_digest


SCHEMA = "text_library_candidates_v1"


class CachedTextLibraryCandidates:
    def __init__(self, directory: str | Path, text_index: LemmaIndex, *,
                 checkpoint_path: str | Path | None = None) -> None:
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("invalid text-only library candidate schema")
        if (self.manifest.get("query_policy") != "goal_first_v1"
                or self.manifest.get("selection_policy") != "exact_cosine_no_gold_injection"):
            raise ValueError("text candidate query or selection policy differs")
        if self.manifest.get("text_index_manifest") != text_index.manifest:
            raise ValueError("candidate cache uses a different text lemma index")
        if checkpoint_path is not None and self.manifest.get("text_checkpoint_sha256") != file_sha256(Path(checkpoint_path)):
            raise ValueError("candidate cache uses a different text checkpoint")
        self.text_index = text_index
        self.tables: dict[str, dict[str, np.ndarray]] = {}
        for split, metadata in self.manifest["splits"].items():
            arrays = {}
            for field in ("rows", "ids", "scores"):
                path = self.directory / f"{split}_{field}.npy"
                if file_sha256(path) != metadata[f"{field}_sha256"]:
                    raise ValueError(f"text candidate table file changed: {path}")
                arrays[field] = np.load(path, mmap_mode="r")
            if (arrays["ids"].shape != arrays["scores"].shape
                    or arrays["ids"].shape != (int(metadata["count"]), int(self.manifest["k"]))
                    or arrays["rows"].shape != (int(metadata["count"]),)
                    or not np.issubdtype(arrays["ids"].dtype, np.integer)
                    or not np.issubdtype(arrays["scores"].dtype, np.floating)
                    or not np.all(np.diff(arrays["rows"]) > 0)):
                raise ValueError(f"invalid {split} text candidate table shapes or row order")
            self.tables[split] = arrays

    def validate_dataset(self, dataset, *, split: str) -> None:
        if (split not in self.tables
                or source_rows_digest(dataset) != self.manifest["splits"][split]["source_rows_sha256"]):
            raise ValueError(f"prepared {split} supervision differs from text candidate cache")

    def search(self, batch, *, k: int) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        """Fetch text vectors by stable lemma ID; account for PyG row_index offsets."""
        if not 1 <= k <= self.manifest["k"]:
            raise ValueError("requested K exceeds frozen text candidate budget")
        ids, scores, positions = [], [], []
        for graph in range(int(batch.y.numel())):
            split = str(batch.split[graph])
            table = self.tables.get(split)
            if table is None:
                raise ValueError(f"no frozen text candidates for split {split}")
            row_index = int(batch.row_index[graph]) - int(batch.ptr[graph])
            position = int(np.searchsorted(table["rows"], row_index))
            if position >= len(table["rows"]) or int(table["rows"][position]) != row_index:
                raise ValueError(f"text candidates absent for {split} row {row_index}")
            row_ids = [int(value) for value in table["ids"][position, :k]]
            if not np.isfinite(table["scores"][position, :k]).all():
                raise ValueError(f"nonfinite frozen text scores at {split} row {row_index}")
            try:
                positions.append([self.text_index.id_to_position[value] for value in row_ids])
            except KeyError as exc:
                raise ValueError(f"text candidate {exc.args[0]} absent from frozen text index") from exc
            ids.append(row_ids)
            scores.append(table["scores"][position, :k])
        vectors = self.text_index.lemma_vectors[np.asarray(positions, dtype=np.int64)]
        return ids, np.asarray(vectors, dtype=np.float32), np.asarray(scores, dtype=np.float32)
