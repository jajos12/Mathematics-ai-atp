"""Experimental, full-corpus premise-retrieval index and reporting helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .lemma_corpus import LemmaRecord
from .lemma_index import LemmaIndex
from .premise_retrieval import retrieval_metrics


def encode_corpus_index(
    records: Sequence[LemmaRecord],
    encode: Callable[[Sequence[int]], Tensor],
    *,
    batch_size: int,
    manifest: dict[str, object],
    output_dir: Path | None = None,
) -> LemmaIndex:
    """Re-encode every indexed lemma with current tower; use exact cosine search.

    Never reuse frozen index vectors after lemma-tower updates. Write index only
    after all vectors exist so interrupted builds cannot look complete.
    """
    if batch_size < 1 or not records:
        raise ValueError("batch_size must be positive and corpus nonempty")
    import faiss

    ids = [record.lemma_id for record in records]
    names = [record.name for record in records]
    ids_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    vectors = []
    start_row = 0
    matrix = None
    progress_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        progress_path = output_dir / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("manifest") != manifest or progress.get("ids_sha256") != ids_sha:
                raise ValueError("unfinished index was built for a different lemma tower/corpus")
            start_row = int(progress["next_row"])
            matrix = np.load(output_dir / "lemma_vectors.npy", mmap_mode="r+")
            if not 0 <= start_row <= len(ids) or matrix.shape[0] != len(ids):
                raise ValueError("unfinished index progress disagrees with vector file")
            print(f"Resuming lemma index after {start_row}/{len(ids)} vectors", flush=True)
    for start in range(start_row, len(ids), batch_size):
        block = ids[start : start + batch_size]
        with torch.no_grad():
            encoded = encode(block).detach().float().cpu().numpy()
        if encoded.ndim != 2 or encoded.shape[0] != len(block):
            raise ValueError("lemma encoder output count does not match corpus batch")
        if not np.isfinite(encoded).all():
            raise ValueError("non-finite lemma vector during index rebuild")
        norms = np.linalg.norm(encoded, axis=1, keepdims=True)
        if (norms <= 1e-12).any():
            raise ValueError("zero-length lemma vector during index rebuild")
        encoded = np.ascontiguousarray(encoded / norms)
        if output_dir is None:
            vectors.append(encoded)
            continue
        if matrix is None:
            matrix = np.lib.format.open_memmap(
                output_dir / "lemma_vectors.npy", mode="w+", dtype=np.float32,
                shape=(len(ids), encoded.shape[1]),
            )
        if encoded.shape[1] != matrix.shape[1]:
            raise ValueError("lemma encoder dimension changed mid-index")
        matrix[start:start + len(block)] = encoded
        matrix.flush()
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "manifest": manifest, "ids_sha256": ids_sha,
            "next_row": start + len(block),
        }), encoding="utf-8")
        os.replace(temporary, progress_path)
    if output_dir is None:
        matrix = np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32)
    if matrix is None:
        raise ValueError("index has no encoded lemma vectors")
    index = faiss.IndexFlatIP(matrix.shape[1])
    for start in range(0, len(ids), batch_size):
        index.add(np.ascontiguousarray(matrix[start:start + batch_size]))
    result = LemmaIndex(
        index, ids, matrix, lemma_names=names,
        normalize_queries=True, manifest=manifest,
    )
    if output_dir is not None:
        (output_dir / "lemma_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        (output_dir / "lemma_names.json").write_text(json.dumps(names), encoding="utf-8")
        faiss.write_index(index, str(output_dir / "faiss.index"))
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        progress_path.unlink(missing_ok=True)
    return result


def evaluate_index(
    index: LemmaIndex,
    texts_or_batches,
    *,
    encode_queries: Callable[[object], Tensor],
    extract_positives: Callable[[object], list[list[int]]],
    ks: tuple[int, ...] = (1, 10, 50, 200),
) -> dict[str, int | float]:
    """Use same any-positive/covered-step denominators as frozen retriever."""
    ranked: list[list[int]] = []
    positives: list[list[int]] = []
    for batch in texts_or_batches:
        with torch.no_grad():
            query = F.normalize(encode_queries(batch), dim=-1)
        scores, positions = index.index.search(
            np.ascontiguousarray(query.detach().float().cpu().numpy()), max(ks)
        )
        del scores
        ranked.extend([
            [index.lemma_ids[int(pos)] for pos in row if pos >= 0]
            for row in positions
        ])
        positives.extend(extract_positives(batch))
    return retrieval_metrics(ranked, positives, ks=ks).as_dict()
