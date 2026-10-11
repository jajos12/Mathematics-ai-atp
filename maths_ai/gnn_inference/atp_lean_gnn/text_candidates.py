"""Frozen text-selected IDs, with pointer-space vectors for mixed-pool scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bundle import file_sha256
from .lemma_index import LemmaIndex, state_dict_sha256
from .premise_retrieval import retrieval_metrics
from .text_retriever import TextDualEncoder


def source_rows_digest(dataset) -> str:
    """Bind table to prepared row identities and their trace supervision."""
    rows = sorted(dataset, key=lambda data: int(data.row_index))
    digest = hashlib.sha256()
    for data in rows:
        identity = (
            str(data.split), int(data.row_index), str(data.dataset_name),
            str(data.theorem), str(data.tactic_raw), int(data.arg_count),
            data.arg_node_indices.tolist(), data.arg_lemma_ids.tolist(),
        )
        digest.update(json.dumps(identity, separators=(",", ":")).encode())
    return digest.hexdigest()


def citation_coverage(ranked_rows, positive_rows, *, ks=(1, 10, 50, 200)):
    """Report all-known-citations coverage separately from the existing any-hit."""
    result = retrieval_metrics(ranked_rows, positive_rows, ks=ks).as_dict()
    labeled = [(ids, set(gold)) for ids, gold in zip(ranked_rows, positive_rows) if gold]
    positive_count = sum(len(gold) for _, gold in labeled)
    for k in ks:
        hits = [len(set(ids[:k]) & gold) for ids, gold in labeled]
        result[f"all_known_citations_at_{k}"] = sum(
            hit == len(gold) for hit, (_, gold) in zip(hits, labeled)
        ) / max(len(labeled), 1)
        result[f"citation_micro_recall_at_{k}"] = sum(hits) / max(positive_count, 1)
    return result


def load_frozen_text_retriever(checkpoint_path: Path, index_path: Path, corpus_path: Path, *, device):
    """Reject a changed checkpoint, tokenizer or stale text index before querying."""
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if saved.get("model_type") != "pretrained_text_dual_encoder" or saved.get("phase") != "complete":
        raise ValueError("text snapshot must be a completed pretrained text retriever checkpoint")
    corpus_sha = file_sha256(corpus_path)
    if saved.get("corpus_sha256") != corpus_sha:
        raise ValueError("text checkpoint was trained on a different lemma corpus")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(saved["model_name"])
    tokenizer_sha = hashlib.sha256(json.dumps(
        tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if tokenizer_sha != saved["tokenizer_sha256"]:
        raise ValueError("text tokenizer differs from checkpoint")
    model = TextDualEncoder.from_pretrained(saved["model_name"])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    index = LemmaIndex.load(index_path)
    expected = {
        "model_type": "pretrained_text_dual_encoder_index_v1",
        "model_name": saved["model_name"],
        "tokenizer_sha256": tokenizer_sha,
        "state_text_policy": "goal_first_v1",
        "corpus_sha256": corpus_sha,
        "lemma_encoder_sha256": state_dict_sha256(model.lemma_encoder.state_dict()),
        "max_length": saved["max_length"],
        "normalize": True,
    }
    if index.manifest != expected or saved.get("state_text_policy") != "goal_first_v1":
        raise ValueError("text lemma index does not match frozen retriever snapshot")
    model.to(device).eval().requires_grad_(False)
    return model, tokenizer, index, saved


class CachedTextCandidates:
    """Return frozen text ranking with pointer-index vectors, never text vectors."""

    def __init__(self, directory: Path, pointer_index: LemmaIndex) -> None:
        self.directory = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest.get("schema") != "text_candidates_v1":
            raise ValueError("invalid text candidate table schema")
        if self.manifest.get("pointer_index_manifest") != pointer_index.manifest:
            raise ValueError("candidate table uses different pointer-vector index")
        self.pointer_index = pointer_index
        self.tables = {}
        for split, metadata in self.manifest["splits"].items():
            arrays = {}
            for field in ("rows", "ids", "scores"):
                path = directory / f"{split}_{field}.npy"
                if file_sha256(path) != metadata[f"{field}_sha256"]:
                    raise ValueError(f"text candidate table file changed: {path}")
                arrays[field] = np.load(path, mmap_mode="r")
            if (arrays["ids"].shape != arrays["scores"].shape
                    or arrays["ids"].shape != (len(arrays["rows"]), self.manifest["k"])
                    or len(arrays["rows"]) != metadata["count"]
                    or not np.all(np.diff(arrays["rows"]) > 0)):
                raise ValueError("text candidate table has invalid row order or shape")
            self.tables[split] = arrays

    def validate_dataset(self, dataset, *, split: str) -> None:
        metadata = self.manifest["splits"].get(split)
        if metadata is None or source_rows_digest(dataset) != metadata["source_rows_sha256"]:
            raise ValueError(f"prepared {split} rows differ from text candidate table")

    def search(self, batch, *, k: int):
        if not 1 <= k <= self.manifest["k"]:
            raise ValueError("requested K exceeds frozen candidate-table budget")
        rows = []
        scores = []
        for graph in range(int(batch.y.numel())):
            split = str(batch.split[graph])
            table = self.tables.get(split)
            if table is None:
                raise ValueError(f"text candidates unavailable for split {split}")
            # PyG auto-offsets row_index because its name contains 'index'.
            row_index = int(batch.row_index[graph]) - int(batch.ptr[graph])
            position = int(np.searchsorted(table["rows"], row_index))
            if position >= len(table["rows"]) or int(table["rows"][position]) != row_index:
                raise ValueError(f"text candidates absent for {split} row {row_index}")
            ids = table["ids"][position, :k].tolist()
            if any(lemma_id not in self.pointer_index.id_to_position for lemma_id in ids):
                raise ValueError("text-selected lemma is absent from pointer-vector index")
            rows.append(ids)
            scores.append(table["scores"][position, :k])
        vectors = self.pointer_index.lemma_vectors[[
            [self.pointer_index.id_to_position[lemma_id] for lemma_id in ids] for ids in rows
        ]]
        return rows, vectors, np.asarray(scores, dtype=np.float32)
