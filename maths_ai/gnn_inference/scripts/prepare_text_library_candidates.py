"""Cache frozen text-only lemma IDs and scores for corrected graph rows.

The existing text candidate script binds old graph-pointer vectors. This one
binds the text index alone and leaves its lemma vectors in one shared index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME, iter_dataset_rows
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import (
    citation_coverage, load_frozen_text_retriever, source_rows_digest,
)
from maths_ai.gnn_inference.atp_lean_gnn.text_library_candidates import SCHEMA
from maths_ai.gnn_inference.atp_lean_gnn.text_retriever import matched_text_rows, tokenize_texts
from maths_ai.gnn_inference.atp_lean_gnn.training import PreparedGraphDataset, load_prepared_metadata


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _open_arrays(root: Path, split: str, *, count: int, k: int, resume: bool):
    specs = (("ids", np.int32, (count, k)), ("scores", np.float32, (count, k)))
    result = {}
    for name, dtype, shape in specs:
        path = root / f"{split}_{name}.npy"
        if resume:
            if not path.exists():
                raise ValueError(f"missing partial candidate table: {path}")
            result[name] = np.load(path, mmap_mode="r+")
            if result[name].shape != shape or result[name].dtype != dtype:
                raise ValueError(f"resumed candidate table shape or dtype changed: {path}")
        else:
            result[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "text-index", "prepared-root", "corpus-path", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--splits", default="val,train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--faiss-threads", type=int, default=4)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.k, args.faiss_threads) < 1:
        raise ValueError("batch size, K, and FAISS threads must be positive")
    import faiss

    faiss.omp_set_num_threads(args.faiss_threads)
    splits = tuple(dict.fromkeys(x.strip() for x in args.splits.split(",") if x.strip()))
    if not splits or any(x not in {"train", "val"} for x in splits):
        raise ValueError("candidate cache supports train and/or val only")
    device = torch.device(args.device)
    model, tokenizer, text_index, saved = load_frozen_text_retriever(
        args.checkpoint, args.text_index, args.corpus_path, device=device,
    )
    if saved["dataset_name"] != args.dataset_name:
        raise ValueError("frozen retriever and source dataset differ")
    if (args.k > len(text_index.lemma_ids) or min(text_index.lemma_ids) < 0
            or max(text_index.lemma_ids) > np.iinfo(np.int32).max):
        raise ValueError("invalid K or lemma ID outside int32 range")
    label_path = args.prepared_root / "argument_labels.json"
    packed_path = args.prepared_root / "packed/bidirectional/manifest.json"
    binding = {
        "schema": SCHEMA, "k": args.k,
        "prepared_root": str(args.prepared_root.resolve()),
        "argument_label_binding_sha256": file_sha256(label_path),
        "prepared_packed_manifest_sha256": file_sha256(packed_path),
        "node_vocab_sha256": file_sha256(args.prepared_root / "vocab/node_vocab.json"),
        "tactic_vocab_sha256": file_sha256(args.prepared_root / "vocab/tactic_vocab.json"),
        "text_checkpoint_sha256": file_sha256(args.checkpoint),
        "text_epoch": int(saved["epoch"]),
        "text_index_manifest": text_index.manifest,
        "dataset_name": args.dataset_name,
        "corpus_sha256": saved["corpus_sha256"],
        "query_policy": "goal_first_v1",
        "selection_policy": "exact_cosine_no_gold_injection",
        "faiss_threads": args.faiss_threads,
        "splits_requested": list(splits),
    }
    root = args.output_dir
    if root.exists() and not args.resume:
        raise FileExistsError(f"candidate output already exists: {root}; pass --resume")
    root.mkdir(parents=True, exist_ok=True)
    binding_path = root / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("candidate cache resume uses different source or retrieval policy")
    if args.resume and any(root.iterdir()) and not binding_path.exists():
        raise ValueError("partial candidate cache has no source binding")
    _atomic_json(binding_path, binding)
    metadata = load_prepared_metadata(args.prepared_root, splits=splits)
    split_reports = {}
    lemma_ids = np.asarray(text_index.lemma_ids, dtype=np.int32)
    for split in splits:
        dataset = PreparedGraphDataset(metadata, split=split, cache_in_memory=True)
        if not dataset.packed_cache_loaded:
            raise ValueError(f"{split} requires a complete packed corrected-label cache")
        rows = sorted(dataset, key=lambda graph: int(graph.row_index))
        pairs = matched_text_rows(
            rows, iter_dataset_rows(dataset_name=args.dataset_name, split=split),
            name_to_id=text_index.name_to_id,
        )
        row_ids = np.asarray([int(graph.row_index) for graph in rows], dtype=np.int64)
        source_sha = source_rows_digest(rows)
        row_path = root / f"{split}_rows.npy"
        if row_path.exists() and not np.array_equal(np.load(row_path), row_ids):
            raise ValueError(f"{split} cached source row IDs changed")
        if not row_path.exists():
            np.save(row_path, row_ids)
        progress_path = root / f"{split}_progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
        if progress is not None and progress.get("source_rows_sha256") != source_sha:
            raise ValueError(f"{split} cached supervision changed during resume")
        start = int(progress["next_row"]) if progress else 0
        if not 0 <= start <= len(rows):
            raise ValueError(f"{split} invalid cached progress")
        arrays = _open_arrays(root, split, count=len(rows), k=args.k, resume=progress is not None)
        if progress is None:
            _atomic_json(progress_path, {"source_rows_sha256": source_sha, "next_row": 0})
        for offset in range(start, len(rows), args.batch_size):
            end = min(offset + args.batch_size, len(rows))
            with torch.no_grad():
                queries = model.encode_states(tokenize_texts(
                    tokenizer, [text for text, _ in pairs[offset:end]],
                    device=device, max_length=saved["max_length"],
                )).float().cpu().numpy()
            scores, positions = text_index.index.search(np.ascontiguousarray(queries), args.k)
            if (positions < 0).any():
                raise ValueError("retrieval returned an invalid lemma position")
            arrays["ids"][offset:end] = lemma_ids[positions]
            arrays["scores"][offset:end] = scores
            arrays["ids"].flush()
            arrays["scores"].flush()
            _atomic_json(progress_path, {"source_rows_sha256": source_sha, "next_row": end})
            if end == len(rows) or (offset // args.batch_size) % 100 == 0:
                print(f"{split}: cached {end}/{len(rows)} states", flush=True)
        coverage = citation_coverage(arrays["ids"], [gold for _, gold in pairs])
        _atomic_json(root / f"{split}_metrics.json", coverage)
        split_reports[split] = {
            "count": len(rows), "source_rows_sha256": source_sha,
            **{f"{field}_sha256": file_sha256(root / f"{split}_{field}.npy")
               for field in ("rows", "ids", "scores")},
            "retrieval_metrics": coverage,
        }
        print(f"{split} frozen epoch {saved['epoch']}: {json.dumps(coverage)}", flush=True)
        del dataset, rows, pairs, arrays
    _atomic_json(root / "manifest.json", {**binding, "splits": split_reports})
    print(f"Frozen text-only library candidates ready: {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
