"""Freeze text-retrieved train/val candidates for pointer-space action scoring.

Only proof-state text selects candidates; gold citations are used solely for
coverage reports. Outputs contain IDs/scores, not 384-D text vectors. Scorer
looks up selected IDs in its verified 128-D pointer lemma index.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME, iter_dataset_rows
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import load_index_for_encoder
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import (
    citation_coverage, load_frozen_text_retriever, source_rows_digest,
)
from maths_ai.gnn_inference.atp_lean_gnn.text_retriever import matched_text_rows, tokenize_texts
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PreparedGraphDataset, REQUIRED_POINTER_DATA_FIELDS,
    load_pointer_config, load_prepared_metadata, resolve_device,
)


def _atomic_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "text-index", "pointer-checkpoint", "pointer-index", "config",
                 "prepared-root", "corpus-path", "output-dir"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.k < 200:
        raise ValueError("batch size must be positive; K must be at least 200 for coverage report")
    device = resolve_device(args.device)
    model, tokenizer, text_index, saved = load_frozen_text_retriever(
        args.checkpoint, args.text_index, args.corpus_path, device=device,
    )
    if saved["dataset_name"] != args.dataset_name:
        raise ValueError("text checkpoint and upstream dataset differ")
    config = load_pointer_config(args.config)
    metadata = load_prepared_metadata(args.prepared_root)
    pointer = torch.load(args.pointer_checkpoint, map_location="cpu", weights_only=False)
    pointer_state = pointer.get("model_state_dict", pointer)
    pointer_index = load_index_for_encoder(
        args.pointer_index, encoder_state_dict=pointer_state,
        node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        corpus_path=args.corpus_path, expected_edge_mode=config.edge_mode,
    )
    if (text_index.lemma_ids != pointer_index.lemma_ids
            or text_index.lemma_names != pointer_index.lemma_names):
        raise ValueError("text and pointer indices disagree on declaration identity")
    if args.k > len(text_index.lemma_ids):
        raise ValueError("K exceeds corpus size")
    binding = {
        "schema": "text_candidates_v1", "k": args.k,
        "text_checkpoint_sha256": file_sha256(args.checkpoint),
        "text_epoch": int(saved["epoch"]),
        "text_index_manifest": text_index.manifest,
        "pointer_index_manifest": pointer_index.manifest,
        "dataset_name": args.dataset_name,
        "corpus_sha256": saved["corpus_sha256"],
        "query_policy": "goal_first_v1",
        "selection_policy": "exact_cosine_no_gold_injection",
    }
    root = args.output_dir
    if root.exists() and not args.resume:
        raise FileExistsError(f"candidate output exists: {root}; pass --resume")
    root.mkdir(parents=True, exist_ok=True)
    binding_path = root / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("candidate preparation resume uses a different retriever or pointer index")
    _atomic_json(binding_path, binding)
    splits = {}
    for split in ("val", "train"):
        dataset = PreparedGraphDataset(
            metadata, split=split, edge_mode=config.edge_mode, cache_in_memory=True,
            required_fields=REQUIRED_POINTER_DATA_FIELDS + ("arg_lemma_ids", "arg_count"),
        )
        rows = sorted(dataset, key=lambda data: int(data.row_index))
        source_sha = source_rows_digest(rows)
        pairs = matched_text_rows(
            rows, iter_dataset_rows(dataset_name=args.dataset_name, split=split),
            name_to_id=text_index.name_to_id,
        )
        row_ids = np.asarray([int(data.row_index) for data in rows], dtype=np.int64)
        np.save(root / f"{split}_rows.npy", row_ids)
        progress_path = root / f"{split}_progress.json"
        progress = {"source_rows_sha256": source_sha, "next_row": 0}
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["source_rows_sha256"] != source_sha:
                raise ValueError(f"prepared {split} source changed during candidate preparation")
        start_row = int(progress["next_row"])
        if not 0 <= start_row <= len(rows):
            raise ValueError("invalid candidate preparation progress")
        mode = "r+" if progress_path.exists() else "w+"
        ids = np.lib.format.open_memmap(
            root / f"{split}_ids.npy", mode=mode, dtype=np.int64,
            shape=(len(rows), args.k),
        )
        scores = np.lib.format.open_memmap(
            root / f"{split}_scores.npy", mode=mode, dtype=np.float32,
            shape=(len(rows), args.k),
        )
        for start in range(start_row, len(rows), args.batch_size):
            end = min(start + args.batch_size, len(rows))
            with torch.no_grad():
                queries = model.encode_states(tokenize_texts(
                    tokenizer, [text for text, _ in pairs[start:end]],
                    device=device, max_length=saved["max_length"],
                )).float().cpu().numpy()
            batch_scores, positions = text_index.index.search(
                np.ascontiguousarray(queries), args.k,
            )
            if (positions < 0).any():
                raise ValueError("text retrieval returned an invalid corpus position")
            ids[start:end] = np.asarray(text_index.lemma_ids)[positions]
            scores[start:end] = batch_scores
            ids.flush()
            scores.flush()
            _atomic_json(progress_path, {"source_rows_sha256": source_sha, "next_row": end})
            if start // args.batch_size % 100 == 0:
                print(f"{split}: cached {end}/{len(rows)} source states", flush=True)
        metrics = citation_coverage(ids, [gold for _, gold in pairs])
        print(f"{split} frozen epoch {saved['epoch']} coverage:\n{json.dumps(metrics, indent=2)}", flush=True)
        splits[split] = {
            "count": len(rows), "source_rows_sha256": source_sha,
            **{f"{field}_sha256": file_sha256(root / f"{split}_{field}.npy")
               for field in ("rows", "ids", "scores")},
            "retrieval_metrics": metrics,
        }
        _atomic_json(root / f"{split}_metrics.json", metrics)
        del rows, dataset, pairs, ids, scores
    _atomic_json(root / "manifest.json", {**binding, "splits": splits})
    print(f"Frozen candidate tables ready at {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
