"""Create new packed argument-label version while preserving graph tensors.

Uses digest-bound Lean source-syntax traces when present; otherwise resolves
bare source names against Hyp/FV graph structure and the verified name index.
Rows the flat-reference decoder cannot express remain explicit unsupported
rows. External citation labels are stored separately from action arguments.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import torch

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import ActionTargetError
from maths_ai.gnn_inference.atp_lean_gnn.argument_relabeling import POLICY, relabel_arguments
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.preparation import SourceSyntaxTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.training import load_prepared_metadata


def _write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _load_trace(trace_root, split, data):
    name = f"{int(data.row_index):09d}.json"
    trace_path = trace_root / split / "action_trace_v3" / name
    if not trace_path.exists():
        return None, None
    raw_path = trace_root / split / "sexpr" / name
    if not raw_path.exists():
        return None, "trace_without_raw_binding"
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None, "trace_unreadable"
    if not isinstance(trace, dict) or not isinstance(raw, dict):
        return None, "trace_invalid_object"
    for key, expected in (
        ("dataset", str(data.dataset_name)), ("split", split),
        ("row_index", int(data.row_index)), ("theorem", str(data.theorem)),
        ("tactic", str(data.tactic_raw)),
    ):
        if raw.get(key) != expected or trace.get(key) != expected:
            return None, "trace_source_mismatch"
    if (raw.get("schema_version") != 4 or trace.get("schema_version") != 3
            or trace.get("extractor_version") != SourceSyntaxTraceCache.EXTRACTOR_VERSION
            or trace.get("raw_record_sha256") != SourceSyntaxTraceCache.raw_record_sha256(raw)):
        return None, "trace_binding_mismatch"
    return trace, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--index-path", required=True, type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--edge-mode", default="bidirectional", choices=("forward", "bidirectional"))
    parser.add_argument("--max-args", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.source_root.resolve() == args.output_root.resolve() or args.max_args < 1:
        raise ValueError("output must be a new root and max_args positive")
    source = load_prepared_metadata(args.source_root)
    id_to_label = {value: name for name, value in source.node_vocab.items()}
    packed_root = args.source_root / "packed" / args.edge_mode
    source_manifest_path = packed_root / "manifest.json"
    packed = json.loads(source_manifest_path.read_text())
    index = LemmaIndex.load(args.index_path)
    if not index.name_to_id or len(index.lemma_names) != len(index.lemma_ids):
        raise ValueError("relabeling requires an index with complete declaration names")
    trace_root = args.trace_root or args.source_root
    binding = {
        "policy": POLICY, "max_args": args.max_args,
        "source_root": str(args.source_root.resolve()),
        "trace_root": str(trace_root.resolve()),
        "source_packed_manifest_sha256": file_sha256(source_manifest_path),
        "node_vocab_sha256": file_sha256(args.source_root / "vocab/node_vocab.json"),
        "tactic_vocab_sha256": file_sha256(args.source_root / "vocab/tactic_vocab.json"),
        "lemma_index_manifest": index.manifest,
        "edge_mode": args.edge_mode,
    }
    output = args.output_root
    if output.exists() and not args.resume:
        raise FileExistsError(f"new label root already exists: {output}; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    binding_path = output / "argument_labels.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("relabel resume source/index/policy differs")
    _write_json(binding_path, binding)
    (output / "vocab").mkdir(exist_ok=True)
    (output / "manifests").mkdir(exist_ok=True)
    (output / "reports").mkdir(exist_ok=True)
    for filename in ("node_vocab.json", "tactic_vocab.json"):
        shutil.copyfile(args.source_root / "vocab" / filename, output / "vocab" / filename)
    new_packed_root = output / "packed" / args.edge_mode
    new_packed_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for split, split_info in packed["splits"].items():
        destination = new_packed_root / split
        destination.mkdir(exist_ok=True)
        (output / split / "pyg").mkdir(parents=True, exist_ok=True)
        counts = Counter()
        chunks = []
        for filename in split_info["chunks"]:
            source_path = packed_root / split / filename
            output_path = destination / filename
            progress_path = destination / f"{filename}.json"
            source_sha = file_sha256(source_path)
            if progress_path.exists():
                progress = json.loads(progress_path.read_text())
                if (progress["source_sha256"] != source_sha
                        or not output_path.exists()
                        or progress["output_sha256"] != file_sha256(output_path)):
                    raise ValueError(f"source/output chunk changed during relabeling: {filename}")
                counts.update(progress["counts"])
                chunks.append(filename)
                continue
            graphs = torch.load(source_path, map_location="cpu", weights_only=False)
            converted = []
            chunk_counts = Counter()
            for data in graphs:
                trace, trace_error = _load_trace(trace_root, split, data)
                try:
                    changed = relabel_arguments(
                        data, node_vocab=source.node_vocab, name_to_id=index.name_to_id,
                        max_args=args.max_args, trace=trace, trace_error=trace_error,
                        id_to_label=id_to_label,
                    )
                except ActionTargetError as exc:
                    changed = relabel_arguments(
                        data, node_vocab=source.node_vocab, name_to_id=index.name_to_id,
                        max_args=args.max_args, trace_error=f"invalid_trace:{exc.code}",
                        id_to_label=id_to_label,
                    )
                changed.argument_trace_sha256 = (
                    SourceSyntaxTraceCache.raw_record_sha256(trace) if trace is not None else ""
                )
                chunk_counts["rows"] += 1
                chunk_counts[f"action:{changed.action_target_reason}"] += 1
                chunk_counts[f"citations:{changed.citation_label_policy}"] += 1
                chunk_counts["known_citation_count"] += changed.citation_count
                chunk_counts["trace_citations_outside_corpus"] += changed.citation_outside_corpus_count
                chunk_counts["rows_with_known_citations"] += int(changed.citation_count > 0)
                chunk_counts["supported_argument_count"] += int(changed.arg_count)
                converted.append(changed)
            torch.save(converted, output_path.with_suffix(".tmp"))
            output_path.with_suffix(".tmp").replace(output_path)
            _write_json(progress_path, {"source_sha256": source_sha,
                                       "output_sha256": file_sha256(output_path),
                                       "counts": dict(chunk_counts)})
            counts.update(chunk_counts)
            chunks.append(filename)
            print(f"{split}: relabeled {counts['rows']}/{split_info['count']}", flush=True)
        if counts["rows"] != split_info["count"]:
            raise ValueError(f"relabeling changed {split} row population")
        split_manifest = dict(source.split_manifest(split))
        split_manifest["argument_label_policy"] = POLICY
        split_manifest["argument_label_counts"] = dict(counts)
        split_manifest["artifact_paths"] = {**split_manifest["artifact_paths"], "pyg_dir": f"{split}/pyg"}
        _write_json(output / "manifests" / f"{split}.json", split_manifest)
        summaries[split] = dict(counts)
    _write_json(new_packed_root / "manifest.json", {**packed, "argument_label_policy": POLICY})
    _write_json(output / "reports/argument_relabeling.json", {"binding": binding, "splits": summaries})
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
