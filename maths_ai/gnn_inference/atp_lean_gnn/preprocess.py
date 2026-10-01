from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from .cache import (
    SplitReport,
    append_failure_record,
    build_failure_record,
    build_json_payload,
    build_summary,
    prepare_output_root,
    write_json_artifact,
    write_manifest,
    write_pyg_artifact,
    write_summary_json,
    write_summary_markdown,
    write_vocab,
)
from .dataset import DATASET_NAME, DatasetRow, canonicalize_split_name, iter_dataset_rows
from .preparation import (
    ModelSExprCache,
    prepare_example,
    SExprCache,
)
from .pilot_sampling import load_selection_manifest, selected_row_indices
from .labels import build_tactic_vocab, encode_tactic_name, label_example
from .lemma_corpus import load_lemma_name_index
from .graph import root_state_node_id
from .pyg import build_vocab_from_labels, dag_to_pyg
from .reporting import console_print


DEFAULT_OUTPUT_ROOT = Path("artifacts") / "prepared" / "v1"


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_name: str = "cat-searcher/leandojo-benchmark-4-random"
    splits: tuple[str, ...] = ("train", "val", "test")
    output_root: Path = Path("artifacts") / "prepared" / "v1"
    sample_per_split: int | None = None
    lemma_corpus_path: Path | None = None
    force: bool = False
    resume: bool = False
    write_json_artifacts: bool = True
    use_sexpr: bool = False
    sexpr_cache_root: Path | None = None
    sexpr_variant: str = "model"
    project_path: str = "maths_ai/lean_mathlib"
    selection_manifest: Path | None = None


def _normalize_splits(raw_splits: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(raw_splits, str):
        candidates = [part.strip() for part in raw_splits.split(",")]
    else:
        candidates = [part.strip() for part in raw_splits]

    splits: list[str] = []
    for split in candidates:
        if not split:
            continue
        canonical_split = canonicalize_split_name(split)
        if canonical_split not in splits:
            splits.append(canonical_split)

    if not splits:
        raise ValueError("At least one split must be provided.")
    if "train" not in splits:
        raise ValueError("The requested splits must include 'train' so train-only vocabularies can be built.")
    return ["train", *[split for split in splits if split != "train"]]


def _load_rows(
    dataset_name: str,
    split: str,
    sample_limit: int | None = None,
    selection_manifest: Path | None = None,
) -> list[DatasetRow]:
    """Load all rows for a split into memory, grouped by order."""
    if selection_manifest is not None and sample_limit is not None:
        raise ValueError("selection_manifest and sample_limit cannot be combined.")
    source_split = split
    if selection_manifest is not None:
        payload = load_selection_manifest(selection_manifest)
        split_payload = payload["splits"].get(split)
        if not isinstance(split_payload, dict):
            raise ValueError(
                f"Pilot selection manifest has no '{split}' split."
            )
        source_split = str(split_payload.get("source_split", split))
    rows = list(iter_dataset_rows(
        dataset_name=dataset_name,
        split=source_split,
        sample_limit=sample_limit,
    ))
    if selection_manifest is None:
        return rows
    selected = selected_row_indices(
        selection_manifest, split, dataset_name=dataset_name
    )
    filtered = [row for row in rows if row.row_index in selected]
    missing = selected - {row.row_index for row in filtered}
    if missing:
        raise RuntimeError(
            f"Selection manifest contains {len(missing)} row indices "
            f"not present in split '{split}'."
        )
    return filtered


def _load_selected_rows_once(
    dataset_name: str,
    splits: tuple[str, ...],
    selection_manifest: Path,
) -> dict[str, list[DatasetRow]]:
    """Load theorem-holdout partitions with one dataset scan per source split."""
    payload = load_selection_manifest(selection_manifest)
    requested: dict[str, tuple[str, set[int]]] = {}
    by_source: dict[str, set[int]] = {}
    for split in splits:
        split_payload = payload["splits"].get(split)
        if not isinstance(split_payload, dict):
            raise ValueError(f"Pilot selection manifest has no '{split}' split.")
        source_split = str(split_payload.get("source_split", split))
        indices = selected_row_indices(
            selection_manifest, split, dataset_name=dataset_name
        )
        requested[split] = (source_split, indices)
        by_source.setdefault(source_split, set()).update(indices)

    source_rows: dict[str, dict[int, DatasetRow]] = {}
    for source_split, wanted in by_source.items():
        found: dict[int, DatasetRow] = {}
        for row in iter_dataset_rows(dataset_name=dataset_name, split=source_split):
            if row.row_index in wanted:
                found[row.row_index] = row
                if len(found) == len(wanted):
                    break
        missing = wanted - found.keys()
        if missing:
            raise RuntimeError(
                f"Selection manifest contains {len(missing)} row indices not "
                f"present in source split '{source_split}'."
            )
        source_rows[source_split] = found

    return {
        split: [source_rows[source][index] for index in sorted(indices)]
        for split, (source, indices) in requested.items()
    }


def _build_sexpr_map(
    rows: list[DatasetRow],
    project_path: str,
    use_sexpr: bool,
    sexpr_cache: Optional[SExprCache] = None,
    split_label: str = "",
    sexpr_variant: str = "model",
    require_complete: bool = False,
) -> dict[int, dict]:
    """Load validated Phase 2 records without silently generating or falling back."""
    if not use_sexpr:
        return {}

    sexpr_map: dict[int, dict] = {}

    if sexpr_cache is None:
        raise ValueError("S-expression mode requires a Phase 2 cache root.")
    if sexpr_variant not in {"raw", "model"}:
        raise ValueError("sexpr_variant must be either 'raw' or 'model'.")
    model_cache = (
        ModelSExprCache(sexpr_cache.output_root)
        if sexpr_variant == "model"
        else None
    )
    theorem_rows: dict[str, list[DatasetRow]] = {}
    for row in rows:
        theorem_rows.setdefault(row.theorem, []).append(row)
    for grouped_rows in theorem_rows.values():
        for row in sorted(grouped_rows, key=lambda item: item.row_index):
            raw = sexpr_cache.load_for_row(
                row, extractor_version=SExprCache.EXTRACTOR_VERSION
            )
            cached = (
                model_cache.load_for_raw_record(row.split, row.row_index, raw)
                if model_cache is not None and raw is not None
                else raw
            )
            if cached is not None:
                sexpr_map[row.row_index] = cached

    console_print(
        f"    Validated {sexpr_variant} S-expression cache: "
        f"{len(sexpr_map)}/{len(rows)} rows"
    )
    if require_complete and len(sexpr_map) != len(rows):
        raise RuntimeError(
            f"Selected {sexpr_variant} S-expression cache is incomplete for "
            f"split '{split_label}': {len(rows) - len(sexpr_map)} missing rows. "
            "Refusing to create a mixed-representation ablation."
        )

    return sexpr_map


def scan_train_split(
    *,
    dataset_name: str,
    sample_per_split: int | None,
    output_root: Path,
    sexpr_cache: Optional[SExprCache],
    project_path: str = "maths_ai/lean_mathlib",
    use_sexpr: bool = False,
    sexpr_variant: str = "model",
    selection_manifest: Path | None = None,
    rows: list[DatasetRow] | None = None,
) -> tuple[dict[str, int], dict[str, int], SplitReport]:
    node_labels: set[str] = set()
    tactic_names: list[str] = []
    report = SplitReport(split="train")

    if rows is None:
        rows = _load_rows(
            dataset_name, "train", sample_per_split, selection_manifest
        )
    sexpr_map = _build_sexpr_map(
        rows,
        project_path,
        use_sexpr,
        sexpr_cache,
        "train",
        sexpr_variant,
        require_complete=selection_manifest is not None,
    )

    total_rows = len(rows)
    for position, row in enumerate(rows, start=1):
        if position == 1 or position % 1_000 == 0 or position == total_rows:
            console_print(
                f"    Train vocabulary scan: {position}/{total_rows} rows"
            )
        try:
            sd = sexpr_map.get(row.row_index)
            example = prepare_example(
                row,
                sexpr_cache=None if use_sexpr else sexpr_cache,
                sexpr_data=sd,
                use_sexpr=use_sexpr,
            )
        except Exception as exc:
            failure_record = build_failure_record(row, exc)
            report.record_failure(
                category=str(failure_record["failure_category"]),
                phase=str(failure_record["phase"]),
            )
            continue

        report.record_success(dag=example.dag, tactic_name=example.tactic_name)
        node_labels.update(node.label for node in example.dag.nodes)
        tactic_names.append(example.tactic_name)

    if report.success_count == 0:
        raise RuntimeError("The train split produced zero successful examples while building vocabularies.")

    node_vocab = build_vocab_from_labels(node_labels)
    tactic_vocab = build_tactic_vocab(tactic_names)
    return node_vocab, tactic_vocab, report


def _resolve_arg_node_indices(dag, arg_tokens: list[str]) -> list[int]:
    """Best-effort: match each argument token to a DAG node index by label.

    Returns a list of node indices (one per argument token), using ``-1``
    when no matching node is found in the graph.
    """
    # Build label → node_id map (first match wins)
    label_to_id: dict[str, int] = {}
    for node in dag.nodes:
        if node.label not in label_to_id:
            label_to_id[node.label] = node.id

    return [label_to_id.get(token, -1) for token in arg_tokens]


def _resolve_arg_lemma_ids(
    arg_tokens: list[str],
    lemma_name_index: dict[str, int] | None,
) -> list[int]:
    if not arg_tokens:
        return []
    if lemma_name_index is None:
        return [-1 for _ in arg_tokens]
    return [lemma_name_index.get(token, -1) for token in arg_tokens]


def _load_resumable_artifact(
    path: Path,
    *,
    row: DatasetRow,
    split: str,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
):
    """Load an existing PyG artifact only when it matches this exact run."""
    import torch

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        tactic_name = str(label_example(row)["tactic_name"])
        expected_y = encode_tactic_name(tactic_name, tactic_vocab)
        if (
            getattr(data, "dataset_name", None) != row.dataset_name
            or getattr(data, "split", None) != split
            or getattr(data, "row_index", None) != row.row_index
            or getattr(data, "theorem", None) != row.theorem
            or getattr(data, "tactic_raw", None) != row.tactic
            or getattr(data, "tactic_name", None) != tactic_name
            or not hasattr(data, "x")
            or not hasattr(data, "edge_index")
            or not hasattr(data, "y")
            or data.x.numel() == 0
            or int(data.x.min().item()) < 0
            or int(data.x.max().item()) >= len(node_vocab)
            or data.y.numel() != 1
            or int(data.y.item()) != expected_y
            or data.edge_index.dim() != 2
            or data.edge_index.size(0) != 2
        ):
            return None
        node_count = int(data.num_nodes)
        edge_count = int(data.edge_index.size(1))
        if data.edge_index.numel() and (
            int(data.edge_index.min().item()) < 0
            or int(data.edge_index.max().item()) >= node_count
        ):
            return None
        return data, tactic_name
    except Exception:
        return None


def process_split(
    *,
    dataset_name: str,
    split: str,
    sample_per_split: int | None,
    output_root: Path,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    lemma_name_index: dict[str, int] | None,
    sexpr_cache: Optional[SExprCache],
    project_path: str = "maths_ai/lean_mathlib",
    use_sexpr: bool = False,
    sexpr_variant: str = "model",
    selection_manifest: Path | None = None,
    rows: list[DatasetRow] | None = None,
    resume: bool = False,
    write_json_artifacts: bool = True,
) -> tuple[SplitReport, dict[str, object]]:
    import torch

    from .labels import parse_tactic_arguments
    from .pyg import build_premise_mask

    report = SplitReport(split=split)
    resumed_artifact_count = 0

    if rows is None:
        rows = _load_rows(
            dataset_name, split, sample_per_split, selection_manifest
        )
    sexpr_map = _build_sexpr_map(
        rows,
        project_path,
        use_sexpr,
        sexpr_cache,
        split,
        sexpr_variant,
        require_complete=selection_manifest is not None,
    )

    total_rows = len(rows)
    for position, row in enumerate(rows, start=1):
        if position == 1 or position % 1_000 == 0 or position == total_rows:
            console_print(
                f"    {split} artifacts: {position}/{total_rows} rows"
            )
        pyg_path = (
            output_root / split / "pyg" / f"{row.row_index:09d}.pt"
        )
        json_path = (
            output_root / split / "json" / f"{row.row_index:09d}.json"
        )
        if resume and pyg_path.exists() and (
            not write_json_artifacts or json_path.exists()
        ):
            cached = _load_resumable_artifact(
                pyg_path,
                row=row,
                split=split,
                node_vocab=node_vocab,
                tactic_vocab=tactic_vocab,
            )
            if cached is not None:
                data, tactic_name = cached
                resumed_artifact_count += 1
                report.record_cached_success(
                    node_count=int(data.num_nodes),
                    edge_count=int(data.edge_index.size(1)),
                    reused_node_count=int(
                        (
                            torch.bincount(
                                data.edge_index[0], minlength=int(data.num_nodes)
                            )
                            > 1
                        ).sum().item()
                    ),
                    tactic_name=tactic_name,
                )
                continue

        try:
            sd = sexpr_map.get(row.row_index)
            example = prepare_example(
                row,
                sexpr_cache=None if use_sexpr else sexpr_cache,
                sexpr_data=sd,
                use_sexpr=use_sexpr,
            )
        except Exception as exc:
            failure_record = build_failure_record(row, exc)
            append_failure_record(output_root, split=split, record=failure_record)
            report.record_failure(
                category=str(failure_record["failure_category"]),
                phase=str(failure_record["phase"]),
            )
            continue

        if write_json_artifacts:
            json_payload = build_json_payload(
                row,
                parsed_state=example.parsed_state,
                dag=example.dag,
                tactic_name=example.tactic_name,
            )
            write_json_artifact(
                output_root,
                split=split,
                row_index=row.row_index,
                payload=json_payload,
            )

        dag = example.dag
        tactic_name = example.tactic_name
        data = dag_to_pyg(dag, node_vocab)
        data.y = torch.tensor(
            [encode_tactic_name(tactic_name, tactic_vocab)],
            dtype=torch.long,
        )
        data.split = split
        data.row_index = row.row_index
        data.dataset_name = row.dataset_name
        data.theorem = row.theorem
        data.tactic_raw = row.tactic
        data.tactic_name = tactic_name

        # Materialize the readout node once. Older prepared datasets omit this
        # field and remain supported by PreparedGraphDataset's fallback.
        data.state_node_index = torch.tensor(
            [root_state_node_id(dag)], dtype=torch.long
        )

        # --- Argument-selection ground truth (additive) ---------------
        premise_mask = build_premise_mask(dag)
        data.premise_mask = torch.tensor(premise_mask, dtype=torch.bool)

        _, arg_tokens = parse_tactic_arguments(row.tactic)
        arg_indices = _resolve_arg_node_indices(dag, arg_tokens)
        arg_lemma_ids = _resolve_arg_lemma_ids(arg_tokens, lemma_name_index)
        for idx, node_id in enumerate(arg_indices):
            if node_id >= 0 and idx < len(arg_lemma_ids):
                arg_lemma_ids[idx] = -1
        data.arg_node_indices = torch.tensor(arg_indices, dtype=torch.long) if arg_indices else torch.tensor([], dtype=torch.long)
        data.arg_lemma_ids = torch.tensor(arg_lemma_ids, dtype=torch.long) if arg_lemma_ids else torch.tensor([], dtype=torch.long)
        data.arg_count = len(arg_indices)
        # --------------------------------------------------------------

        write_pyg_artifact(
            output_root,
            split=split,
            row_index=row.row_index,
            data=data,
        )

        report.record_success(dag=dag, tactic_name=tactic_name)

    if report.success_count == 0:
        raise RuntimeError(f"Split '{split}' produced zero successful examples.")

    manifest = report.to_manifest(
        dataset_name=dataset_name,
        output_root=output_root,
        vocab_source="train",
        sample_limit=sample_per_split,
    )
    manifest["json_artifacts_enabled"] = write_json_artifacts
    manifest["resume_enabled"] = resume
    manifest["resumed_artifact_count"] = resumed_artifact_count

    if selection_manifest is not None:
        manifest["selection_manifest"] = str(selection_manifest)
    write_manifest(output_root, split=split, manifest=manifest)
    return report, manifest


def run_preprocessing(config: PreprocessConfig) -> dict[str, object]:
    output_root = Path(config.output_root)
    if config.force and config.resume:
        raise ValueError("--force and --resume cannot be combined.")
    if output_root.exists() and not config.force and not config.resume:
        raise FileExistsError(
            f"Output root '{output_root}' already exists. "
            "Re-run with --force to overwrite or --resume to reuse valid PyG artifacts."
        )

    sexpr_cache = None
    if config.use_sexpr:
        if config.sexpr_cache_root is None:
            raise ValueError(
                "S-expression preprocessing requires --sexpr-cache-root from "
                "the completed theorem-replay extraction stage."
            )
        if config.sexpr_cache_root.resolve() == output_root.resolve():
            raise ValueError(
                "--sexpr-cache-root must differ from --output-root because "
                "preprocessing replaces the output directory."
            )
        console_print("  S-expression mode: consuming validated theorem-replay cache")
        sexpr_cache = SExprCache(
            config.sexpr_cache_root,
            config.project_path,
            enabled=True,
        )

    selected_rows = None
    if config.selection_manifest is not None:
        console_print("  Loading selected dataset rows in one source scan...")
        selected_rows = _load_selected_rows_once(
            config.dataset_name,
            config.splits,
            config.selection_manifest,
        )
        console_print(
            "  Selected rows loaded: "
            + ", ".join(
                f"{split}={len(rows)}" for split, rows in selected_rows.items()
            )
        )

    existing_node_vocab = output_root / "vocab" / "node_vocab.json"
    existing_tactic_vocab = output_root / "vocab" / "tactic_vocab.json"
    if config.resume and existing_node_vocab.exists() and existing_tactic_vocab.exists():
        console_print("\n  Resume: loading existing train vocabularies...")
        try:
            node_vocab = json.loads(existing_node_vocab.read_text(encoding="utf-8"))
            tactic_vocab = json.loads(
                existing_tactic_vocab.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Cannot resume with unreadable vocabularies.") from exc
        for name, vocab in (
            ("node_vocab.json", node_vocab),
            ("tactic_vocab.json", tactic_vocab),
        ):
            if (
                not isinstance(vocab, dict)
                or not vocab
                or not all(
                    isinstance(label, str)
                    and isinstance(index, int)
                    and index >= 0
                    for label, index in vocab.items()
                )
                or len(set(vocab.values())) != len(vocab)
            ):
                raise RuntimeError(
                    f"Cannot resume with invalid vocabulary: {name}"
                )
        console_print(
            f"  Resume vocabularies: nodes={len(node_vocab)}, "
            f"tactics={len(tactic_vocab)}"
        )
    else:
        console_print(
            f"\n  Scanning train split from {config.dataset_name} "
            "to build train-only vocabularies..."
        )
        node_vocab, tactic_vocab, train_scan = scan_train_split(
            dataset_name=config.dataset_name,
            sample_per_split=config.sample_per_split,
            output_root=config.output_root,
            sexpr_cache=sexpr_cache,
            project_path=config.project_path,
            use_sexpr=config.use_sexpr,
            sexpr_variant=config.sexpr_variant,
            selection_manifest=config.selection_manifest,
            rows=None if selected_rows is None else selected_rows["train"],
        )
        console_print(
            f"  Train scan complete: attempted={train_scan.attempted_count}, "
            f"success={train_scan.success_count}, failure={train_scan.failure_count}"
        )

    lemma_name_index = None
    if config.lemma_corpus_path is not None:
        lemma_name_index = load_lemma_name_index(config.lemma_corpus_path)

    prepare_output_root(
        output_root,
        splits=list(config.splits),
        force=config.force,
        resume=config.resume,
        write_json_artifacts=config.write_json_artifacts,
    )
    write_vocab(output_root, name="node_vocab.json", vocab=node_vocab)
    write_vocab(output_root, name="tactic_vocab.json", vocab=tactic_vocab)

    split_reports: dict[str, SplitReport] = {}
    manifests: dict[str, dict[str, object]] = {}
    for split in config.splits:
        console_print(f"\n  Processing split '{split}'...")
        report, manifest = process_split(
            dataset_name=config.dataset_name,
            split=split,
            sample_per_split=config.sample_per_split,
            output_root=output_root,
            node_vocab=node_vocab,
            tactic_vocab=tactic_vocab,
            lemma_name_index=lemma_name_index,
            sexpr_cache=sexpr_cache,
            project_path=config.project_path,
            use_sexpr=config.use_sexpr,
            sexpr_variant=config.sexpr_variant,
            selection_manifest=config.selection_manifest,
            rows=None if selected_rows is None else selected_rows[split],
            resume=config.resume,
            write_json_artifacts=config.write_json_artifacts,
        )
        split_reports[split] = report
        manifests[split] = manifest
        console_print(
            f"  Finished '{split}': attempted={report.attempted_count}, "
            f"success={report.success_count}, failure={report.failure_count}"
        )

    summary = build_summary(
        dataset_name=config.dataset_name,
        output_root=output_root,
        splits=list(config.splits),
        manifests=manifests,
        split_reports=split_reports,
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
    )
    summary_json_path = write_summary_json(output_root, summary)
    summary_md_path = write_summary_markdown(output_root, summary)

    console_print(f"\n  Wrote node vocab     : {output_root / 'vocab' / 'node_vocab.json'}")
    console_print(f"  Wrote tactic vocab   : {output_root / 'vocab' / 'tactic_vocab.json'}")
    console_print(f"  Wrote JSON summary   : {summary_json_path}")
    console_print(f"  Wrote Markdown summary: {summary_md_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare cached graph artifacts from LeanDojo proof states")
    parser.add_argument("--dataset-name", type=str, default=DATASET_NAME, help="Dataset name to stream from Hugging Face")
    parser.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to preprocess (must include train)")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Output directory for prepared artifacts")
    parser.add_argument("--sample-per-split", type=int, default=None, help="Optional limit of examples to process per split")
    parser.add_argument(
        "--selection-manifest",
        type=str,
        default=None,
        help="Exact theorem-level pilot selection produced by build_sexpr_pilot",
    )
    parser.add_argument("--lemma-corpus", type=str, default=None, help="Optional lemma corpus JSONL for library premise labels")
    parser.add_argument("--force", action="store_true", help="Overwrite the output root if it already exists")
    parser.add_argument("--use-sexpr", action="store_true", default=False, help="Use S-expressions from Pantograph (default: False, text parser only for training)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated PyG artifacts already present under --output-root.",
    )
    parser.add_argument(
        "--no-json-artifacts",
        action="store_false",
        dest="write_json_artifacts",
        help="Skip large diagnostic JSON graphs and write training PyG artifacts only.",
    )
    parser.add_argument("--no-sexpr", action="store_false", dest="use_sexpr", help="Disable S-expressions, use text parser only")
    parser.add_argument("--project-path", type=str, default="maths_ai/lean_mathlib", help="Path to Lean project for Pantograph")
    parser.add_argument("--sexpr-cache-root", type=str, default=None, help="Validated cache root produced by generate_sexprs")
    parser.add_argument(
        "--sexpr-variant",
        choices=("raw", "model"),
        default="model",
        help=(
            "Consume normalized model sidecars (the default: their hypotheses carry "
            "the local-context indices the graph builder needs for pointer targets) "
            "or source-faithful raw S-expressions."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = PreprocessConfig(
            dataset_name=args.dataset_name,
            splits=tuple(_normalize_splits(args.splits)),
            output_root=Path(args.output_root),
            sample_per_split=args.sample_per_split,
            lemma_corpus_path=None if args.lemma_corpus is None else Path(args.lemma_corpus),
            force=args.force,
            resume=args.resume,
            write_json_artifacts=args.write_json_artifacts,
            use_sexpr=args.use_sexpr,
            sexpr_cache_root=None if args.sexpr_cache_root is None else Path(args.sexpr_cache_root),
            project_path=args.project_path,
            sexpr_variant=args.sexpr_variant,
            selection_manifest=(
                None
                if args.selection_manifest is None
                else Path(args.selection_manifest)
            ),
        )
        run_preprocessing(config)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
