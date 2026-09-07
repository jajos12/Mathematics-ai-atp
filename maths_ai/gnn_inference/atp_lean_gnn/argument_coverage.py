from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch

from .action_targets.compiler import ActionTargetError
from .action_targets.source_syntax import compile_source_syntax_trace
from .dataset import canonicalize_split_name
from .labels import get_tactic_arity, parse_tactic_arguments
from .lemma_corpus import load_lemma_corpus
from .preparation import ModelSExprCache, SExprCache, SourceSyntaxTraceCache
from .reporting import console_print
from .training import load_prepared_metadata


ALL_FRESH_IDENTIFIER_TACTICS = {
    "intro",
    "intros",
    "rintro",
    "rename_i",
}
FIRST_FRESH_IDENTIFIER_TACTICS = {"generalize", "let", "set"}
_LITERAL_RE = re.compile(
    r"^(?:[+-]?[0-9]+(?:\.[0-9]+)?|true|false|True|False|none|some)$"
)

# Operations in a version-3 target that name something instead of describing
# syntax structure.  These are the positions a pointer or retriever must fill.
STRUCTURED_REFERENCE_OPERATIONS = frozenset(
    {
        "LOCAL",
        "GLOBAL",
        "CONSTRUCTOR",
        "SCOPED_LOCAL",
        "FRESH_NAME",
        "IDENTIFIER",
        "MISSING",
    }
)
# Reference positions that a decoder can be supervised on as they stand.  A
# global reference counts only once it has been checked against a candidate pool:
# ``global_unchecked`` records that no pool was supplied, which is an absence of
# evidence rather than a resolvable target, so it is deliberately excluded here.
STRUCTURED_RESOLVED_CATEGORIES = frozenset(
    {
        "local_selectable",
        "global_library_lemma",
        "tactic_scoped_binding",
        "fresh_identifier",
    }
)
_HYGIENE_MARKER = "✝"
_ANONYMOUS_HYPOTHESIS_NAMES = {"", "_"}


@dataclass(frozen=True)
class ArgumentCoverageConfig:
    prepared_root: Path
    output_dir: Path
    splits: tuple[str, ...] = ("train", "val", "test")
    lemma_corpus: Path | None = None
    lemma_index: Path | None = None
    # Root holding the S-expression and trace sidecars.  A corpus rebuilt into a
    # new prepared root keeps reusing the sidecars extracted for the old one, so
    # the two roots must be allowed to differ.  Defaults to ``prepared_root``.
    sexpr_cache_root: Path | None = None
    structured_traces: bool = False
    max_items_per_split: int | None = None
    sample_limit_per_category: int = 20
    force: bool = False

    def normalized(self) -> "ArgumentCoverageConfig":
        splits = tuple(canonicalize_split_name(split) for split in self.splits)
        if not splits:
            raise ValueError("At least one split is required.")
        if self.max_items_per_split is not None and self.max_items_per_split < 1:
            raise ValueError("max_items_per_split must be positive when provided.")
        if self.sample_limit_per_category < 0:
            raise ValueError("sample_limit_per_category cannot be negative.")
        if self.lemma_corpus is not None and self.lemma_index is not None:
            raise ValueError("Specify either lemma_corpus or lemma_index, not both.")
        return ArgumentCoverageConfig(
            prepared_root=self.prepared_root.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            splits=splits,
            lemma_corpus=(
                self.lemma_corpus.expanduser().resolve()
                if self.lemma_corpus is not None
                else None
            ),
            lemma_index=(
                self.lemma_index.expanduser().resolve()
                if self.lemma_index is not None
                else None
            ),
            sexpr_cache_root=(
                self.sexpr_cache_root.expanduser().resolve()
                if self.sexpr_cache_root is not None
                else None
            ),
            structured_traces=self.structured_traces,
            max_items_per_split=self.max_items_per_split,
            sample_limit_per_category=self.sample_limit_per_category,
            force=self.force,
        )


def _tensor_values(data, field: str) -> list[int]:
    value = getattr(data, field, None)
    if value is None:
        return []
    if torch.is_tensor(value):
        return [int(item) for item in value.view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _parse_shape(expected_arity: int, parsed_count: int) -> str:
    if parsed_count == expected_arity:
        return "exact"
    if parsed_count < expected_arity:
        return "missing_tokens"
    if expected_arity == 0:
        return "registry_zero_but_parser_found_tokens"
    return "excess_tokens_or_compound_expression"


def _is_fresh_identifier_slot(tactic_name: str, step: int) -> bool:
    return tactic_name in ALL_FRESH_IDENTIFIER_TACTICS or (
        tactic_name in FIRST_FRESH_IDENTIFIER_TACTICS and step == 0
    )


def _selectable_flags(data) -> list[bool]:
    premise_mask = getattr(data, "premise_mask", None)
    if torch.is_tensor(premise_mask):
        return [bool(item) for item in premise_mask.view(-1).tolist()]
    if isinstance(premise_mask, (list, tuple)):
        return [bool(item) for item in premise_mask]
    return []


def _node_label_ids(data) -> list[int]:
    labels = getattr(data, "x", None)
    if torch.is_tensor(labels):
        return [int(item) for item in labels.view(-1).tolist()]
    if isinstance(labels, (list, tuple)):
        return [int(item) for item in labels]
    return []


def _leading_token(text: str) -> str:
    stripped = text.strip()
    return stripped.split(maxsplit=1)[0] if stripped else "<empty>"


def _normalized_hypothesis_name(name: object) -> str:
    """Compare hypothesis names without Lean's inaccessible-name marker."""
    return str(name or "").replace(_HYGIENE_MARKER, "").strip()


def hypothesis_names_by_context_index(record: dict) -> dict[int, str] | None:
    """Map local-context indices to hypothesis names in the audited proof state.

    Returns ``None`` when the record's hypotheses carry no context indices at
    all.  Only the normalized ``model_sexpr_v2`` sidecars record them; the raw
    source-faithful records do not, and treating their absence as "index not in
    the state" would report every local reference as a skew that does not exist.
    """
    names: dict[int, str] = {}
    for hypothesis in record.get("hyp_sexps") or ():
        if not isinstance(hypothesis, dict):
            continue
        context_index = hypothesis.get("context_index")
        if isinstance(context_index, int):
            names[context_index] = str(hypothesis.get("name", ""))
    return names or None


def _classify_local_reference(
    context_index: int,
    *,
    selectable: list[bool],
    label_ids: list[int],
    node_vocab: dict[str, int],
    hypothesis_names: dict[int, str] | None,
    trace_names: dict[int, str],
) -> tuple[str, int]:
    """Resolve one ``LOCAL`` reference to a graph node and say why it failed.

    The graph builder creates one ``FV{context_index}`` node per hypothesis, so a
    Lean-supplied context index is a direct lookup instead of a name match.
    """
    if hypothesis_names is not None and context_index not in hypothesis_names:
        return "local_absent_from_state", -1
    if hypothesis_names is not None:
        state_name = _normalized_hypothesis_name(hypothesis_names.get(context_index))
        trace_name = _normalized_hypothesis_name(trace_names.get(context_index))
        if (
            state_name not in _ANONYMOUS_HYPOTHESIS_NAMES
            and trace_name not in _ANONYMOUS_HYPOTHESIS_NAMES
            and state_name != trace_name
        ):
            # The trace and the graphed state disagree about which hypothesis
            # this index names, so any mask verdict below would describe the
            # wrong node.  Report the disagreement instead of hiding it.
            return "local_name_mismatch", -1
    label_id = node_vocab.get(f"FV{context_index}")
    if label_id is None:
        return "local_label_outside_vocab", -1
    node_ids = [index for index, value in enumerate(label_ids) if value == label_id]
    if not node_ids:
        return "local_node_absent", -1
    if len(node_ids) > 1:
        return "local_node_ambiguous", node_ids[0]
    node_id = node_ids[0]
    if node_id >= len(selectable):
        return "local_node_outside_premise_mask", node_id
    if selectable[node_id]:
        return "local_selectable", node_id
    return "local_present_but_masked", node_id


def classify_structured_argument_slots(
    data,
    trace: dict,
    *,
    split: str,
    node_vocab: dict[str, int],
    lemma_names: set[str] | None = None,
    hypothesis_names: dict[int, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Classify every naming position of one Lean-annotated tactic-syntax target.

    Positions come from Lean's own parse of the written tactic, so a compound
    argument such as ``exact f x`` is a single tree rather than several guessed
    tokens, and each identifier carries the meaning Lean resolved for it.
    """
    target = compile_source_syntax_trace(trace)
    selectable = _selectable_flags(data)
    label_ids = _node_label_ids(data)
    trace_names = {
        int(entry["context_index"]): str(entry.get("user_name", ""))
        for entry in trace.get("local_context") or ()
        if isinstance(entry, dict) and isinstance(entry.get("context_index"), int)
    }
    tactic = str(trace.get("tactic", ""))
    tactic_name = _leading_token(tactic)
    row_index = int(getattr(data, "row_index", -1))
    theorem = str(getattr(data, "theorem", ""))

    records: list[dict[str, object]] = []
    for operation in target.operations:
        operation_name = str(operation["op"])
        if operation_name not in STRUCTURED_REFERENCE_OPERATIONS:
            continue
        node_id = -1
        reference = ""
        if operation_name == "LOCAL":
            context_index = int(operation["context_index"])
            reference = f"FV{context_index}"
            category, node_id = _classify_local_reference(
                context_index,
                selectable=selectable,
                label_ids=label_ids,
                node_vocab=node_vocab,
                hypothesis_names=hypothesis_names,
                trace_names=trace_names,
            )
        elif operation_name in {"GLOBAL", "CONSTRUCTOR"}:
            reference = str(operation["name"])
            if lemma_names is None:
                category = "global_unchecked"
            elif reference in lemma_names:
                category = "global_library_lemma"
            else:
                category = "global_outside_corpus"
        elif operation_name == "SCOPED_LOCAL":
            reference = str(operation["source"])
            category = "tactic_scoped_binding"
        elif operation_name == "FRESH_NAME":
            reference = str(operation["source"])
            category = "fresh_identifier"
        elif operation_name == "IDENTIFIER":
            reference = str(operation["source"])
            category = "unannotated_identifier"
        else:
            category = "missing_syntax"

        records.append(
            {
                "split": split,
                "row_index": row_index,
                "theorem": theorem,
                "tactic_name": tactic_name,
                "tactic_raw": tactic,
                "argument_step": len(records),
                "operation": operation_name,
                "reference": reference,
                "category": category,
                "node_id": node_id,
            }
        )

    row_info = {
        "tactic_name": tactic_name,
        "operation_count": target.operation_count,
        "structural_operation_count": target.node_count + target.atom_count,
        "reference_slot_count": len(records),
        "local_reference_count": target.local_reference_count,
        "resolved": all(
            str(record["category"]) in STRUCTURED_RESOLVED_CATEGORIES
            for record in records
        ),
    }
    return records, row_info


def classify_argument_slots(
    data,
    *,
    split: str,
    lemma_names: set[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Classify each pointer-expected argument slot in one prepared example."""
    tactic_raw = str(getattr(data, "tactic_raw", ""))
    stored_tactic_name = str(getattr(data, "tactic_name", ""))
    parsed_tactic_name, tokens = parse_tactic_arguments(tactic_raw)
    tactic_name = stored_tactic_name or parsed_tactic_name
    expected_arity = get_tactic_arity(tactic_name)
    parse_shape = _parse_shape(expected_arity, len(tokens))
    node_indices = _tensor_values(data, "arg_node_indices")
    lemma_ids = _tensor_values(data, "arg_lemma_ids")
    selectable = _selectable_flags(data)

    row_index = int(getattr(data, "row_index", -1))
    theorem = str(getattr(data, "theorem", ""))
    records: list[dict[str, object]] = []
    ambiguous_parse = parse_shape in {
        "excess_tokens_or_compound_expression",
        "registry_zero_but_parser_found_tokens",
    }

    for step in range(expected_arity):
        token = tokens[step] if step < len(tokens) else None
        node_id = node_indices[step] if step < len(node_indices) else -1
        lemma_id = lemma_ids[step] if step < len(lemma_ids) else -1

        if token is None:
            category = "parser_missing_argument"
        elif node_id >= 0:
            if node_id >= len(selectable):
                category = "invalid_cached_node_index"
            elif selectable[node_id]:
                category = "local_selectable"
            else:
                category = "local_present_but_masked"
        elif lemma_id >= 0 or (lemma_names is not None and token in lemma_names):
            category = "global_library_lemma"
        elif _is_fresh_identifier_slot(tactic_name, step):
            category = "fresh_identifier"
        elif _LITERAL_RE.fullmatch(token):
            category = "literal"
        elif "." in token:
            category = "unresolved_qualified_name"
        else:
            category = "unmapped_token"

        records.append(
            {
                "split": split,
                "row_index": row_index,
                "theorem": theorem,
                "tactic_name": tactic_name,
                "tactic_raw": tactic_raw,
                "argument_step": step,
                "argument_token": token,
                "category": category,
                "node_id": node_id,
                "lemma_id": lemma_id,
                "expected_arity": expected_arity,
                "parsed_argument_count": len(tokens),
                "parse_shape": parse_shape,
                "semantically_ambiguous_parse": ambiguous_parse,
            }
        )

    row_info = {
        "tactic_name": tactic_name,
        "expected_arity": expected_arity,
        "parsed_argument_count": len(tokens),
        "ignored_parsed_token_count": max(0, len(tokens) - expected_arity),
        "parse_shape": parse_shape,
    }
    return records, row_info


def _iter_packed_graphs(prepared_root: Path, split: str) -> Iterator[object]:
    manifest_path = prepared_root / "packed" / "bidirectional" / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_payload = dict(manifest.get("splits", {})).get(split)
    if not isinstance(split_payload, dict):
        return
    chunk_names = split_payload.get("chunks", [])
    if not isinstance(chunk_names, list) or not chunk_names:
        return
    packed_root = manifest_path.parent / split
    for chunk_name in chunk_names:
        chunk_path = packed_root / str(chunk_name)
        if not chunk_path.exists():
            raise FileNotFoundError(f"Packed graph chunk does not exist: {chunk_path}")
        chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
        if not isinstance(chunk, list):
            raise ValueError(f"Packed graph chunk must contain a list: {chunk_path}")
        yield from chunk


def _iter_graphs(prepared_root: Path, split: str) -> tuple[Iterable[object], str]:
    packed_manifest = prepared_root / "packed" / "bidirectional" / "manifest.json"
    if packed_manifest.exists():
        manifest = json.loads(packed_manifest.read_text(encoding="utf-8"))
        split_payload = dict(manifest.get("splits", {})).get(split)
        if isinstance(split_payload, dict) and split_payload.get("chunks"):
            return _iter_packed_graphs(prepared_root, split), "packed"

    metadata = load_prepared_metadata(prepared_root, splits=(split,))
    pyg_dir = metadata.split_pyg_dir(split)
    files = sorted(pyg_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No prepared PyG artifacts found under {pyg_dir}")

    def individual_graphs() -> Iterator[object]:
        for path in files:
            yield torch.load(path, map_location="cpu", weights_only=False)

    return individual_graphs(), "individual"


def _new_tactic_stats() -> dict[str, object]:
    return {
        "row_count": 0,
        "expected_argument_slots": 0,
        "parsed_argument_tokens": 0,
        "ignored_parsed_tokens": 0,
        "categories": Counter(),
        "parse_shapes": Counter(),
        "arg_count_distribution": Counter(),
        "registry_disagreement_rows": 0,
    }


def _finalize_stats(stats: dict[str, object]) -> dict[str, object]:
    categories = Counter(stats["categories"])
    expected = int(stats["expected_argument_slots"])
    selectable = int(categories["local_selectable"])
    unambiguous_selectable = int(stats.get("unambiguous_selectable_slots", 0))
    return {
        "row_count": int(stats["row_count"]),
        "expected_argument_slots": expected,
        "parsed_argument_tokens": int(stats["parsed_argument_tokens"]),
        "ignored_parsed_tokens": int(stats["ignored_parsed_tokens"]),
        "categories": dict(sorted(categories.items())),
        "parse_shapes": dict(sorted(Counter(stats["parse_shapes"]).items())),
        "arg_count_distribution": {
            str(count): value
            for count, value in sorted(Counter(stats["arg_count_distribution"]).items())
        },
        "registry_disagreement_rows": int(stats["registry_disagreement_rows"]),
        "local_selectable_coverage": selectable / max(expected, 1),
        "conservative_unambiguous_local_coverage": (
            unambiguous_selectable / max(expected, 1)
        ),
    }


def _new_structured_stats() -> dict[str, object]:
    return {
        "row_count": 0,
        "rows_with_reference_slots": 0,
        "fully_resolved_row_count": 0,
        "reference_slots": 0,
        "local_reference_slots": 0,
        "operations": 0,
        "structural_operations": 0,
        "categories": Counter(),
    }


def _finalize_structured_stats(stats: dict[str, object]) -> dict[str, object]:
    categories = Counter(stats["categories"])
    reference_slots = int(stats["reference_slots"])
    local_slots = int(stats["local_reference_slots"])
    rows_with_slots = int(stats["rows_with_reference_slots"])
    resolved_rows = int(stats["fully_resolved_row_count"])
    resolved_slots = sum(int(categories[name]) for name in STRUCTURED_RESOLVED_CATEGORIES)
    return {
        "row_count": int(stats["row_count"]),
        "rows_with_reference_slots": rows_with_slots,
        "fully_resolved_row_count": resolved_rows,
        "fully_resolved_row_rate": resolved_rows / max(rows_with_slots, 1),
        "reference_slots": reference_slots,
        "local_reference_slots": local_slots,
        "operations": int(stats["operations"]),
        "structural_operations": int(stats["structural_operations"]),
        "categories": dict(sorted(categories.items())),
        "resolved_reference_coverage": resolved_slots / max(reference_slots, 1),
        "local_selectable_coverage": (
            int(categories["local_selectable"]) / max(local_slots, 1)
        ),
    }


def _trace_row_indices(prepared_root: Path, split: str) -> set[int]:
    directory = prepared_root / split / SourceSyntaxTraceCache.SIDECAR_DIR
    if not directory.is_dir():
        return set()
    rows: set[int] = set()
    for path in directory.glob("*.json"):
        try:
            rows.add(int(path.stem))
        except ValueError:
            continue
    return rows


def _render_structured_markdown(structured: dict[str, object]) -> list[str]:
    reference_slots = max(int(structured["reference_slots"]), 1)
    local_slots = int(structured["local_reference_slots"])
    lines = [
        "### Lean-annotated syntax targets",
        "",
        f"- Version-3 trace rows available: `{structured['trace_row_count']}`",
        f"- Rows classified: `{structured['row_count']}`",
        f"- Reference slots: `{structured['reference_slots']}`",
        f"- Local reference slots: `{local_slots}`",
        "- Local selectable coverage: "
        f"`{structured['local_selectable_coverage']:.4%}`",
        "- Resolved reference coverage: "
        f"`{structured['resolved_reference_coverage']:.4%}`",
        f"- Fully resolved rows: `{structured['fully_resolved_row_count']}` "
        f"(`{structured['fully_resolved_row_rate']:.4%}`)",
        f"- Target operations: `{structured['operations']}` "
        f"(structural: `{structured['structural_operations']}`)",
        "",
        "| Category | Slots | Share |",
        "| --- | ---: | ---: |",
    ]
    for category, count in dict(structured["categories"]).items():
        lines.append(f"| {category} | {count} | {count / reference_slots:.2%} |")
    unchecked = int(dict(structured["categories"]).get("global_unchecked", 0))
    if unchecked:
        lines.extend(
            [
                "",
                f"`global_unchecked` covers `{unchecked}` slots that no candidate pool",
                "was supplied for, so they are excluded from the resolved coverage above",
                "rather than assumed resolvable. Pass `--lemma-corpus` or `--lemma-index`",
                "to split them into `global_library_lemma` and `global_outside_corpus`.",
            ]
        )
    lines.extend(
        ["", "#### Trace availability", "", "| State | Rows |", "| --- | ---: |"]
    )
    for state, count in dict(structured["trace_states"]).items():
        lines.append(f"| {state} | {count} |")
    compile_failures = dict(structured["compile_failures"])
    if compile_failures:
        lines.extend(
            ["", "#### Compilation failures", "", "| Code | Rows |", "| --- | ---: |"]
        )
        for code, count in compile_failures.items():
            lines.append(f"| {code} | {count} |")
    lines.append("")
    return lines


def _render_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Argument Coverage Audit",
        "",
        f"- Prepared root: `{summary['prepared_root']}`",
        f"- Lemma corpus: `{summary.get('lemma_corpus') or 'not supplied'}`",
        f"- Lemma index: `{summary.get('lemma_index') or 'not supplied'}`",
        f"- S-expression cache root: `{summary.get('sexpr_cache_root') or 'not supplied'}`",
        "",
    ]
    if summary.get("structured_traces"):
        context_index_labels = int(summary.get("context_index_node_labels", 0))
        lines.extend(
            [
                f"- Context-index node labels in vocabulary: `{context_index_labels}`",
                "",
            ]
        )
        if context_index_labels == 0:
            lines.extend(
                [
                    "This prepared corpus contains no `FV{context_index}` nodes, so no",
                    "local reference can be pointed at whatever its mask says. The graph",
                    "builder emits those nodes only for hypotheses that record a context",
                    "index, and only the normalized `model_sexpr_v2` sidecars record one.",
                    "Rebuild the corpus with `--sexpr-variant model` before reading the",
                    "local coverage below as a property of the targets.",
                    "",
                ]
            )
        lines.extend(
            [
                "Two metrics are reported over exactly the same rows: the",
                "regex-and-arity path, whose denominator is the argument slots the",
                "static tactic-arity table expects, and the Lean-annotated syntax",
                "path, whose denominator is the naming positions in the tactic Lean",
                "actually parsed. Those denominators are different quantities, so",
                "compare the category shares and the local selectable coverage",
                "rather than the two totals.",
                "",
            ]
        )
    for split, payload in dict(summary["splits"]).items():
        lines.extend(
            [
                f"## {split}",
                "",
                f"- Rows: `{payload['row_count']}`",
                f"- Expected argument slots: `{payload['expected_argument_slots']}`",
                f"- Local selectable coverage: `{payload['local_selectable_coverage']:.4%}`",
                "- Conservative unambiguous local coverage: "
                f"`{payload['conservative_unambiguous_local_coverage']:.4%}`",
                f"- Ignored parsed tokens: `{payload['ignored_parsed_tokens']}`",
                "",
                "| Category | Slots | Share |",
                "| --- | ---: | ---: |",
            ]
        )
        expected = max(int(payload["expected_argument_slots"]), 1)
        for category, count in dict(payload["categories"]).items():
            lines.append(f"| {category} | {count} | {count / expected:.2%} |")
        lines.extend(["", "### Parser shape", "", "| Shape | Rows |", "| --- | ---: |"])
        for shape, count in dict(payload["parse_shapes"]).items():
            lines.append(f"| {shape} | {count} |")
        lines.append("")
        structured = payload.get("structured")
        if isinstance(structured, dict):
            lines.extend(_render_structured_markdown(structured))
    return "\n".join(lines) + "\n"


def run_argument_coverage_audit(config: ArgumentCoverageConfig) -> dict[str, object]:
    config = config.normalized()
    metadata = load_prepared_metadata(config.prepared_root, splits=config.splits)
    sexpr_cache_root = config.sexpr_cache_root or config.prepared_root
    if config.output_dir.exists() and any(config.output_dir.iterdir()) and not config.force:
        raise FileExistsError(
            f"Output directory is not empty: {config.output_dir}. Use --force to overwrite reports."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    lemma_names: set[str] | None = None
    if config.lemma_corpus is not None:
        lemma_names = {record.name for record in load_lemma_corpus(config.lemma_corpus)}
    elif config.lemma_index is not None:
        names_path = (
            config.lemma_index
            if config.lemma_index.is_file()
            else config.lemma_index / "lemma_names.json"
        )
        if not names_path.exists():
            raise FileNotFoundError(
                f"Lemma index names file does not exist: {names_path}"
            )
        names_payload = json.loads(names_path.read_text(encoding="utf-8"))
        if not isinstance(names_payload, list) or not all(
            isinstance(name, str) for name in names_payload
        ):
            raise ValueError(f"Lemma names file must contain a JSON string list: {names_path}")
        lemma_names = set(names_payload)

    summary: dict[str, object] = {
        "prepared_root": str(config.prepared_root),
        "lemma_corpus": str(config.lemma_corpus) if config.lemma_corpus else None,
        "lemma_index": str(config.lemma_index) if config.lemma_index else None,
        "sexpr_cache_root": str(sexpr_cache_root),
        "structured_traces": config.structured_traces,
        "splits": {},
    }
    records_path = config.output_dir / "argument_samples.jsonl"
    sample_counts: Counter[tuple[str, str]] = Counter()
    node_vocab = dict(metadata.node_vocab) if config.structured_traces else {}
    if config.structured_traces:
        # A prepared corpus can only support local pointer targets when its
        # graphs carry the ``FV{context_index}`` nodes, which the builder emits
        # only for hypotheses that record a context index.  Report the count so
        # a corpus built from the raw S-expression variant is diagnosed here
        # rather than mistaken for a masking problem downstream.
        summary["context_index_node_labels"] = sum(
            1 for label in node_vocab if label.startswith("FV")
        )
    raw_cache = (
        SExprCache(sexpr_cache_root, project_path="", enabled=True)
        if config.structured_traces
        else None
    )
    model_cache = (
        ModelSExprCache(sexpr_cache_root, enabled=True)
        if config.structured_traces
        else None
    )
    trace_cache = (
        SourceSyntaxTraceCache(sexpr_cache_root, enabled=True)
        if config.structured_traces
        else None
    )

    with records_path.open("w", encoding="utf-8") as records_handle:
        for split in config.splits:
            trace_rows: set[int] | None = None
            if config.structured_traces:
                trace_rows = _trace_row_indices(sexpr_cache_root, split)
                if not trace_rows:
                    raise FileNotFoundError(
                        "No "
                        f"{SourceSyntaxTraceCache.SIDECAR_DIR} sidecars found under "
                        f"{sexpr_cache_root / split / SourceSyntaxTraceCache.SIDECAR_DIR}"
                    )
            graphs, source = _iter_graphs(config.prepared_root, split)
            stats = _new_tactic_stats()
            per_tactic: defaultdict[str, dict[str, object]] = defaultdict(_new_tactic_stats)
            structured_stats = _new_structured_stats()
            structured_per_tactic: defaultdict[str, dict[str, object]] = defaultdict(
                _new_structured_stats
            )
            trace_states: Counter[str] = Counter()
            structured_failures: Counter[str] = Counter()
            rows_outside_population = 0
            processed = 0

            def record_sample(metric: str, category: str, record: dict) -> None:
                sample_key = (split, f"{metric}:{category}")
                if sample_counts[sample_key] >= config.sample_limit_per_category:
                    return
                records_handle.write(
                    json.dumps(
                        {"metric": metric, **record}, ensure_ascii=False, sort_keys=True
                    )
                    + "\n"
                )
                sample_counts[sample_key] += 1

            for data in graphs:
                row_index = int(getattr(data, "row_index", -1))
                if trace_rows is not None and row_index not in trace_rows:
                    # Both metrics must describe the same rows, so a row without
                    # a version-3 trace is excluded from the regex path too.
                    rows_outside_population += 1
                    continue
                if config.max_items_per_split is not None and processed >= config.max_items_per_split:
                    break
                processed += 1
                records, row = classify_argument_slots(
                    data,
                    split=split,
                    lemma_names=lemma_names,
                )
                tactic_name = str(row["tactic_name"])
                for target in (stats, per_tactic[tactic_name]):
                    target["row_count"] = int(target["row_count"]) + 1
                    target["expected_argument_slots"] = (
                        int(target["expected_argument_slots"])
                        + int(row["expected_arity"])
                    )
                    target["parsed_argument_tokens"] = (
                        int(target["parsed_argument_tokens"])
                        + int(row["parsed_argument_count"])
                    )
                    target["ignored_parsed_tokens"] = (
                        int(target["ignored_parsed_tokens"])
                        + int(row["ignored_parsed_token_count"])
                    )
                    parse_shapes = target["parse_shapes"]
                    if not isinstance(parse_shapes, Counter):
                        raise TypeError("Internal audit parse-shape counter is invalid.")
                    parse_shapes[str(row["parse_shape"])] += 1
                    arg_count_value = _tensor_values(data, "arg_count")
                    arg_count = arg_count_value[0] if arg_count_value else len(
                        _tensor_values(data, "arg_node_indices")
                    )
                    target["arg_count_distribution"][arg_count] += 1
                    if arg_count != int(row["expected_arity"]):
                        target["registry_disagreement_rows"] = int(
                            target["registry_disagreement_rows"]
                        ) + 1

                for record in records:
                    category = str(record["category"])
                    for target in (stats, per_tactic[tactic_name]):
                        categories = target["categories"]
                        if not isinstance(categories, Counter):
                            raise TypeError("Internal audit category counter is invalid.")
                        categories[category] += 1
                        if (
                            category == "local_selectable"
                            and not bool(record["semantically_ambiguous_parse"])
                        ):
                            target["unambiguous_selectable_slots"] = (
                                int(target.get("unambiguous_selectable_slots", 0)) + 1
                            )
                    record_sample("regex", category, record)

                if config.structured_traces:
                    if raw_cache is None or trace_cache is None:
                        raise TypeError("Structured audit caches were not initialised.")
                    raw_record = (
                        raw_cache.load(split, row_index) if row_index >= 0 else None
                    )
                    trace = (
                        trace_cache.load_for_raw_record(split, row_index, raw_record)
                        if raw_record is not None
                        else None
                    )
                    if raw_record is None:
                        trace_states["missing_raw_record"] += 1
                        record_sample(
                            "structured",
                            "missing_raw_record",
                            {"split": split, "row_index": row_index},
                        )
                    elif trace is None:
                        trace_states["stale_or_unreadable_trace"] += 1
                        record_sample(
                            "structured",
                            "stale_or_unreadable_trace",
                            {"split": split, "row_index": row_index},
                        )
                    else:
                        # Context indices live in the normalized sidecar; the
                        # raw record is the fallback for corpora prepared from
                        # a variant that already carries them.
                        model_record = (
                            model_cache.load_for_raw_record(
                                split, row_index, raw_record
                            )
                            if model_cache is not None
                            else None
                        )
                        hypothesis_names = hypothesis_names_by_context_index(
                            model_record if model_record is not None else raw_record
                        )
                        try:
                            structured_records, structured_row = (
                                classify_structured_argument_slots(
                                    data,
                                    trace,
                                    split=split,
                                    node_vocab=node_vocab,
                                    lemma_names=lemma_names,
                                    hypothesis_names=hypothesis_names,
                                )
                            )
                        except ActionTargetError as exc:
                            trace_states["compile_failure"] += 1
                            structured_failures[exc.code] += 1
                            record_sample(
                                "structured",
                                exc.code,
                                {
                                    "split": split,
                                    "row_index": row_index,
                                    "tactic_raw": trace.get("tactic"),
                                    "error": str(exc),
                                },
                            )
                        else:
                            trace_states["classified"] += 1
                            structured_name = str(structured_row["tactic_name"])
                            for target in (
                                structured_stats,
                                structured_per_tactic[structured_name],
                            ):
                                target["row_count"] = int(target["row_count"]) + 1
                                target["reference_slots"] = int(
                                    target["reference_slots"]
                                ) + int(structured_row["reference_slot_count"])
                                target["local_reference_slots"] = int(
                                    target["local_reference_slots"]
                                ) + int(structured_row["local_reference_count"])
                                target["operations"] = int(target["operations"]) + int(
                                    structured_row["operation_count"]
                                )
                                target["structural_operations"] = int(
                                    target["structural_operations"]
                                ) + int(structured_row["structural_operation_count"])
                                if int(structured_row["reference_slot_count"]) > 0:
                                    target["rows_with_reference_slots"] = (
                                        int(target["rows_with_reference_slots"]) + 1
                                    )
                                    if bool(structured_row["resolved"]):
                                        target["fully_resolved_row_count"] = (
                                            int(target["fully_resolved_row_count"]) + 1
                                        )
                                categories = target["categories"]
                                if not isinstance(categories, Counter):
                                    raise TypeError(
                                        "Internal structured category counter is invalid."
                                    )
                                for record in structured_records:
                                    categories[str(record["category"])] += 1
                            for record in structured_records:
                                record_sample(
                                    "structured", str(record["category"]), record
                                )

                if processed == 1 or processed % 10000 == 0:
                    console_print(f"  [{split}] audited {processed} rows ({source})")

            split_summary = _finalize_stats(stats)
            split_summary["artifact_source"] = source
            split_summary["per_tactic"] = {
                tactic: _finalize_stats(tactic_stats)
                for tactic, tactic_stats in sorted(per_tactic.items())
            }
            if config.structured_traces:
                structured_summary = _finalize_structured_stats(structured_stats)
                structured_summary["trace_row_count"] = len(trace_rows or ())
                structured_summary["rows_outside_trace_population"] = (
                    rows_outside_population
                )
                structured_summary["trace_extractor_version"] = (
                    SourceSyntaxTraceCache.EXTRACTOR_VERSION
                )
                structured_summary["trace_states"] = dict(sorted(trace_states.items()))
                structured_summary["compile_failures"] = dict(
                    sorted(structured_failures.items())
                )
                structured_summary["per_tactic"] = {
                    tactic: _finalize_structured_stats(tactic_stats)
                    for tactic, tactic_stats in sorted(structured_per_tactic.items())
                }
                split_summary["structured"] = structured_summary
            summary_splits = summary["splits"]
            if not isinstance(summary_splits, dict):
                raise TypeError("Internal audit split summary is invalid.")
            summary_splits[split] = split_summary
            console_print(
                f"  [{split}] coverage={split_summary['local_selectable_coverage']:.2%} "
                f"expected_slots={split_summary['expected_argument_slots']}"
            )
            if config.structured_traces:
                structured_payload = split_summary["structured"]
                if not isinstance(structured_payload, dict):
                    raise TypeError("Internal structured split summary is invalid.")
                console_print(
                    f"  [{split}] v3 local_selectable="
                    f"{structured_payload['local_selectable_coverage']:.2%} "
                    "resolved_references="
                    f"{structured_payload['resolved_reference_coverage']:.2%} "
                    f"reference_slots={structured_payload['reference_slots']}"
                )

    summary_path = config.output_dir / "summary.json"
    markdown_path = config.output_dir / "summary.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    console_print(f"  Wrote summary : {summary_path}")
    console_print(f"  Wrote report  : {markdown_path}")
    console_print(f"  Wrote samples : {records_path}")
    return summary
