"""Explain unsupported coverage rows with exclusive blockers and syntax evidence.

Consumes a complete template_coverage_v1 report; does not reload prepared
graphs, change support policy, execute Lean or export training targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from maths_ai.gnn_inference.atp_lean_gnn.action_targets import compile_source_syntax_trace
from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256


POLICY = "unsupported_census_v1"
PRECEDENCE = ["data", "missing_syntax", "unresolved_identifiers", "new_bindings",
              "root_sequence", "nested_proof", "tactic_control", "reference_resolution",
              "template_rejection", "flat_rejection", "unclassified"]
WRAPPERS = {"null", "Lean.Parser.Tactic.tacticSeq", "Lean.Parser.Tactic.tacticSeq1Indented"}
TEMPLATE_BLOCKERS = {
    "rule_budget": "grammar.rule_budget", "simp_rule_modifier": "grammar.simp_rule_modifier",
    "combined_location": "grammar.combined_location", "location_budget": "grammar.location_budget",
    "configuration": "grammar.configuration_or_discharger",
    "family_outside_v1": "grammar.unsupported_tactic_root", "family_mismatch": "data.family_mismatch",
    "unsupported_syntax_shape": "grammar.unsupported_syntax_shape",
    "multi_tactic_wrapper": "structure.non_single_tactic_wrapper",
    "invalid_rw": "grammar.empty_rw_rules", "unsupported_marker": "grammar.unsupported_marker",
    "unexpected_marker": "grammar.unsupported_marker", "unexpected_token": "grammar.unexpected_token",
}
WORK_AREAS = {"data": "inspect extraction/cache integrity", "annotation": "inspect identifier annotation policy",
              "scope": "typed binder/scope support", "structure": "structured tactic-control support",
              "term": "typed term-expression support", "reference": "native reference resolution",
              "grammar": "bounded tactic grammar", "unclassified": "inspect examples and refine classification"}
COARSE_BLOCKERS = {"unclassified", "term.other_compound_rule", "term.other_nonreference_syntax",
                   "reference.other_resolution", "grammar.other_modifier_or_location",
                   "grammar.unsupported_syntax_shape", "grammar.unsupported_tactic_root"}
# Named parsers and keyword tokens from pinned Lean 4.10; unnamed macros such
# as `<;>` have generated constructor names, so inspect their literal atoms.
CONTROL_FORMS = (
    ("and_then_all", {"<;>"}, set()),
    ("sequence", {";"}, {"nestedTactic", "paren"}),
    ("alternatives", {"first", "solve"}, {"first"}),
    ("try", {"try"}, set()),
    ("repeat", {"repeat", "repeat'", "repeat1'"}, {"repeat'", "repeat1'"}),
    ("goal_dispatch", {"all_goals", "any_goals"}, {"allGoals", "anyGoals"}),
    ("case_block", {"case", "next"}, {"case", "case'", "match", "introMatch"}),
    ("focus", {"focus"}, {"focus"}),
)


def _rows(path):
    with path.open("rb") as handle:
        for line in handle:
            yield json.loads(line), line


def verify_coverage(report, path):
    if (report.get("complete") is not True or report["manifest"]["schema"] != "template_coverage_v1"
            or report["manifest"]["splits"] != ["train", "val"]):
        raise ValueError("complete train/val template_coverage_v1 report required")
    seen, digests, statuses = defaultdict(set), defaultdict(hashlib.sha256), defaultdict(Counter)
    for row, line in _rows(path):
        split, index = row["split"], row["row_index"]
        if split not in {"train", "val"} or type(index) is not int or index < 0 or index in seen[split]:
            raise ValueError("duplicate or invalid coverage row")
        if row["status"] not in {"renderable", "render_unsupported", "source_trace_invalid", "cache_invalid"}:
            raise ValueError("unknown coverage status")
        if row["template_mode_renderable"] is not (row["status"] == "renderable"):
            raise ValueError("coverage support/status mismatch")
        seen[split].add(index)
        digests[split].update(line)
        statuses[split][row["status"]] += 1
    for split in ("train", "val"):
        source = report["splits"][split]
        if (source.get("complete") is not True
                or len(seen[split]) != source["expected_steps"]
                or len(seen[split]) != source["summary"]["selected_steps"]
                or dict(statuses[split]) != source["summary"]["status_counts"]
                or digests[split].hexdigest() != source["row_records_sha256"]):
            raise ValueError(f"coverage population/digest mismatch in {split}")


def read_snapshot(row, cache):
    """Check every cache hash recorded for this unsupported row before inspection."""
    payloads = {}
    for directory, expected in row["cache_sha256"].items():
        if directory not in {"sexpr", "model_sexpr_v2", "action_trace_v3"}:
            raise ValueError("unknown cache directory in coverage record")
        path = cache / row["split"] / directory / f"{row['row_index']:09d}.json"
        if expected is None:
            if path.exists():
                raise ValueError(f"cache changed since coverage audit: {path}")
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"cache changed since coverage audit: {path}")
        if directory == "model_sexpr_v2":
            continue
        try:
            value = json.loads(content)
            payloads[directory] = value if isinstance(value, dict) else None
        except (ValueError, UnicodeError, RecursionError):
            if row["status"] != "cache_invalid":
                raise ValueError(f"unreadable previously valid cache: {path}") from None
    if row["status"] != "cache_invalid" and not all(payloads.get(key) for key in ("sexpr", "action_trace_v3")):
        raise ValueError("coverage row lacks source/trace snapshot bindings")
    return payloads.get("sexpr") or {}, payloads.get("action_trace_v3") or {}


def syntax_facts(trace, *, compile_trace=True):
    flags, kinds, atoms, rule_kinds = set(), Counter(), set(), []
    root = trace.get("source_syntax", {})
    if not isinstance(root, dict):
        root = {}
    while root.get("tag") == "node" and root.get("kind") in WRAPPERS:
        children = root.get("children", [])
        if not isinstance(children, list) or len(children) != 1:
            flags.add("non_single_root_wrapper")
            break
        root = children[0] if isinstance(children[0], dict) else {}
    local_context = trace.get("local_context", [])
    context = {item["context_index"] for item in local_context
               if isinstance(item, dict) and type(item.get("context_index")) is int} if isinstance(local_context, list) else set()
    unknown_locals = set()
    pending = [trace.get("source_syntax")]
    while pending:
        node = pending.pop()
        if not isinstance(node, dict):
            continue
        children = node.get("children", [])
        if isinstance(children, list):
            pending.extend(reversed(children))
        kind = node.get("kind", "")
        if not isinstance(kind, str):
            kind = "invalid_kind"
        if node.get("tag") == "node":
            kinds[kind] += 1
            if kind in {"Lean.Parser.Tactic.rwRule", "Lean.Parser.Tactic.simpLemma"} and children:
                term = children[-1]
                if isinstance(term, dict) and term.get("tag") != "identifier":
                    rule_kinds.append(term.get("kind", term.get("tag", "unknown")))
        if node.get("tag") == "atom" and isinstance(node.get("source"), str):
            atoms.add(node["source"].strip())
        index = node.get("contextIndex")
        if node.get("semanticRole") == "local" and type(index) is int and index not in context:
            unknown_locals.add(index)
    for kind, flag in {"Lean.Parser.Term.app": "applied_term", "Lean.Parser.Term.paren": "parenthesized_term",
                       "Lean.Parser.Term.byTactic": "nested_proof", "Lean.Parser.Term.num": "literal_term",
                       "Lean.Parser.Term.str": "literal_term", "Lean.Parser.Term.hole": "placeholder"}.items():
        if kind in kinds:
            flags.add(flag)
    controls = [name for name, tokens, constructors in CONTROL_FORMS
                if atoms & tokens or any(f"Lean.Parser.Tactic.{kind}" in kinds for kind in constructors)]
    flags.update(f"tactic_control:{name}" for name in controls)
    counts = {}
    if compile_trace:
        compiled = compile_source_syntax_trace(trace)
        for field, flag in (("missing_count", "missing_syntax"),
                            ("unannotated_identifier_count", "unresolved_identifiers"),
                            ("fresh_name_count", "fresh_names"), ("scoped_local_count", "scoped_locals")):
            counts[field] = getattr(compiled, field)
            if counts[field]:
                flags.add(flag)
    return {"flags": flags, "syntax_kinds": kinds, "atoms": atoms, "rule_term_kinds": rule_kinds,
            "root_kind": root.get("kind", root.get("tag", "unknown")),
            "unknown_local_indices": sorted(unknown_locals), "compiler_counts": counts,
            "control_kind": controls[0] if controls else None}


def primary_blocker(row, facts):
    """One observed blocker under declared precedence, never a sole-cause claim."""
    if row["status"] == "cache_invalid":
        return "data.cache_invalid"
    if row["status"] == "source_trace_invalid":
        return "data.invalid_context_reference" if row.get("error") == "unknown_local_reference" else "data.invalid_source_trace"
    flags = facts["flags"]
    if "missing_syntax" in flags:
        return "annotation.missing_syntax"
    if "unresolved_identifiers" in flags:
        return "annotation.unresolved_identifiers"
    if flags & {"fresh_names", "scoped_locals"}:
        return "scope.new_bindings"
    if "non_single_root_wrapper" in flags:
        return "structure.non_single_tactic_wrapper"
    if "nested_proof" in flags:
        return "term.nested_proof"
    if facts.get("control_kind"):
        return "structure." + facts["control_kind"]
    error = row.get("render_error") or ""
    for text, category in (
        ("derived expression or renamed alias", "reference.derived_expression_or_alias"),
        ("local spelling requires", "reference.local_expression_or_alias"),
        ("possible local method receiver", "reference.local_receiver"),
        ("private or numeric declaration", "reference.private_or_numeric_name"),
        ("inaccessible local name", "reference.inaccessible_local_name"),
        ("ambiguous local name", "reference.ambiguous_local_name"),
        ("ambiguous or unannotated source reference", "reference.annotation_lookup"),
    ):
        if text in error:
            return category
    code = row.get("template_error_code")
    if code == "compound_rule_term":
        outer = facts["rule_term_kinds"][0] if facts["rule_term_kinds"] else ""
        return {"Lean.Parser.Term.app": "term.applied_rule", "Lean.Parser.Term.paren": "term.parenthesized_rule"}.get(
            outer, "term.other_compound_rule")
    if code in TEMPLATE_BLOCKERS:
        return TEMPLATE_BLOCKERS[code]
    if code == "reference_resolution":
        return "reference.other_resolution"
    if code:  # Unexpected parser codes must remain visible, not enter a catch-all grammar bucket.
        return "unclassified"
    if error == "fresh_name_or_binding_syntax":
        return "scope.fresh_binding_syntax"
    if error == "syntax_modifier_or_location":
        for atom, category in (("at", "location"), ("only", "only_modifier"),
                               ("using", "using_clause"), ("with", "with_clause")):
            if atom in facts["atoms"]:
                return f"grammar.{category}"
        return "grammar.other_modifier_or_location"
    if error == "compound_or_nonreference_syntax":
        for flag, category in (("applied_term", "applied_expression"), ("parenthesized_term", "parenthesized_expression"),
                               ("literal_term", "literal_expression"), ("placeholder", "placeholder")):
            if flag in flags:
                return f"term.{category}"
        return "term.other_nonreference_syntax"
    if error == "over_decoder_budget":
        return "grammar.flat_reference_budget"
    if error == "unsupported tactic name":
        return "grammar.unsupported_tactic_name"
    return "unclassified"


def _new_stats():
    return {key: Counter() for key in ("primary_counts", "by_family", "secondary_flags", "alignment_kinds",
                                      "syntax_kinds", "source_status_counts", "counts")}


def _add(stats, row, primary, facts):
    stats["counts"]["selected_steps"] += 1
    stats["source_status_counts"][row["status"]] += 1
    if row["status"] == "renderable":
        stats["counts"]["renderable_steps"] += 1
        return
    stats["counts"].update({"unsupported_or_invalid_steps": 1, "unclassified_steps": primary == "unclassified",
                            "needs_subclassification_steps": primary in COARSE_BLOCKERS,
                            "legacy_supported_exclusions": bool(row["legacy_label_supported"]),
                            "flat_only_steps": bool(row["flat_renderable"]),
                            "exact_aligned_unsupported_steps": row["alignment_kind"] == "exact"
                            and row["dataset_target_matches_source"] is True})
    stats["primary_counts"][primary] += 1
    stats["by_family"][row["family"]] += 1
    stats["secondary_flags"].update(facts["flags"])
    stats["alignment_kinds"][row["alignment_kind"]] += 1
    stats["syntax_kinds"].update(facts["syntax_kinds"].keys())  # Rows containing each kind, not occurrence counts.


def _summary(stats):
    counts = {key: stats["counts"][key] for key in (
        "selected_steps", "renderable_steps", "unsupported_or_invalid_steps", "unclassified_steps",
        "needs_subclassification_steps",
        "legacy_supported_exclusions", "flat_only_steps", "exact_aligned_unsupported_steps")}
    if (sum(stats["primary_counts"].values()) != counts["unsupported_or_invalid_steps"]
            or counts["selected_steps"] != counts["renderable_steps"] + counts["unsupported_or_invalid_steps"]):
        raise ValueError("census primary categories do not partition the population")
    return {**counts, **{key: dict(value) for key, value in stats.items() if key != "counts"}}


def _example(row, primary, facts, raw, trace):
    preview, pending = [], [("root", trace.get("source_syntax"))]
    while pending and len(preview) < 80:
        path, node = pending.pop()
        if not isinstance(node, dict):
            continue
        entry = {key: value[:160] if isinstance(value, str) else value for key, value in node.items()
                 if key in {"tag", "kind", "source", "semanticRole", "name", "contextIndex"}}
        preview.append({"path": path, **entry})
        children = node.get("children", [])
        if isinstance(children, list):
            pending.extend((f"{path}.{i}", child) for i, child in reversed(list(enumerate(children))))
    tactic, state = str(raw.get("tactic", "")), str(raw.get("text_state", ""))
    return {"split": row["split"], "row_index": row["row_index"], "theorem": row["theorem"],
            "family": row["family"], "primary_blocker": primary, "secondary_flags": sorted(facts["flags"]),
            "alignment_kind": row["alignment_kind"], "dataset_target_matches_source": row["dataset_target_matches_source"],
            "dataset_tactic": tactic[:4000], "tactic_truncated": len(tactic) > 4000,
            "state_preview": state[:2000], "state_preview_truncated": len(state) > 2000,
            "file_path": raw.get("file_path"),
            "render_error": row.get("render_error"), "template_error_code": row.get("template_error_code"),
            "flat_render_error": row.get("flat_render_error"), "source_error": row.get("error"),
            "compiler_counts": facts["compiler_counts"], "unknown_local_indices": facts["unknown_local_indices"],
            "local_context_preview": trace["local_context"][:32] if isinstance(trace.get("local_context"), list) else [],
            "local_context_preview_truncated": isinstance(trace.get("local_context"), list)
            and len(trace["local_context"]) > 32,
            "syntax_preview": preview, "syntax_preview_truncated": bool(pending)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-family-blocker", type=int, default=3)
    args = parser.parse_args(argv)
    if args.samples_per_family_blocker < 1:
        raise ValueError("sample count must be positive")
    coverage, output = args.coverage_dir.resolve(), args.output_dir.resolve()
    source_path, rows_path = coverage / "report.json", coverage / "rows.jsonl"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_hash, rows_hash = file_sha256(source_path), file_sha256(rows_path)
    verify_coverage(source, rows_path)
    cache = Path(source["manifest"]["cache_root"]).resolve()
    for root in (coverage, cache, Path(source["manifest"]["prepared_root"]).resolve()):
        if output == root or root in output.parents:
            raise ValueError("output must be outside input artifact roots")
    code_root = Path(__file__).parents[1]
    code = [Path(__file__), *sorted((code_root / "atp_lean_gnn/action_targets").glob("*.py"))]
    report = {"manifest": {"policy": POLICY, "precedence": PRECEDENCE,
                           "coverage_report_sha256": source_hash, "coverage_rows_sha256": rows_hash,
                           "coverage_dir": str(coverage), "cache_root": str(cache),
                           "samples_per_family_blocker": args.samples_per_family_blocker,
                           "sample_policy": "lowest_sha256_split_row_index_v1",
                           "code_sha256": {str(path.relative_to(code_root)): file_sha256(path) for path in code},
                           "native_execution_measured": False, "proof_success_measured": False},
              "complete": False, "splits": {}}
    output.mkdir(parents=True, exist_ok=False)

    def checkpoint():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(output / "report.json")

    checkpoint()
    totals, families, groups, samples = defaultdict(_new_stats), defaultdict(_new_stats), defaultdict(_new_stats), {}
    with (output / "diagnostics.jsonl").open("w", encoding="utf-8") as handle:
        for processed, (row, _) in enumerate(_rows(rows_path), 1):
            split, primary = row["split"], None
            facts = {"flags": set(), "syntax_kinds": Counter()}
            if row["status"] != "renderable":
                raw, trace = read_snapshot(row, cache)
                facts = syntax_facts(trace, compile_trace=row["status"] == "render_unsupported")
                if row.get("template_error_code"):
                    facts["flags"].add("template_error:" + row["template_error_code"])
                if row.get("flat_render_error"):
                    facts["flags"].add("flat_error:" + row["flat_render_error"].split(":", 1)[0])
                primary = primary_blocker(row, facts)
                handle.write(json.dumps({"split": split, "row_index": row["row_index"], "family": row["family"],
                                         "primary_blocker": primary, "secondary_flags": sorted(facts["flags"]),
                                         "root_kind": facts["root_kind"], "alignment_kind": row["alignment_kind"],
                                         "dataset_target_matches_source": row["dataset_target_matches_source"],
                                         "unknown_local_indices": facts["unknown_local_indices"]},
                                        ensure_ascii=False, sort_keys=True) + "\n")
                _add(groups[(split, primary)], row, primary, facts)
                key = (split, primary, row["family"])
                rank = hashlib.sha256(f"{split}:{row['row_index']}".encode()).hexdigest()
                chosen = samples.setdefault(key, [])
                if len(chosen) < args.samples_per_family_blocker or rank < chosen[-1][0]:
                    chosen.append((rank, _example(row, primary, facts, raw, trace)))
                    chosen.sort(key=lambda item: item[0])
                    del chosen[args.samples_per_family_blocker:]
            _add(totals[split], row, primary, facts)
            _add(families[(split, row["family"])], row, primary, facts)
            if processed % 10000 == 0:
                report["processed_steps"] = processed
                report["splits"] = {name: {"summary": _summary(stats)} for name, stats in totals.items()}
                checkpoint()
                print(f"classified {processed} coverage rows", flush=True)
    if file_sha256(source_path) != source_hash or file_sha256(rows_path) != rows_hash:
        raise ValueError("coverage artifacts changed during census")
    with (output / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for key in sorted(samples):
            for _, example in samples[key]:
                handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
    for split in ("train", "val"):
        summary = _summary(totals[split])
        if summary["selected_steps"] != source["splits"][split]["summary"]["selected_steps"]:
            raise ValueError("census population changed")
        report["splits"][split] = {"summary": summary,
            "by_family": {name: _summary(stats) for (s, name), stats in families.items() if s == split},
            "by_primary": {name: {"work_area": WORK_AREAS[name.split(".")[0]], **_summary(stats)}
                           for (s, name), stats in groups.items() if s == split}}
    report["train_ranked_blockers"] = [
        {"primary_blocker": name, "rows": count, "work_area": WORK_AREAS[name.split(".")[0]],
         "exact_aligned_rows": groups[("train", name)]["counts"]["exact_aligned_unsupported_steps"],
         "needs_subclassification": name in COARSE_BLOCKERS}
        for name, count in totals["train"]["primary_counts"].most_common()]
    report.update(complete=True, processed_steps=sum(stats["counts"]["selected_steps"] for stats in totals.values()),
                  examples_file="examples.jsonl", diagnostics_file="diagnostics.jsonl")
    checkpoint()
    print(json.dumps({split: {key: value for key, value in entry["summary"].items()
                             if key in {"selected_steps", "renderable_steps", "unsupported_or_invalid_steps",
                                        "unclassified_steps", "needs_subclassification_steps", "primary_counts"}}
                      for split, entry in report["splits"].items()}, indent=2), flush=True)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
