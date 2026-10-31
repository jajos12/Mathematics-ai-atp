"""Exclusive accounting and examples must remain tied to the source census."""

import json
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.scripts import audit_template_coverage as coverage
from maths_ai.gnn_inference.scripts import audit_unsupported_actions as census
from maths_ai.gnn_inference.tests.test_template_coverage import (
    _atom, _cache_row, _local, _node, _prepared, _snapshot,
)


def _rw(term, *, count=1):
    rules = []
    for i in range(count):
        if i:
            rules.append(_atom(","))
        rules.append(_node("Lean.Parser.Tactic.rwRule", _node("null"), term))
    return _node("Lean.Parser.Tactic.rwSeq", _atom("rw"), _node("null"),
                 _node("Lean.Parser.Tactic.rwRuleSeq", _atom("["), _node("null", *rules), _atom("]")), _node("null"))


@pytest.fixture
def coverage_artifacts(tmp_path):
    prepared, cache, audit = (tmp_path / name for name in ("prepared", "cache", "coverage"))
    derived = _node("Lean.Parser.Tactic.exact", _atom("exact"), {
        "tag": "identifier", "source": "witness.finish", "semanticRole": "global", "name": "Receiver.finish"})
    scope = _node("Lean.Parser.Tactic.have", _atom("have"),
                  {"tag": "identifier", "source": "hx", "semanticRole": "scoped_local"},
                  {"tag": "identifier", "source": "unknown"},
                  _node("Lean.Parser.Term.byTactic", _atom("by"), _local()))
    app = _rw(_node("Lean.Parser.Term.app", _local(), _node("null", _local())))
    train = [
        _cache_row(cache, "train", 0),
        _cache_row(cache, "train", 1, derived, "exact witness.finish", legacy=True),
        _cache_row(cache, "train", 2, _node("Lean.Parser.Tactic.exact", _atom("exact"), _local(99)),
                   "exact h99", legacy=True),
        _cache_row(cache, "train", 3, scope, "have hx := by exact unknown"),
        _cache_row(cache, "train", 4, app, "rw [h0 h0]", alignment="branch_opener", target=False),
        _cache_row(cache, "train", 5, _node("Lean.Parser.Tactic.simpa", _atom("simpa"), _atom("using"), _local()),
                   "simpa using h0"),
        _cache_row(cache, "train", 6, _rw(_local(), count=4), "rw [h0, h0, h0, h0]"),
        SimpleNamespace(row_index=7, split="train", dataset_name="fixture", theorem="T7",
                        tactic_raw="simp", action_target_supported=False),
    ]
    _prepared(prepared, {"train": train, "val": [_cache_row(cache, "val", 0, app, "rw [h0 h0]"),
                                                _cache_row(cache, "val", 1)]})
    assert coverage.main(["--prepared-root", str(prepared), "--cache-root", str(cache),
                          "--output-dir", str(audit)]) == 0
    return audit, cache


def test_exclusive_census_preserves_denominator_and_overlapping_flags(tmp_path, coverage_artifacts):
    source, _ = coverage_artifacts
    output = tmp_path / "census"
    before = _snapshot(tmp_path)
    argv = ["--coverage-dir", str(source), "--output-dir", str(output)]
    assert census.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    train, val = (report["splits"][split]["summary"] for split in ("train", "val"))
    assert report["complete"] and report["processed_steps"] == 10
    assert train["selected_steps"] == 8 and train["renderable_steps"] == 1
    assert train["unsupported_or_invalid_steps"] == sum(train["primary_counts"].values()) == 7
    assert train["primary_counts"] == {
        "reference.derived_expression_or_alias": 1, "data.invalid_context_reference": 1,
        "annotation.unresolved_identifiers": 1, "term.applied_rule": 1,
        "grammar.using_clause": 1, "grammar.rule_budget": 1, "data.cache_invalid": 1,
    }
    assert train["unclassified_steps"] == 0 and train["legacy_supported_exclusions"] == 2
    assert train["needs_subclassification_steps"] == 0
    assert train["exact_aligned_unsupported_steps"] == 5
    assert val["unsupported_or_invalid_steps"] == val["renderable_steps"] == 1
    assert report["splits"]["train"]["by_family"]["rw"]["unsupported_or_invalid_steps"] == 2
    assert report["splits"]["train"]["by_primary"]["term.applied_rule"]["alignment_kinds"] == {"branch_opener": 1}
    assert train["secondary_flags"]["scoped_locals"] == train["secondary_flags"]["nested_proof"] == 1
    diagnostics = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    assert len(diagnostics) == 8
    invalid = next(row for row in diagnostics if row["split"] == "train" and row["row_index"] == 2)
    assert invalid["unknown_local_indices"] == [99]
    examples = [json.loads(line) for line in (output / "examples.jsonl").read_text().splitlines()]
    derived = next(row for row in examples if row["split"] == "train" and row["row_index"] == 1)
    assert derived["dataset_tactic"] == "exact witness.finish"
    assert any(node.get("name") == "Receiver.finish" for node in derived["syntax_preview"])
    after = _snapshot(tmp_path)
    assert all(after[path] == digest for path, digest in before.items())
    assert all(path in before or path.startswith("census/") for path in after)
    assert census.main(["--coverage-dir", str(source), "--output-dir", str(tmp_path / "repeat")]) == 0
    assert (output / "examples.jsonl").read_bytes() == (tmp_path / "repeat/examples.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        census.main(argv)


@pytest.mark.parametrize("change", ["incomplete", "rows_changed", "count_mismatch"])
def test_incomplete_or_changed_coverage_cannot_be_reclassified(tmp_path, coverage_artifacts, change):
    source, _ = coverage_artifacts
    report_path = source / "report.json"
    report = json.loads(report_path.read_text())
    if change == "incomplete":
        report["complete"] = False
    elif change == "count_mismatch":
        report["splits"]["train"]["summary"]["selected_steps"] += 1
    else:
        path = source / "rows.jsonl"
        lines = path.read_text().splitlines()
        row = json.loads(lines[0])
        row["theorem"] = "changed"
        lines[0] = json.dumps(row, ensure_ascii=False, sort_keys=True)
        path.write_text("\n".join(lines) + "\n")
    report_path.write_text(json.dumps(report))
    output = tmp_path / "census"
    with pytest.raises(ValueError, match="complete train/val|population/digest"):
        census.main(["--coverage-dir", str(source), "--output-dir", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("directory", ["sexpr", "model_sexpr_v2", "action_trace_v3"])
def test_snapshot_changes_abort_instead_of_changing_failure_population(tmp_path, coverage_artifacts, directory):
    source, cache = coverage_artifacts
    path = cache / "train" / directory / "000000001.json"
    path.write_bytes(path.read_bytes() + b" ")
    output = tmp_path / "census"
    with pytest.raises(ValueError, match="cache changed since coverage"):
        census.main(["--coverage-dir", str(source), "--output-dir", str(output)])
    assert not json.loads((output / "report.json").read_text())["complete"]


def test_unknown_rejections_remain_explicit_and_rule_outer_structure_is_preserved():
    parenthesized = _rw(_node("Lean.Parser.Term.paren", _atom("("),
                             _node("Lean.Parser.Term.app", _local(), _node("null", _local())), _atom(")")))
    facts = census.syntax_facts({"source_syntax": parenthesized, "syntax_args": [],
                                 "local_context": [{"context_index": 0, "user_name": "h0"}]})
    row = {"status": "render_unsupported", "template_error_code": "compound_rule_term"}
    assert census.primary_blocker(row, facts) == "term.parenthesized_rule"
    assert {"applied_term", "parenthesized_term"} <= facts["flags"]
    assert census.primary_blocker({**row, "template_error_code": "future_code"}, facts) == "unclassified"


def test_invalid_traces_can_still_produce_diagnostic_examples():
    trace = {"source_syntax": [], "local_context": None}
    facts = census.syntax_facts(trace, compile_trace=False)
    row = {"status": "source_trace_invalid", "error": "invalid_source_syntax", "split": "train", "row_index": 1,
           "theorem": "T", "family": "exact", "alignment_kind": "unknown", "dataset_target_matches_source": False}
    primary = census.primary_blocker(row, facts)
    assert primary == "data.invalid_source_trace"
    example = census._example(row, primary, facts, {}, trace)
    assert example["syntax_preview"] == example["local_context_preview"] == []


@pytest.mark.parametrize("kind, token, expected", [
    ("Lean.Parser.Tactic.first", "first", "structure.alternatives"),
    ("generated_macro_kind", "<;>", "structure.and_then_all"),
])
def test_tactic_control_is_not_mislabeled_as_compound_term(kind, token, expected):
    trace = {"source_syntax": _node(kind, _atom(token), _local()), "syntax_args": [],
             "local_context": [{"context_index": 0, "user_name": "h0"}]}
    facts = census.syntax_facts(trace)
    assert census.primary_blocker({"status": "render_unsupported", "render_error": "compound_or_nonreference_syntax"},
                                  facts) == expected
