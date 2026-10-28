"""Renderer contracts, native branch restoration and complete goal-set checks."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.preparation import ModelSExprCache, SExprCache, SourceSyntaxTraceCache
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import (
    FlatReference, RenderError, assess_replay, goal_signature, model_input_matches,
    render_flat_action, render_trace_action, run_native_file, select_theorems, summarize_replays,
)


def _context():
    return [{"context_index": 0, "user_name": "h"}, {"context_index": 1, "user_name": "x"}]


def test_resolved_renderer_distinguishes_list_tactics_and_term_operands():
    h, x = FlatReference("local", 0), FlatReference("local", 1)
    lemma = FlatReference("library", "Nat.add_comm")
    assert render_flat_action("rw", [h, lemma], _context()) == "rw [«h», _root_.«Nat».«add_comm»]"
    assert render_flat_action("simp", [lemma], []) == "simp [_root_.«Nat».«add_comm»]"
    assert render_flat_action("exact", [h, x], _context()) == "exact «h» «x»"
    assert render_flat_action("apply", [h], _context()) == "apply «h»"
    assert render_flat_action("rfl", [], []) == "rfl"
    assert render_flat_action("exact", [FlatReference("library", "A.«a.b»")], []) == (
        "exact _root_.«A».«a.b»"
    )


def test_renderer_rejects_ambiguous_inaccessible_and_unsupported_references():
    h = FlatReference("local", 0)
    with pytest.raises(RenderError, match="ambiguous local"):
        render_flat_action("exact", [h], [{"context_index": 0, "user_name": "h"},
                                          {"context_index": 1, "user_name": "h"}])
    with pytest.raises(RenderError, match="inaccessible"):
        render_flat_action("exact", [h], [{"context_index": 0, "user_name": "h✝"}])
    with pytest.raises(RenderError, match="absent"):
        render_flat_action("exact", [h], [])
    with pytest.raises(RenderError, match="budget"):
        render_flat_action("rw", [h] * 4, _context())
    with pytest.raises(RenderError, match="fresh_name"):
        render_flat_action("intro", [h], _context())
    with pytest.raises(RenderError, match="Lean resolver"):
        render_flat_action("exact", [FlatReference("library", "_private.A.0.foo")], [])
    with pytest.raises(RenderError, match="proof placeholder"):
        render_flat_action("admit", [], [])


def test_trace_renderer_uses_resolved_names_and_never_guesses_unannotated_leaves():
    trace = {"local_context": _context(), "source_syntax": {"tag": "node", "children": [
        {"tag": "identifier", "source": "h", "semanticRole": "local", "contextIndex": 0},
        {"tag": "identifier", "source": "foo", "semanticRole": "global", "name": "A.foo"},
    ]}}
    assert render_trace_action("rw [h, <a>foo</a>]", trace) == "rw [«h», _root_.«A».«foo»]"
    with pytest.raises(RenderError, match="unannotated"):
        render_trace_action("exact missing", trace)
    with pytest.raises(RenderError, match="syntax_modifier"):
        render_trace_action("simp only [foo]", trace)


def _goal(target="(:c P)"):
    return {"target": {"modelSexp": target, "modelSexpVersion": 1}, "vars": [
        {"contextIndex": 0, "userName": "h", "binderRole": ":explicit",
         "type": {"modelSexp": "(:c P)", "modelSexpVersion": 1}},
    ]}


def _model_record():
    return {"goal_sexp": "(:c P)", "hyp_sexps": [
        {"context_index": 0, "name": "h", "binder_role": ":explicit",
         "is_instance": False, "is_let": False, "sexp": "(:c P)"},
    ]}


def test_assessment_checks_every_sibling_and_never_claims_whole_proof_success():
    native = {"row_index": 7, "source_before": [_goal()], "source_after": [_goal(), _goal()],
              "original": {"status": "executed", "goals": [_goal(), _goal()]},
              "rendered": {"status": "executed", "goals": [_goal()]}}
    result = assess_replay(native, _model_record())
    assert result["input_model_match"] and result["original_transition_match"]
    assert result["status"] == "rendered_replay_failed"
    assert not result["rendered_transition_match"]
    assert summarize_replays([result])["proof_success_measured"] is False
    assert not model_input_matches([_goal(), _goal()], _model_record())
    native["source_before"][0]["target"]["modelSexp"] = "(:c Q)"
    assert assess_replay(native, _model_record())["status"] == "input_model_mismatch"


def test_goal_signature_keeps_local_let_values_and_requires_versioned_ast():
    a, b = _goal(), _goal()
    a["vars"][0]["value"] = {"modelSexp": "(:lit 1)", "modelSexpVersion": 1}
    b["vars"][0]["value"] = {"modelSexp": "(:lit 2)", "modelSexpVersion": 1}
    assert goal_signature([a]) != goal_signature([b])
    del b["target"]["modelSexpVersion"]
    with pytest.raises(ValueError, match="normalized expression"):
        goal_signature([b])


def test_theorem_selection_is_independent_of_input_order_and_target_support():
    rows = [DatasetRow("P", f"T{i}", "simp only [h]" if i % 2 else "exact h", "val", i,
                       file_path=f"A{i}.lean") for i in range(10)]
    first = select_theorems(rows, count=3, seed=42)
    second = select_theorems(list(reversed(rows)), count=3, seed=42)
    assert {r.theorem for r in first} == {r.theorem for r in second}
    assert len(first) == 3


def test_native_environment_resolves_both_projects_search_paths(tmp_path, monkeypatch):
    from maths_ai.gnn_inference.atp_lean_gnn import gold_replay

    source, panto = tmp_path / "mathlib checkout", tmp_path / "pantograph"

    def command(args, *, cwd, **kwargs):
        if args[:2] == ["elan", "which"]:
            assert cwd == source
            return SimpleNamespace(stdout="/lean/bin/lean\n")
        assert args == ["lake", "env", "printenv", "LEAN_PATH"]
        relative = ".lake/build/lib" if cwd == panto else ".lake/packages/mathlib/lib"
        return SimpleNamespace(stdout=relative + os.pathsep + "/shared/lean/lib\n")

    monkeypatch.setattr(gold_replay.subprocess, "run", command)
    binary, env = gold_replay.native_environment(source, panto)
    assert binary == "/lean/bin/lean"
    assert env["LEAN_PATH"].split(os.pathsep) == [
        str(panto / ".lake/build/lib"), "/shared/lean/lib", str(source / ".lake/packages/mathlib/lib"),
    ]
    assert env["LEAN_SRC_PATH"].split(os.pathsep) == [str(source), str(panto)]


def test_native_replay_restores_branches_preserves_siblings_and_source_environment():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT to run real pinned-Lean replay")
    root = Path(configured).resolve()
    fixture = (Path(__file__).parent / "fixtures/native_replay.lean").read_bytes()
    with tempfile.TemporaryDirectory(prefix="ReplaySmoke_", dir=root) as temporary:
        source_path = Path(temporary) / "Example.lean"
        source_path.write_bytes(fixture)
        from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
            PantographInvocationClient, _make_model_record,
        )

        async def capture_reference():
            client = PantographInvocationClient(
                source_root=root, pantograph_repl=root / ".lake/build/bin/repl", capture_model_sexprs=True,
            )
            try:
                return await client.process_file(str(source_path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture_reference())
        requests = []
        for i, (source, rendered) in enumerate([
            ("exact h", "exact «h»"), ("constructor", "constructor"), ("rfl", "rfl"),
            ("apply h", "apply «h»"), ("rw [h]", "rw [«h»]"),
            ("exact True.intro", "exact _root_.«True».«intro»"),
            ("exact hm", "exact «hm»"),
            ("exact Nonempty.intro default", "exact Nonempty.intro default"),
            ("exact h", "exact ReplaySmoke.twoGoals P h"),
        ]):
            token = source.encode()
            start = fixture.index(token)
            unit_index, invocation_index = next(
                (u, v) for u, unit in enumerate(units)
                for v, invocation in enumerate(unit.get("invocations", []))
                if invocation["tactic"].strip() == source
            )
            requests.append({"rowIndex": i, "sourceStart": start, "sourceEnd": start + len(token),
                             "unitIndex": unit_index, "invocationIndex": invocation_index,
                             "renderedTactic": rendered})
        response = run_native_file(source_path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not response["frontend_errors"], response
    assert len(response["rows"]) == len(requests)
    for row, request in zip(response["rows"][:-1], requests[:-1]):
        expected = units[request["unitIndex"]]["invocations"][request["invocationIndex"]]
        dataset_row = DatasetRow(expected["goalBefore"], "ReplaySmoke", expected["tactic"],
                                 "val", request["rowIndex"], target_state=expected["goalAfter"])
        cached_model = _make_model_record(dataset_row, {"invocation": expected}, {})
        assert model_input_matches(row["source_before"], cached_model)
        assert goal_signature(row["source_before"]) == goal_signature(expected["goalsBefore"])
        assert goal_signature(row["source_after"]) == goal_signature(expected["goalsAfter"])
        assert row["original"]["status"] == "executed", row
        assert row["rendered"]["status"] == "executed", row
        assert goal_signature(row["original"]["goals"]) == goal_signature(row["source_after"])
        assert goal_signature(row["rendered"]["goals"]) == goal_signature(row["source_after"])
    constructor = response["rows"][1]
    assert len(constructor["original"]["goals"]) == 2
    assert len(constructor["rendered"]["goals"]) == 2
    # A theorem defined later in this same source must not leak into the snapshot.
    assert any(v["isLet"] for v in response["rows"][6]["source_before"][0]["vars"])
    assert any(v["isInstance"] for v in response["rows"][7]["source_before"][0]["vars"])
    future = response["rows"][-1]["rendered"]
    assert future["status"] == "lean_error"
    assert "unknown" in future["message"].lower(), future


def test_replay_runner_retains_failures_and_resume_binding(tmp_path, monkeypatch):
    from maths_ai.gnn_inference.scripts import replay_gold_actions as runner

    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "A.lean").write_text("theorem T (P : Prop) (h : P) : P := by exact h")
    cache_root = tmp_path / "cache"
    raw_cache = SExprCache(cache_root, project_path=str(source_root))
    model_cache = ModelSExprCache(cache_root)
    trace_cache = SourceSyntaxTraceCache(cache_root)
    rows = [DatasetRow("h : P\n⊢ P", "T", "exact h", "val", 1,
                       target_state="no goals", file_path="A.lean", repo_commit=runner.DATASET_MATHLIB_COMMIT),
            DatasetRow("h : P\n⊢ P", "T", "simp only [h]", "val", 2,
                       target_state="no goals", file_path="A.lean", repo_commit=runner.DATASET_MATHLIB_COMMIT)]

    def syntax_node(kind, *children):
        return {"tag": "node", "kind": kind, "children": list(children)}

    def syntax_atom(value):
        return {"tag": "atom", "source": value}

    for row in rows:
        raw = {"schema_version": SExprCache.SCHEMA_VERSION,
               "extractor_version": SExprCache.EXTRACTOR_VERSION,
               "dataset": row.dataset_name, "split": row.split, "row_index": row.row_index,
               "theorem": row.theorem, "repo_commit": row.repo_commit, "file_path": row.file_path,
               "state_sha256": SExprCache.row_state_sha256(row),
               "tactic_sha256": SExprCache.row_tactic_sha256(row),
               "target_state_sha256": SExprCache.row_target_state_sha256(row),
               "unit_index": 0, "invocation_index": row.row_index}
        raw_cache.save("val", row.row_index, raw)
        model_cache.save("val", row.row_index, {
            **_model_record(), "schema_version": ModelSExprCache.SCHEMA_VERSION,
            "normalization": ModelSExprCache.NORMALIZATION,
            "raw_record_sha256": ModelSExprCache.raw_record_sha256(raw),
        })
        identifier = {"tag": "identifier", "source": "h", "semanticRole": "local", "contextIndex": 0}
        syntax = syntax_node("Lean.Parser.Tactic.exact", identifier)
        if row.row_index == 2:
            syntax = syntax_node(
                "Lean.Parser.Tactic.simp", syntax_atom("simp"), syntax_node("null"), syntax_node("null"),
                syntax_node("null", syntax_atom("only")),
                syntax_node("null", syntax_atom("["), syntax_node("null", syntax_node(
                    "Lean.Parser.Tactic.simpLemma", syntax_node("null"), syntax_node("null"), identifier,
                )), syntax_atom("]")), syntax_node("null"),
            )
        syntax.update(sourceStart=1, sourceEnd=8)
        trace_cache.save("val", row.row_index, {
            "schema_version": SourceSyntaxTraceCache.SCHEMA_VERSION,
            "extractor_version": SourceSyntaxTraceCache.EXTRACTOR_VERSION,
            "raw_record_sha256": SourceSyntaxTraceCache.raw_record_sha256(raw),
            "local_context": _context(), "syntax_args": [], "source_syntax": syntax,
        })
    calls = []

    def replay(path, requests, **kwargs):
        calls.append(requests)
        return {"frontend_errors": [], "rows": [
            {"row_index": r["rowIndex"], "source_before": [_goal()], "source_after": [],
             "original": {"status": "executed", "goals": []},
             "rendered": {"status": "executed", "goals": []} if r["renderedTactic"] else
                         {"status": "not_rendered"}}
            for r in requests
        ]}

    monkeypatch.setattr(runner, "_verify_environment", lambda *args: None)
    monkeypatch.setattr(runner, "iter_dataset_rows", lambda **kwargs: iter(rows))
    monkeypatch.setattr(runner, "run_native_file", replay)
    output = tmp_path / "output"
    argv = ["--source-root", str(source_root), "--pantograph-root", str(tmp_path),
            "--cache-root", str(cache_root), "--output-dir", str(output), "--num-theorems", "1"]
    assert runner.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["complete"]
    assert report["summary"]["selected_steps"] == 2
    assert report["summary"]["status_counts"] == {"rendered_transition_match": 1, "render_unsupported": 1}
    assert report["summary"]["original_transition_match"] == 2
    assert report["summary"]["proof_success_measured"] is False
    assert runner.main([*argv, "--resume"]) == 0
    assert len(calls) == 1
    with pytest.raises(ValueError, match="resume bindings changed"):
        runner.main([*argv, "--seed", "43", "--resume"])
    with pytest.raises(ValueError, match="resume bindings changed"):
        runner.main([*argv, "--renderer", "template-v1", "--resume"])
    template_output = tmp_path / "template_output"
    assert runner.main([*argv, "--output-dir", str(template_output), "--renderer", "template-v1"]) == 0
    templates = json.loads((template_output / "report.json").read_text())
    assert templates["manifest"]["template_schema"] == "rw_simp_template_v1"
    assert templates["summary"]["flat_renderable_steps"] == 1
    assert templates["summary"]["template_v1_parsed_steps"] == 1
    assert templates["summary"]["newly_renderable_vs_flat"] == 1
    assert templates["summary"]["rendered_transition_match"] == 2
    typed = templates["files"]["A.lean"]["rows"][1]["template_target"]
    assert typed["only"] is True and typed["rule_count"] == 1

    def fail_native(*args, **kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(runner, "run_native_file", fail_native)
    broken_output = tmp_path / "broken_native"
    assert runner.main([*argv, "--output-dir", str(broken_output)]) == 2
    broken = json.loads((broken_output / "report.json").read_text())
    assert broken["summary"]["status_counts"] == {"native_file_failed": 2}
    assert broken["summary"]["selected_steps"] == 2
    monkeypatch.setattr(runner, "run_native_file", replay)

    # A stale cache is counted, not silently dropped from the selected cohort.
    model_path = cache_root / "val/model_sexpr_v2/000000001.json"
    invalid = json.loads(model_path.read_text())
    invalid["raw_record_sha256"] = "stale"
    model_path.write_text(json.dumps(invalid))
    failed_output = tmp_path / "invalid_cache"
    assert runner.main([*argv, "--output-dir", str(failed_output)]) == 2
    failed = json.loads((failed_output / "report.json").read_text())
    assert failed["summary"]["selected_steps"] == 2
    assert failed["summary"]["status_counts"]["cache_invalid"] == 1
