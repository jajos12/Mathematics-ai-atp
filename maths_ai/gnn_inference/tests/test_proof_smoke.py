"""Fixed smoke cohort, bounded feedback and checkpoint/target isolation."""

import os
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.scripts import run_proof_smoke as smoke


REFERENCE_PROOFS = (
    "intro P Q R h\nexact ⟨h.1.1, h.1.2, h.2⟩",
    "intro P Q R hp hq h\nrcases h with h | h\n· exact hp h\n· exact hq h",
    "intro n\nring",
    "intro a b c h\nomega",
    "intro x y h\nomega",
    "intro x y\nring",
    "intro x\nnlinarith [sq_nonneg x]",
    "intro α xs ys\nsimp",
    "intro α xs\nsimp",
    "intro α A B C\next x\nsimp only [Set.mem_inter_iff, Set.mem_union]\naesop",
)


def _report():
    return {"tasks": [{"id": task[0], "attempts": []} for task in smoke.TASKS]}


def test_fixed_ten_tasks_have_no_runtime_reference_proofs():
    assert len(smoke.TASKS) == len(REFERENCE_PROOFS) == 10
    assert len({task[0] for task in smoke.TASKS}) == 10
    assert {task[1] for task in smoke.TASKS} == {"logic", "nat", "int", "real", "list", "set"}
    assert all(len(task) == 3 for task in smoke.TASKS)


def test_equivalent_binders_supported_without_changing_frozen_specification():
    task = smoke.TASKS[0]
    raw = "theorem result (P Q R : Prop) (h : (P ∧ Q) ∧ R) : P ∧ (Q ∧ R) := by aesop"
    source, name = smoke._candidate_source(task, raw)
    assert source.index(f"def spec_{task[0]} : Prop := {task[2]}") < source.index("theorem result")
    assert "exact Candidate_and_assoc.result" in source
    assert f"#print axioms {name}" in source
    with pytest.raises(ValueError, match="Unterminated"):
        smoke._candidate_source(task, "```lean4\ntheorem unfinished")


def test_bounded_repair_no_cross_task_history_and_no_duplicate_completed_resume():
    report = _report()
    calls = []

    def generate(messages, seed):
        calls.append((messages, seed))
        return {"raw_text": "norm_num", "seed": seed, "done_reason": "stop"}

    seen = {}

    def validate(task, raw):
        if raw == smoke.AUTOMATION:
            return {"status": "passed"}
        count = seen.get(task[0], 0)
        seen[task[0]] = count + 1
        return {"status": "lean_rejected", "diagnostics": "actual checker error"} if count == 0 else {"status": "passed"}

    summary = smoke._run(report, generate, validate, lambda: None, 2, 42)
    assert summary["selected_tasks"] == summary["completed_tasks"] == 10
    assert summary["first_try_solved"] == 0
    assert summary["newly_solved_by_repair"] == 10
    assert summary["model_calls"] == summary["generation_requests_started"] == 20
    assert summary["model_only_solved"] == 0  # Automation also solved every fake case.
    for index in range(10):
        initial, repair = calls[index * 2:index * 2 + 2]
        assert len(initial[0]) == 1
        assert len(repair[0]) == 3 and "actual checker error" in repair[0][-1]["content"]
        assert initial[1] == 42 + index * 10 and repair[1] == initial[1] + 1
    smoke._run(report, lambda *a: pytest.fail("duplicate inference"), validate, lambda: None, 2, 42)


def test_resume_saved_generation_validates_it_without_resampling():
    report = _report()
    for row in report["tasks"][1:]:
        row["complete"] = True
    report["tasks"][0]["automation"] = {"status": "passed"}
    report["tasks"][0]["attempts"] = [{"raw_text": "aesop", "seed": 42}]
    summary = smoke._run(report, lambda *a: pytest.fail("resampled stored output"),
                          lambda *a: {"status": "passed"}, lambda: None, 2, 42)
    assert summary["solved_with_repairs"] == 1


def test_failed_proofs_remain_in_ten_task_denominator():
    report = _report()
    summary = smoke._run(report, lambda *a: {"raw_text": "sorry"},
                          lambda *a: {"status": "lean_rejected"}, lambda: None, 1, 0)
    assert summary["selected_tasks"] == summary["completed_tasks"] == 10
    assert summary["solved_with_repairs"] == 0 and summary["model_calls"] == 10


def test_frozen_target_uses_same_kernel_checker(monkeypatch):
    captured = []

    def check(source, name, context, timeout):
        captured.append(source)
        return {"status": "lean_rejected"}

    monkeypatch.setattr(smoke.control, "_check_source", check)
    task = smoke.TASKS[0]
    result = smoke._validate(task, "theorem cheat (h : False) : True := by exact h", None, 1)
    assert result["status"] == "lean_rejected"
    assert task[2] in captured[0] and "exact Candidate_and_assoc.cheat" in captured[0]


def test_context_overflow_is_not_silently_truncated():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            return list(range(20)) if tokenize else "text"

        def encode(self, *a, **k):
            return list(range(20))

        def decode(self, *a, **k):
            return "text"

    with pytest.raises(ValueError, match="exceeds context"):
        smoke._render(Tokenizer(), [], SimpleNamespace(max_new_tokens=20, context_tokens=30))


def test_repair_history_compaction_never_changes_original_task():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            text = "\n".join(m["content"] for m in messages)
            return list(text.encode()) if tokenize else text

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, ids, **kwargs):
            return bytes(ids).decode()

    messages = [{"role": "user", "content": "ORIGINAL FIXED TARGET"},
                {"role": "assistant", "content": "x" * 4000},
                {"role": "user", "content": "y" * 1800}]
    actual, _, ids = smoke._fit_prompt(Tokenizer(), messages,
                                       SimpleNamespace(max_new_tokens=200, context_tokens=1500))
    assert actual[0] == messages[0]
    assert "history elided" in actual[1]["content"]
    assert len(ids) + 200 <= 1500


def test_runner_loads_cached_model_once_and_binds_resume(tmp_path, monkeypatch):
    project = tmp_path / "lean"
    project.mkdir()
    (project / "lake-manifest.json").write_text("{}")
    opened = []
    calls = []

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["local_files_only"] is True
            return cls()

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            text = json.dumps(messages, ensure_ascii=False)
            return list(text.encode()) if tokenize else text

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, ids, **kwargs):
            return bytes(ids).decode()

    class Model:
        def __init__(self, args, tokenizer):
            assert args.local_files_only is True

        def __enter__(self):
            opened.append(1)
            return self

        def __exit__(self, *args):
            pass

        def generate(self, ids, seed):
            calls.append(seed)
            return {"seed": seed, "raw_text": "aesop", "done_reason": "stop"}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(PreTrainedTokenizerFast=Tokenizer))
    monkeypatch.setattr(smoke, "_cached_weights", lambda _: {"shards": 2})
    monkeypatch.setattr(smoke.control, "_preflight", lambda _: {"runtime": {"test": "fixed"}})
    monkeypatch.setattr(smoke.control, "_lean_context", lambda _: (project, "test-lean", ""))
    monkeypatch.setattr(smoke.control, "_check_source", lambda *a: {"status": "passed"})
    monkeypatch.setattr(smoke.control, "ReferenceGenerator", Model)
    output = tmp_path / "run"
    argv = ["--output-dir", str(output), "--lean-project", str(project)]
    assert smoke.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    assert len(opened) == 1 and len(calls) == 10
    assert report["summary"]["first_try_solved"] == 10
    assert smoke.main([*argv, "--resume"]) == 0
    assert len(opened) == 1 and len(calls) == 10
    with pytest.raises(ValueError, match="Resume changes"):
        smoke.main([*argv, "--max-attempts", "1", "--resume"])


def test_automation_recheck_preserves_source_and_model_outputs_without_loading_model(tmp_path, monkeypatch):
    project = tmp_path / "lean"
    project.mkdir()
    (project / "lake-manifest.json").write_text("{}")
    report = _report()
    report["manifest"] = {"schema": "ten_proof_smoke_v1", "tasks": smoke.TASKS,
                          "toolchain": "test", "lean_timeout": 120,
                          "lake_manifest_sha256": smoke.control._sha("{}")}
    for row in report["tasks"]:
        row.update(complete=True, automation={"status": "passed"},
                   attempts=[{"raw_text": "saved model proof", "lean": {"status": "passed"}}])
    report["tasks"][3]["attempts"][0]["lean"]["status"] = "lean_rejected"
    report["tasks"][6]["automation"]["status"] = "lean_rejected"
    source = tmp_path / "original.json"
    source.write_text(json.dumps(report))
    before = source.read_text()
    monkeypatch.setattr(smoke, "_cached_weights", lambda *a: pytest.fail("model cache accessed"))
    monkeypatch.setattr(smoke.control, "_lean_context", lambda *a: (project, "test", ""))
    calls = []

    def validate(task, raw, context, timeout):
        calls.append(task[0])
        assert "\nsolve\n" in raw
        return {"status": "passed"}

    monkeypatch.setattr(smoke, "_validate", validate)
    output = tmp_path / "rechecked"
    argv = ["--output-dir", str(output), "--lean-project", str(project),
            "--recheck-automation", str(source)]
    assert smoke.main(argv) == 0
    saved = json.loads((output / "report.json").read_text())
    assert source.read_text() == before
    assert [r["attempts"] for r in saved["tasks"]] == [r["attempts"] for r in report["tasks"]]
    assert saved["summary"]["automation_solved"] == 10
    assert saved["summary"]["solved_with_repairs"] == 9
    assert saved["summary"]["model_only_solved"] == 0
    assert smoke.main([*argv, "--resume"]) == 0 and len(calls) == 10
    with pytest.raises(ValueError, match="original validator time budget"):
        smoke.main([*argv, "--lean-timeout", "1", "--resume"])


@pytest.mark.skipif(not os.environ.get("MATHS_AI_PROOF_SMOKE_LEAN_PROJECT"), reason="compiled Lean project required")
def test_all_reference_tasks_and_extra_assumption_rejection_in_real_lean():
    context = smoke.control._lean_context(Path(os.environ["MATHS_AI_PROOF_SMOKE_LEAN_PROJECT"]))
    sources, names = [], []
    prefix = "import Mathlib\nimport Aesop\n"
    for task, proof in zip(smoke.TASKS, REFERENCE_PROOFS):
        source, name = smoke._candidate_source(task, proof)
        sources.append(source.removeprefix(prefix))
        names.append(name)
    result = smoke.control._check_source(prefix + "\n".join(sources), names[-1], context, 120)
    assert result["status"] == "passed", result["diagnostics"]
    assert all(f"'{name}'" in result["diagnostics"] for name in names)
    task = smoke.TASKS[0]
    rejected = smoke._validate(task, f"theorem fake (h : False) : {task[2]} := by exact False.elim h", context, 120)
    assert rejected["status"] == "lean_rejected"
    valid = smoke._validate(task, "theorem equivalent (P Q R : Prop) (h : (P ∧ Q) ∧ R) : "
                           "P ∧ (Q ∧ R) := by exact ⟨h.1.1, h.1.2, h.2⟩", context, 120)
    assert valid["status"] == "passed", valid["diagnostics"]
    # A normalizer may succeed without closing the goal; solve must backtrack.
    control_task = ("portfolio_control", "logic", "True")
    premature = smoke._validate(control_task, "first\n| skip\n| trivial", context, 120)
    assert premature["status"] == "lean_rejected"
    closed = smoke._validate(control_task, "solve\n| skip\n| trivial", context, 120)
    assert closed["status"] == "passed", closed["diagnostics"]
    nonneg = smoke._validate(smoke.TASKS[6], smoke.AUTOMATION, context, 120)
    assert nonneg["status"] == "passed", nonneg["diagnostics"]
