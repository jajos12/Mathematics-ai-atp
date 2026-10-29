"""Source-faithful development selection and baseline-first execution."""

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.scripts import run_minif2f_dev as dev
from maths_ai.gnn_inference.scripts import run_proof_smoke as smoke


def _source(name):
    return ("import Mathlib\nset_option maxHeartbeats 0\nopen Real Nat\n"
            f"theorem {name} (n : Nat) (h : n = 7) : n + 1 = 8 := by\n"
            "  subst h\n  rfl\n")


def test_selection_is_fixed_stratified_validation_only_and_never_reads_test(monkeypatch, tmp_path):
    names = [f"{group}_{i}" for group in dev.GROUPS for i in range(6)]
    index = "\n".join("import MiniF2F.Valid." + name for name in names)
    calls = []

    def fetch(path):
        calls.append(path)
        assert "Test" not in path
        if path == "MiniF2F/Valid.lean":
            return index
        if path == "lean-toolchain":
            return "leanprover/lean4:v4.24.0"
        if path == "lake-manifest.json":
            return json.dumps({"packages": [{"name": "mathlib", "rev": "pinned"}]})
        return _source(Path(path).stem)

    monkeypatch.setattr(dev, "_fetch", fetch)
    cohort = dev._prepare(tmp_path / "sources", 42, 5)
    assert len(cohort["tasks"]) == 25
    assert all(sum(t["group"] == group for t in cohort["tasks"]) == 5 for group in dev.GROUPS)
    assert [t["id"] for t in cohort["tasks"]] == [name for name, _ in dev._select(index, 42, 5)]
    assert all("rfl" not in t["header"] for t in cohort["tasks"])
    assert dev._prepare(tmp_path / "sources", 42, 5) == cohort
    assert len(calls) == 28  # Three pinned metadata files and 25 sources, no repeat download.
    first = cohort["tasks"][0]
    (tmp_path / "sources" / first["file"]).write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        dev._prepare(tmp_path / "sources", 42, 5)


def test_header_boundary_ignores_comments_and_nested_default_assignments():
    source = ('import Mathlib\n/- theorem hidden : False := by sorry /- nested -/ -/\n'
              'theorem goal (n : Nat := 1) (h : n = n /- := ignored -/) : n = n := by rfl')
    parsed = dev._parse("goal", source)
    assert parsed["header"].endswith(": n = n")
    assert "(n : Nat := 1)" in parsed["header"]
    assert "rfl" not in parsed["header"]
    with pytest.raises(ValueError, match="pre-theorem context"):
        dev._parse("goal", "import MiniF2F.Valid.goal\ntheorem goal : True := by trivial")


def test_prompt_never_contains_original_proof():
    parsed = dev._parse("goal", _source("goal"))
    task = {"id": "goal", "ingest_status": "parsed", **parsed}
    prompt = dev._messages(task)[0]["content"]
    assert "subst h" not in prompt and "rfl" not in prompt
    assert parsed["header"] in prompt and "sorry" in prompt
    spec = dev._spec(task)
    assert "value := info.type" in spec
    assert "Benchmark target already available" in spec


def test_automation_first_skips_solved_and_incompatible_tasks_without_changing_denominator():
    tasks = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    report = {"tasks": [{"id": "a", "attempts": [], "automation": {"status": "passed"}},
                         {"id": "b", "attempts": [], "complete": True, "input_status": "failed"},
                         {"id": "c", "attempts": [], "automation": {"status": "lean_rejected"}}]}
    calls = []

    def generate(messages, seed):
        calls.append((messages, seed))
        return {"raw_text": "proof", "seed": seed}

    smoke._run(report, generate, lambda *a: {"status": "passed"}, lambda: None, 2, 42,
               tasks=tasks, messages_factory=lambda task, previous: [{"role": "user", "content": task["id"]}],
               automation_first=True)
    assert len(calls) == 1 and calls[0][0][0]["content"] == "c"
    assert calls[0][1] == 62
    assert smoke._summary(report)["selected_tasks"] == 3


def test_baseline_then_model_resume_loads_no_model_during_baseline(tmp_path, monkeypatch):
    project = tmp_path / "lean"
    project.mkdir()
    (project / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "mathlib", "rev": "local"}]}))
    tasks = [{"id": name, "group": "imo", "ingest_status": "parsed", **dev._parse(name, _source(name))}
             for name in ("one", "two")]
    cohort = {"tasks": tasks, "upstream_toolchain": "upstream", "upstream_mathlib": "other"}
    def prepare(root, *args):
        root.mkdir(exist_ok=True)
        return cohort

    monkeypatch.setattr(dev, "_prepare", prepare)
    monkeypatch.setattr(control := dev.control, "_lean_context", lambda _: (project, "local-lean", ""))
    monkeypatch.setattr(control, "_runtime", lambda: {"fixed": "version"})
    monkeypatch.setattr(control, "_check_source", lambda *a: {"status": "passed"})
    monkeypatch.setattr(dev, "_validate", lambda task, raw, *a: {
        "status": "passed" if task["id"] == "one" or raw == "generated" else "lean_rejected"})
    loads, caches = [], []
    monkeypatch.setattr(smoke, "_cached_weights", lambda _: caches.append(1) or {})
    monkeypatch.setattr(control, "_preflight", lambda _: {"runtime": {}})

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *a, **k):
            assert k["local_files_only"]
            return cls()

        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            text = json.dumps(messages)
            return list(text.encode()) if tokenize else text

        def encode(self, text, **kwargs):
            return list(text.encode())

        def decode(self, ids, **kwargs):
            return bytes(ids).decode()

    class Model:
        def __init__(self, args, tokenizer):
            assert args.local_files_only

        def __enter__(self):
            loads.append(1)
            return self

        def __exit__(self, *args):
            pass

        def generate(self, ids, seed):
            return {"raw_text": "generated", "seed": seed}

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(PreTrainedTokenizerFast=Tokenizer))
    monkeypatch.setattr(control, "ReferenceGenerator", Model)
    output = tmp_path / "output"
    argv = ["--output-dir", str(output), "--lean-project", str(project), "--allow-environment-port"]
    assert dev.main([*argv, "--baseline-only"]) == 0
    assert not loads and not caches
    assert dev.main([*argv, "--resume"]) == 0
    assert loads == caches == [1]
    result = json.loads((output / "report.json").read_text())
    assert result["summary"]["combined_solved"] == 2
    assert result["summary"]["model_calls"] == 1
    assert result["manifest"]["environment_mode"] == "explicit_development_port"


@pytest.mark.skipif(not os.environ.get("MATHS_AI_MINIF2F_CORE_LEAN"), reason="installed Lean core required")
def test_real_lean_freezes_original_binder_type_and_forbids_using_spec_stub(tmp_path):
    task = {"id": "core_target", "ingest_status": "parsed", "prefix": "import Lean\n",
            "header": "theorem core_target (a b c : Nat) : a + b = a + c → b = c"}
    context = (tmp_path, os.environ["MATHS_AI_MINIF2F_CORE_LEAN"], "")
    valid = dev._validate(task, "intro h\nexact Nat.add_left_cancel h", context, 30)
    assert valid["status"] == "passed", valid.get("diagnostics")
    cheat = dev._validate(task, "intro h\nexact Expected_core_target a b c h", context, 30)
    assert cheat["status"] == "lean_rejected"
    assert "Expected_core_target" in cheat["axioms"]
    changed = dev._validate(task, "theorem different (a b c : Nat) (h : False) : "
                            "a + b = a + c → b = c := by exact fun _ => False.elim h", context, 30)
    assert changed["status"] == "lean_rejected"
