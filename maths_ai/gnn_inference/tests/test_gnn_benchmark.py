"""Fixed cohort ingestion and fault-injected task-boundary resume tests."""

import asyncio
import csv
import json
import os
from types import SimpleNamespace

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn import benchmark
from maths_ai.gnn_inference.atp_lean_gnn.proof_search import GnnSearchBudget
from maths_ai.gnn_inference.atp_lean_gnn.reporting import atomic_json


def sources():
    return {"MiniF2F/Valid.lean": "\n".join(f"import MiniF2F.Valid.{name}" for name in ("mathd_algebra_1", "imo_2", "amc_3")),
        "lean-toolchain": "fixture-toolchain", "lake-manifest.json": '{"packages":[{"name":"mathlib","rev":"fixture-revision"}]}',
        **{f"MiniF2F/Valid/{name}.lean": f"import Mathlib\nopen Nat\ntheorem {name} (P : Prop) (h : P) : P := by\n  sorry\n"
           for name in ("mathd_algebra_1", "imo_2", "amc_3")}}


def test_fixed_selection_original_boundaries_readonly_cache_and_source_integrity(tmp_path):
    upstream = sources()
    upstream["MiniF2F/Valid/amc_3.lean"] = "import Mathlib\nnamespace Unexpected\ntheorem amc_3 : True := by trivial"
    cohort = benchmark.prepare_cohort(tmp_path / "cohort", fetch=upstream.__getitem__, expected_count=3)
    assert [(task["id"], task["group"]) for task in cohort["tasks"]] == benchmark.selected_tasks(upstream["MiniF2F/Valid.lean"], 42)
    assert len(cohort["tasks"]) == 3
    assert sum(task["ingest_status"] == "unsupported" for task in cohort["tasks"]) == 1
    parsed = next(task for task in cohort["tasks"] if task["ingest_status"] == "parsed")
    assert "sorry" not in parsed["header"] and parsed["prefix"] == "import Mathlib\nopen Nat\n"
    owner = tmp_path / "cohort/gnn_cache_owner.json"
    owner.unlink()  # A complete baseline cache is accepted read-only.
    before = (tmp_path / "cohort/cohort.json").read_bytes()
    assert benchmark.prepare_cohort(tmp_path / "cohort", fetch=lambda _: pytest.fail("unexpected fetch"), expected_count=3) == cohort
    assert (tmp_path / "cohort/cohort.json").read_bytes() == before
    (tmp_path / "cohort" / parsed["file"]).write_text("theorem changed : True := by trivial")
    with pytest.raises(ValueError, match="source changed"):
        benchmark.prepare_cohort(tmp_path / "cohort", expected_count=3)


def test_owned_partial_ingestion_recovers_without_reselecting(tmp_path):
    upstream = sources()
    failures = []

    def flaky(path):
        if path.endswith(".lean") and path != "MiniF2F/Valid.lean" and not failures:
            failures.append(path)
            raise OSError("interrupted download")
        return upstream[path]

    with pytest.raises(OSError):
        benchmark.prepare_cohort(tmp_path / "cohort", fetch=flaky, expected_count=3)
    assert len(benchmark.prepare_cohort(tmp_path / "cohort", fetch=flaky, expected_count=3)["tasks"]) == 3
    with pytest.raises(ValueError, match="protocol"):
        benchmark.prepare_cohort(tmp_path / "cohort", seed=43, expected_count=3)
    with pytest.raises(ValueError, match="population"):
        benchmark.prepare_cohort(tmp_path / "wrong", fetch=upstream.__getitem__)


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    (root / "bundle.json").write_text('{"fixture":true}')
    system = SimpleNamespace(root=root, policy=torch.nn.Linear(1, 1), manifest={"policy_state_sha256": "policy", "retriever_state_sha256": "retriever"},
                             profile="accepted_local_cues")
    excluded, calls, rechecks = [], [], []

    class Policy:
        def excluding(self, names):
            excluded.append(names)
            return self

    policy = Policy()
    policy.system = system

    class Session:
        prefix = ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def start_declaration(self, header):
            return None

    class Verifier:
        async def verify(self, header, path, *, timeout):
            rechecks.append((header, path))
            return {"status": "passed", "seconds": .01}

    async def search(policy, session, header, verifier, *, budget, checkpoint):
        calls.append(header)
        report = {"status": "running", "proof_verified": False, "model_calls": 1, "model_responses": 0,
            "tactic_trials": 1, "seconds": .1, "nodes": [{"id": 0}], "accepted_path": [], "verifications": []}
        checkpoint(report)
        if "imo_2" in header:
            raise asyncio.CancelledError
        report.update(status="solved", proof_verified=True, model_responses=1, accepted_path=["exact h"])
        checkpoint(report)
        return report

    monkeypatch.setattr(benchmark, "search_gnn", search)
    cohort = benchmark.prepare_cohort(tmp_path / "cohort", fetch=sources().__getitem__, expected_count=3)
    environment = {"source_root": "/fixture", "pantograph_root": "/fixture", "toolchain": "fixture-toolchain",
                   "revisions": {"source": "fixture-revision"}}
    kwargs = dict(environment=environment, budget=GnnSearchBudget(), preflight=False,
                  session_factory=lambda task: Session(), verifier_factory=lambda task: Verifier())
    return SimpleNamespace(policy=policy, cohort=cohort, output=tmp_path / "run", kwargs=kwargs, calls=calls, rechecks=rechecks, excluded=excluded)


def run(fixture, **kwargs):
    return asyncio.run(benchmark.run_cohort(fixture.policy, fixture.cohort, fixture.output, **(fixture.kwargs | kwargs)))


def test_interruption_resume_recheck_costs_and_no_fresh_budget(run_fixture):
    f = run_fixture
    with pytest.raises(asyncio.CancelledError):
        run(f, invocation_costs={"model_loading_seconds": 2.0})
    report = json.loads((f.output / "report.json").read_text())
    interrupted = json.loads((f.output / "tasks/imo_2.json").read_text())
    assert interrupted["status"] == "interrupted" and interrupted["end_to_end_seconds"] >= 0
    assert report["invocations"][0]["status"] == "interrupted"
    assert report["invocations"][0]["costs"]["model_loading_seconds"] == 2
    report = run(f, resume=True)
    assert report["complete"] and report["manifest"]["environment_mode"] == "upstream"
    assert report["manifest"]["inference"]["device"] == "cpu"
    assert report["summary"]["selected_tasks"] == report["summary"]["completed_tasks"] == 3
    assert report["summary"]["verified_solved"] == 2
    assert report["summary"]["solve_rate"] == 2 / 3
    assert report["summary"]["model_calls"] == 3
    assert f.calls.count(next(header for header in f.calls if "imo_2" in header)) == 1
    assert len(f.excluded) == 3
    assert len(list(csv.DictReader((f.output / "summary.csv").open()))) == 3
    searches = len(f.calls)
    run(f, resume=True)
    assert len(f.calls) == searches and len(f.rechecks) >= 2


def test_resume_refuses_changed_protocol_missing_or_tampered_completed_tasks(run_fixture):
    f = run_fixture
    # Remove the intentionally interrupted fixture task for this complete run.
    f.cohort["tasks"] = [task for task in f.cohort["tasks"] if task["id"] != "imo_2"]
    run(f)
    with pytest.raises(ValueError, match="budgets"):
        run(f, resume=True, budget=GnnSearchBudget(nodes=2))
    path = f.output / "tasks" / (f.cohort["tasks"][0]["id"] + ".json")
    original = path.read_bytes()
    record = json.loads(original)
    record["search"]["accepted_path"] = ["sorry"]
    atomic_json(path, record)
    with pytest.raises(ValueError, match="record changed"):
        run(f, resume=True)
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(ValueError, match="refusing fresh budget"):
        run(f, resume=True)


def test_running_orphan_retains_attempt_costs_and_denominator(run_fixture):
    f = run_fixture
    with pytest.raises(asyncio.CancelledError):
        run(f)
    report = json.loads((f.output / "report.json").read_text())
    record = json.loads((f.output / "tasks/imo_2.json").read_text())
    record["status"] = "running"
    record["search"]["status"] = "running"
    atomic_json(f.output / "tasks/imo_2.json", record)
    del report["task_records"]["imo_2"]  # Crash between atomic task and aggregate writes.
    atomic_json(f.output / "report.json", report)
    result = run(f, resume=True)
    assert result["summary"]["elapsed_cost_is_lower_bound"]
    assert result["summary"]["model_calls"] == 3 and result["summary"]["completed_tasks"] == 3
    assert len(f.calls) == 3
    assert run(f, resume=True)["summary"]["elapsed_cost_is_lower_bound"]


def test_summary_rejects_unverified_solve_and_retains_input_failures():
    with pytest.raises(ValueError, match="unverified"):
        benchmark.summarize([{"status": "solved", "search": {"proof_verified": False}}], 244)
    summary = benchmark.summarize([{"status": "input_error"}, {"status": "input_unsupported"}], 244)
    assert summary["selected_tasks"] == 244 and summary["pending_tasks"] == 242 and summary["solve_rate"] == 0


def test_concurrent_resume_cannot_spend_duplicate_budgets(run_fixture, monkeypatch):
    f = run_fixture
    f.cohort["tasks"] = [task for task in f.cohort["tasks"] if task["id"] != "imo_2"]
    original = benchmark.search_gnn

    async def check():
        entered, release = asyncio.Event(), asyncio.Event()

        async def paused(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original(*args, **kwargs)

        monkeypatch.setattr(benchmark, "search_gnn", paused)
        first = asyncio.create_task(benchmark.run_cohort(f.policy, f.cohort, f.output, **f.kwargs))
        await entered.wait()
        try:
            with pytest.raises(RuntimeError, match="active writer"):
                await benchmark.run_cohort(f.policy, f.cohort, f.output, **(f.kwargs | {"resume": True}))
        finally:
            release.set()
            await first
        assert len(f.calls) == 2

    asyncio.run(check())


def test_real_accepted_policy_native_cohort_and_resume(tmp_path):
    bundle, root = os.environ.get("MATHS_AI_GNN_BUNDLE"), os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not bundle or not root:
        pytest.skip("set accepted bundle and native backend paths for end-to-end runner acceptance")
    from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveJointPolicy
    from maths_ai.gnn_inference.atp_lean_gnn.native_session import LiveLeanSession, environment_binding
    from maths_ai.gnn_inference.atp_lean_gnn.proof_verification import LeanProofVerifier
    environment = environment_binding(root, root, imports=("Init",))
    policy = LiveJointPolicy.from_bundle(bundle, device="cpu")
    cohort = {"upstream_toolchain": environment["toolchain"], "upstream_mathlib": environment["revisions"]["source"],
        "tasks": [{"id": "GnnRunnerFixture", "group": "fixture", "ingest_status": "parsed", "prefix": "",
                   "header": "theorem GnnRunnerFixture (P : Prop) (h : P) : P"}]}
    kwargs = {"environment": environment, "budget": GnnSearchBudget(),
        "session_factory": lambda task: LiveLeanSession(source_root=root, pantograph_root=root, imports=("Init",), prefix=task["prefix"]),
        "verifier_factory": lambda task: LeanProofVerifier(source_root=root, imports=("Init",), prefix=task["prefix"])}
    output = tmp_path / "real_runtime_fixture"
    result = asyncio.run(benchmark.run_cohort(policy, cohort, output, **kwargs))
    assert result["preflight_passed"] and result["complete"]
    assert result["summary"]["verified_solved"] == result["summary"]["model_calls"] == 1
    record_before = (output / "tasks/GnnRunnerFixture.json").read_bytes()
    resumed = asyncio.run(benchmark.run_cohort(policy, cohort, output, resume=True, **kwargs))
    assert resumed["summary"]["model_calls"] == 1
    assert resumed["invocations"][-1]["successful_proof_rechecks"][0]["status"] == "passed"
    assert (output / "tasks/GnnRunnerFixture.json").read_bytes() == record_before
