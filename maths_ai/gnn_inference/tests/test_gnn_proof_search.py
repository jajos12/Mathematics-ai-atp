"""Search and native acceptance: saved parents, budgets, and audited proofs."""

import asyncio
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveCandidate
from maths_ai.gnn_inference.atp_lean_gnn.model_graph import prepare_live_goal
from maths_ai.gnn_inference.atp_lean_gnn.native_session import LiveLeanSession, LiveStateError, NativeSnapshot, TacticFailure
from maths_ai.gnn_inference.atp_lean_gnn.proof_search import GnnSearchBudget, search_gnn
from maths_ai.gnn_inference.atp_lean_gnn.proof_verification import LeanProofVerifier


HEADER = "theorem SubmissionSearch (P : Prop) (h : P) : P"


class Policy:
    def __init__(self, actions):
        self.actions, self.calls = actions, []

    def predict(self, goal, *, top_k):
        self.calls.append(goal["name"])
        candidates = [LiveCandidate(code, code, -cost, ()) for code, cost in self.actions.get(goal["name"], [])]
        return SimpleNamespace(candidates=candidates[:top_k], rejected=[])


class Session:
    prefix, imports, timeout = "", ("Init",), 1

    def __init__(self, transitions):
        self.transitions, self.steps = transitions, []

    async def start_declaration(self, header):
        return NativeSnapshot(0, ({"name": "root"},))

    async def step(self, parent, code):
        self.steps.append((parent.state_id, code))
        child = self.transitions[parent.state_id, code]
        if isinstance(child, Exception):
            raise child
        return child


class Verifier:
    def __init__(self, accepted):
        self.accepted, self.paths = accepted, []

    async def verify(self, header, path, *, timeout):
        self.paths.append(path)
        return {"status": "passed" if path in self.accepted else "lean_rejected", "seconds": 0.001}


def test_best_first_restores_parents_and_continues_after_failed_verification():
    policy = Policy({"root": [("bad", .01), ("untrusted", .1), ("split", .2), ("expensive", 5)],
                     "child": [("finish", .3)]})
    session = Session({(0, "bad"): TacticFailure("bad action"), (0, "untrusted"): NativeSnapshot(1, ()),
                       (0, "split"): NativeSnapshot(2, ({"name": "child"},)), (2, "finish"): NativeSnapshot(3, ())})
    verifier = Verifier([["split", "finish"]])
    report = asyncio.run(search_gnn(policy, session, HEADER, verifier))
    assert report["status"] == "solved" and report["proof_verified"]
    assert report["accepted_path"] == ["split", "finish"]
    assert session.steps == [(0, "bad"), (0, "untrusted"), (0, "split"), (2, "finish")]
    assert policy.calls == ["root", "child"]
    assert report["nodes"][1]["verification_rejected"]
    assert report["model_calls"] == report["model_responses"] == 2
    assert report["tactic_trials"] == 4


@pytest.mark.parametrize("limit,status,trials", [
    ({"nodes": 1}, "node_budget", 0), ({"tactic_trials": 1}, "trial_budget", 1),
    ({"depth": 1}, "depth_budget", 2), ({"model_calls": 1}, "model_call_budget", 2),
])
def test_search_resource_limits(limit, status, trials):
    policy = Policy({"root": [("bad", .1), ("next", .2)], "child": [("finish", .1)]})
    session = Session({(0, "bad"): TacticFailure("invalid"), (0, "next"): NativeSnapshot(1, ({"name": "child"},))})
    report = asyncio.run(search_gnn(policy, session, HEADER, Verifier([]), budget=GnnSearchBudget(**limit)))
    assert report["status"] == status and not report["proof_verified"]
    assert report["tactic_trials"] == trials
    assert report["model_calls"] == 1


def test_cancelled_trial_is_checkpointed_without_new_model_budget():
    class Interrupted(Session):
        async def step(self, parent, code):
            raise asyncio.CancelledError

    checkpoints = []
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(search_gnn(Policy({"root": [("stop", .1)]}), Interrupted({}), HEADER, Verifier([]),
                               checkpoint=lambda report: checkpoints.append(dict(report))))
    assert checkpoints[-1]["status"] == "interrupted"
    assert checkpoints[-1]["model_calls"] == checkpoints[-1]["tactic_trials"] == 1


@pytest.fixture
def native_root():
    value = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not value:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned native Lean acceptance")
    return Path(value).resolve()


def test_native_context_siblings_failed_alternatives_and_axiom_audit(native_root):
    async def check():
        prefix = "namespace SubmissionContext\nvariable (P : Prop)\nlocal notation \"Keep\" => P\n"
        header = "theorem preserved (h : Keep) : Keep ∧ Keep"
        verifier = LeanProofVerifier(source_root=native_root, imports=("Init",), prefix=prefix)
        async with LiveLeanSession(source_root=native_root, pantograph_root=native_root, imports=("Init",), prefix=prefix) as session:
            root = await session.start_declaration(header)
            assert root is await session.start_declaration(header)
            assert [v["contextIndex"] for v in root.goals[0]["vars"]] == [0, 2]
            split = await session.step(root, "constructor")
            assert len(split.goals) == 2
            with pytest.raises(TacticFailure):
                await session.step(root, "exact Nat.zero")
            one = await session.step(split, "exact h")
            alternative = await session.step(split, "exact h")
            assert len(one.goals) == len(alternative.goals) == 1
            assert not (await session.step(root, "skip")).closed
            assert (await session.step(one, "exact h")).closed
        assert session.proc is None and session.temporary is None
        assert (await verifier.verify(header, ["constructor", "exact h", "exact h"], timeout=30))["status"] == "passed"
        placeholder = await verifier.verify(header, ["sorry"], timeout=30)
        assert placeholder["status"] == "lean_rejected" and "sorryAx" in placeholder["forbidden_axioms"]
        assert (await verifier.verify(header, ["exact h"], timeout=30))["status"] == "lean_rejected"
        async with LiveLeanSession(source_root=native_root, pantograph_root=native_root, imports=("Init",)) as session:
            with pytest.raises(LiveStateError, match="GNN_TARGET_ALREADY_AVAILABLE"):
                await session.start_declaration("theorem True.intro : True")
        assert (await LeanProofVerifier(source_root=native_root, imports=("Init",)).verify(
            "theorem True.intro : True", ["exact True.intro"], timeout=30))["status"] == "lean_rejected"

    asyncio.run(check())


def test_native_timeout_kills_process(native_root):
    async def check():
        async with LiveLeanSession(source_root=native_root, pantograph_root=native_root, imports=("Init",), timeout=.1) as session:
            root = await session.start_declaration(HEADER)
            with pytest.raises(asyncio.TimeoutError):
                await session.step(root, "run_tac do\n  IO.sleep 10000")
            assert session.proc is None and session.temporary is None
    asyncio.run(check())


def test_native_let_and_instance_context_are_authoritative(native_root):
    async def check():
        header = "theorem SubmissionLetInstance (α : Type) [Inhabited α] : (default : α) = default"
        async with LiveLeanSession(source_root=native_root, pantograph_root=native_root, imports=("Init",)) as session:
            root = await session.start_declaration(header)
            assert any(variable["isInstance"] for variable in root.goals[0]["vars"])
            child = await session.step(root, "let z : α := default")
            graph = prepare_live_goal(child.goals[0], {"<UNK>": 0}, edge_mode="bidirectional")
            assert any(variable["isLet"] for variable in child.goals[0]["vars"])
            assert ":=" in graph.text and "z" in graph.text
            assert (await session.step(child, "rfl")).closed
        check = await LeanProofVerifier(source_root=native_root, imports=("Init",)).verify(header, ["let z : α := default", "rfl"], timeout=30)
        assert check["status"] == "passed"
    asyncio.run(check())


@pytest.mark.parametrize("cancel", [False, True])
def test_independent_verifier_timeout_and_cancel_reap_process_group(tmp_path, monkeypatch, cancel):
    async def check():
        original = asyncio.create_subprocess_exec
        processes, started = [], asyncio.Event()

        async def sleeping_compiler(*args, **kwargs):
            process = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
            processes.append(process)
            started.set()
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", sleeping_compiler)
        verifier = LeanProofVerifier(source_root=tmp_path, imports=("Init",))
        pending = asyncio.create_task(verifier.verify(HEADER, ["exact h"], timeout=30 if cancel else .02))
        await started.wait()
        if cancel:
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            assert (await pending)["status"] == "verification_timeout"
        assert len(processes) == 1 and processes[0].returncode is not None and processes[0].returncode < 0

    asyncio.run(check())
