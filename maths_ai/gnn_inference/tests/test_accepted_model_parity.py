"""Opt-in numerical acceptance against the preserved pre-template model runtime.

Set MATHS_AI_GNN_BUNDLE and MATHS_AI_GNN_REFERENCE to run the heavy comparison.
The reference is imported only in an isolated subprocess, never by the prover.
"""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from maths_ai.gnn_inference.atp_lean_gnn.flat_actions import FlatReference, RenderError, render_flat_action
from maths_ai.gnn_inference.atp_lean_gnn.model_graph import prepare_live_goal
from maths_ai.gnn_inference.atp_lean_gnn.native_session import LiveStateError


def expression(sexp, pp):
    return {"modelSexpVersion": 1, "modelSexp": sexp, "pp": pp}


def goals():
    variables = [
        {"contextIndex": 1, "userName": "P", "binderRole": ":explicit", "type": expression("(:sort 0)", "Prop")},
        {"contextIndex": 7, "userName": "h_unseen_submission", "binderRole": ":explicit", "type": expression("(:fv FV1)", "P")},
    ]
    identity = {"isConversion": False, "target": expression("(:fv FV1)", "P"), "vars": variables}
    equality = {"isConversion": False,
        "target": expression("(:app (:c Eq) (:arg :implicit 0 (:c Nat)) (:arg :explicit 1 (:fv FV3)) (:arg :explicit 2 (:lit 0)))", "n = 0"),
        "vars": [{"contextIndex": 3, "userName": "n", "binderRole": ":explicit", "type": expression("(:c Nat)", "Nat")},
                 {"contextIndex": 9, "userName": "inst", "binderRole": ":instImplicit", "isInstance": True,
                  "type": expression("(:app (:c Inhabited) (:arg :explicit 0 (:c Nat)))", "Inhabited Nat")},
                 {"contextIndex": 12, "userName": "z", "binderRole": ":explicit", "isLet": True,
                  "type": expression("(:c Nat)", "Nat"), "value": expression("(:lit 0)", "0")}]}
    structural = {"isConversion": False, "vars": [], "target": expression(
        "(:forall x :explicit (:c Nat) (:let y (:c Nat) (:lit 1) "
        "(:mdata (fixture data) (:app (:c Eq) (:arg :implicit 0 (:c Nat)) "
        "(:arg :explicit 1 (:proj Prod 0 (:app (:c Prod.mk) 1 0))) "
        "(:arg :explicit 2 (:app (:lambda z :implicit (:c Nat) 0) "
        "(:arg :instance 0 (:instance-of (:c Inhabited))) (:arg :proof 1 (:proof-of (:c True))) (:metavar)))))))",
        "∀ x : Nat, let y := 1; x = y")}
    return [identity, equality, structural]


def test_oov_native_identities_and_strict_graph_contract():
    graph = prepare_live_goal(goals()[0], {"<UNK>": 0}, edge_mode="bidirectional")
    assert graph.batch.x.unique().tolist() == [0]
    assert set(graph.local_node_context.values()) == {1, 7}
    assert render_flat_action("exact", [FlatReference("local", 7)], graph.local_context) == "exact «h_unseen_submission»"
    assert render_flat_action("intro", [], []) == "intro"
    with pytest.raises(RenderError, match="binding"):
        render_flat_action("intro", [FlatReference("local", 7)], graph.local_context)
    with pytest.raises(RenderError, match="placeholder"):
        render_flat_action("sorry", [], [])
    broken = goals()[0]
    broken["target"] = expression("(:fv FV99)", "not authoritative")
    with pytest.raises(LiveStateError, match="unknown input local"):
        prepare_live_goal(broken, {"<UNK>": 0}, edge_mode="forward")


def _snapshot(bundle, inputs, current):
    import importlib
    from dataclasses import replace
    import torch
    import torch.nn.functional as F
    from maths_ai.gnn_inference.atp_lean_gnn.joint_bundle import load_joint_source_bundle
    from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import candidate_groups_from_batch, local_type_signals
    from maths_ai.gnn_inference.atp_lean_gnn.live_policy import LiveJointPolicy, _action_log_probability
    from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidateSource

    adapter = importlib.import_module("maths_ai.gnn_inference.atp_lean_gnn." + ("model_graph" if current else "live_state"))
    system = load_joint_source_bundle(bundle, device="cpu", profile="accepted_local_cues")
    policy = system.policy
    node_names = {identity: label for label, identity in system.node_vocab.items()}
    tactic_names = {identity: label for label, identity in system.tactic_vocab.items()}
    live = LiveJointPolicy(system, edge_mode="bidirectional")
    snapshots = []
    with torch.inference_mode():
        for goal in inputs:
            graph = adapter.prepare_live_goal(goal, system.node_vocab, edge_mode="bidirectional")
            batch = graph.batch
            batch.y = torch.zeros(1, dtype=torch.long)
            nodes = policy.pointer.backbone.encode_nodes(batch)
            state = policy.pointer.backbone.readout(nodes, batch)[0]
            logits = policy.pointer.backbone.classifier(state)
            ids, vectors, scores = system.retrieve([graph.text])
            pool = candidate_groups_from_batch(batch, nodes, ids, vectors, scores)[0]
            row = {"dag": [node.as_dict() for node in graph.dag.nodes], "edges": graph.dag.edges,
                "graph": {key: batch[key] for key in ("x", "node_type", "is_bound", "binder_depth", "binder_kind", "edge_index", "state_node_index", "premise_mask")},
                "node_embeddings": nodes, "state": state, "tactic_logits": logits,
                "retrieval_ids": ids, "retrieval_vectors": torch.as_tensor(vectors[0]), "retrieval_scores": torch.as_tensor(scores[0]),
                "local_map": graph.local_node_context, "text": graph.text, "families": []}
            for tactic_id in logits.topk(8).indices.tolist():
                tactic = tactic_names[tactic_id]
                signal = local_type_signals(pool, batch.x, node_names, tactic_name=tactic, **graph.normalized)
                candidates = replace(pool, local_type_signal=signal, local_type_tactic=tactic)
                arguments, complete = policy.decode(state, tactic_id, candidates)
                hidden = policy.initial_state(state, tactic_id)
                family = {"tactic": tactic, "complete": complete,
                    "arguments": [(arg.source.value, arg.stable_id) for arg in arguments],
                    "count_logits": policy.arg_count_head(hidden), "steps": []}
                adapted = policy.lemma_adapter(candidates.lemma_vectors)
                for step, arg in enumerate(arguments):
                    family["steps"].append(policy.score_step(hidden, candidates, adapted=adapted, step=step))
                    identities = candidates.local_ids if arg.source is CandidateSource.LOCAL else candidates.lemma_ids
                    position = identities.index(arg.stable_id)
                    hidden = policy.pointer.argument_selector.gru(policy.selected_vector(candidates, arg.source, position, adapted=adapted), hidden)
                if complete:
                    family["log_probability"] = float(F.log_softmax(logits, dim=0)[tactic_id]) + _action_log_probability(policy, state, tactic_id, candidates, arguments)
                row["families"].append(family)
            prediction = live.predict(goal, top_k=8)
            row["actions"] = [candidate.to_dict() for candidate in prediction.candidates]
            row["rejected"] = prediction.rejected
            snapshots.append(row)
    return snapshots


def test_accepted_bundle_matches_reference_numerically(tmp_path):
    bundle, reference = os.environ.get("MATHS_AI_GNN_BUNDLE"), os.environ.get("MATHS_AI_GNN_REFERENCE")
    if not bundle or not reference:
        pytest.skip("set MATHS_AI_GNN_BUNDLE and MATHS_AI_GNN_REFERENCE for numerical release acceptance")
    current = Path(__file__).resolve().parents[3]
    script = "import sys, torch\n" + inspect.getsource(_snapshot) + f"\ntorch.save(_snapshot(sys.argv[1], {goals()!r}, sys.argv[3] == 'current'), sys.argv[2])\n"
    snapshots = []
    for label, root in (("reference", reference), ("current", current)):
        output = tmp_path / (label + ".pt")
        result = subprocess.run([sys.executable, "-B", "-c", script, bundle, str(output), label], cwd=root,
                                capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout + result.stderr
        snapshots.append(torch.load(output, weights_only=True))

    def compare(expected, actual, path="snapshot"):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6, msg=path)
        elif isinstance(expected, dict):
            assert expected.keys() == actual.keys(), path
            for key in expected:
                compare(expected[key], actual[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            assert len(expected) == len(actual), path
            for index, (left, right) in enumerate(zip(expected, actual)):
                compare(left, right, f"{path}[{index}]")
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=1e-6, abs=1e-6), path
        else:
            assert expected == actual, path

    compare(*snapshots)
