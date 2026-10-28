"""Bounded gold syntax extraction and native execution before model training."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from maths_ai.gnn_inference.atp_lean_gnn.dataset import DatasetRow
from maths_ai.gnn_inference.atp_lean_gnn.gold_replay import FlatReference, goal_signature, run_native_file
from maths_ai.gnn_inference.atp_lean_gnn.tactic_templates import (
    TEMPLATE_SCHEMA, RwSimpTemplate, TemplateError, TemplateLocation, TemplateRule,
    parse_rw_simp_trace, prepare_gold_rendering, render_rw_simp_template,
)


def node(kind, *children):
    return {"tag": "node", "kind": kind, "children": list(children)}


def atom(text):
    return {"tag": "atom", "source": text}


def ref(index=0, *, name=None):
    if name is not None:
        return {"tag": "identifier", "source": name, "semanticRole": "global", "name": name}
    return {"tag": "identifier", "source": f"h{index}", "semanticRole": "local", "contextIndex": index}


def location(index=None, *, goal=False):
    if index is None and not goal:
        return node("null")
    marker = node("null", atom("⊢")) if goal else node("null")
    names = node("null", ref(index)) if index is not None else node("null")
    return node("null", node("Lean.Parser.Tactic.location", atom("at"),
                             node("Lean.Parser.Tactic.locationHyp", names, marker)))


def rw_root(*references, backward=False, at=None, config=None):
    rules = []
    for i, reference in enumerate(references):
        if i:
            rules.append(atom(","))
        direction = node("null", atom("←")) if backward else node("null")
        rules.append(node("Lean.Parser.Tactic.rwRule", direction, reference))
    return node("Lean.Parser.Tactic.rwSeq", atom("rw"), config or node("null"),
                node("Lean.Parser.Tactic.rwRuleSeq", atom("["), node("null", *rules), atom("]")),
                at or node("null"))


def simp_root(*references, only=False, at=None):
    rules = []
    for i, reference in enumerate(references):
        if i:
            rules.append(atom(","))
        rules.append(node("Lean.Parser.Tactic.simpLemma", node("null"), node("null"), reference))
    arguments = node("null", atom("["), node("null", *rules), atom("]")) if references else node("null")
    return node("Lean.Parser.Tactic.simp", atom("simp"), node("null"), node("null"),
                node("null", atom("only")) if only else node("null"), arguments, at or node("null"))


def trace(root):
    return {"source_syntax": root, "syntax_args": [], "local_context": [
        {"context_index": i, "user_name": f"h{i}"} for i in range(5)
    ]}


def test_typed_rw_directions_location_and_rule_count_are_distinct_from_references():
    data = trace(rw_root(ref(name="Nat.add_comm"), backward=True, at=location(2)))
    action = parse_rw_simp_trace(data)
    assert action.family == "rw" and action.rules[0].backward
    assert action.rules[0].reference == FlatReference("library", "Nat.add_comm")
    assert action.location == TemplateLocation("local", FlatReference("local", 2))
    assert render_rw_simp_template(action, data["local_context"]) == "rw [← _root_.«Nat».«add_comm»] at «h2»"
    target = action.to_target()
    assert target["schema"] == TEMPLATE_SCHEMA and target["rule_count"] == 1
    assert target["location"]["reference"]["identity"] == 2
    assert target["rules"][0]["reference"]["source"] == "library"


def test_simp_only_and_single_tactic_wrappers_have_one_canonical_target():
    root = simp_root(ref(0), ref(name="Nat.add_zero"), only=True, at=location(2))
    direct = parse_rw_simp_trace(trace(root))
    wrapped = node("Lean.Parser.Tactic.tacticSeq",
                   node("Lean.Parser.Tactic.tacticSeq1Indented", node("null", root)))
    assert parse_rw_simp_trace(trace(wrapped)) == direct
    assert render_rw_simp_template(direct, trace(root)["local_context"]) == (
        "simp only [«h0», _root_.«Nat».«add_zero»] at «h2»"
    )
    assert render_rw_simp_template(parse_rw_simp_trace(trace(simp_root(only=True))), []) == "simp only []"
    assert render_rw_simp_template(parse_rw_simp_trace(trace(simp_root())), []) == "simp"


@pytest.mark.parametrize("root, code", [
    (rw_root(node("Lean.Parser.Term.app", ref(0), node("null", ref(1)))), "compound_rule_term"),
    (rw_root(ref(0), at=location(1, goal=True)), "combined_location"),
    (rw_root(ref(0), config=node("null", atom("config"))), "configuration"),
    (rw_root(ref(0), ref(1), ref(2), ref(3)), "rule_budget"),
    (node("Lean.Parser.Tactic.tacticSeq", node("null", rw_root(ref(0)), simp_root())), "multi_tactic_wrapper"),
    (rw_root(), "invalid_rw"),
])
def test_parser_rejects_unsupported_structures_without_flattening(root, code):
    with pytest.raises(TemplateError) as raised:
        parse_rw_simp_trace(trace(root))
    assert raised.value.code == code


def test_binding_and_unresolved_annotations_are_not_guessed():
    unknown = {"tag": "identifier", "source": "h"}
    with pytest.raises(TemplateError, match="unresolved_source"):
        parse_rw_simp_trace(trace(rw_root(unknown)))
    scoped = {"tag": "identifier", "source": "x", "semanticRole": "scoped_local"}
    with pytest.raises(TemplateError, match="new_binding_scope"):
        parse_rw_simp_trace(trace(rw_root(scoped)))
    with pytest.raises(TemplateError, match="invalid_source_trace"):
        parse_rw_simp_trace(trace(rw_root(ref(99))))


def test_target_invariants_and_location_role_do_not_create_an_extra_lemma():
    action = RwSimpTemplate("rw", tuple(TemplateRule(FlatReference("local", i)) for i in range(3)),
                           location=TemplateLocation("local", FlatReference("local", 3)))
    assert action.to_target()["rule_count"] == 3
    assert render_rw_simp_template(action, trace(rw_root(ref(0)))["local_context"]) == (
        "rw [«h0», «h1», «h2»] at «h3»"
    )
    with pytest.raises(TemplateError, match="invalid_location"):
        TemplateLocation("local", FlatReference("library", "Nat.add_comm"))
    with pytest.raises(TemplateError, match="simp_rule_modifier"):
        RwSimpTemplate("simp", (TemplateRule(FlatReference("local", 0), True),))


def test_parser_success_and_reference_rendering_failure_are_separate():
    data = trace(rw_root(ref(name="_private.A.0.foo")))
    prepared = prepare_gold_rendering("rw [foo]", data, renderer="template-v1")
    assert prepared["template_target"] is not None
    assert prepared["rendered_tactic"] is None
    assert prepared["template_error_code"] == "reference_resolution"
    data = trace(rw_root(ref(0), backward=True, at=location(1)))
    prepared = prepare_gold_rendering("rw [← h0] at h1", data, renderer="template-v1")
    assert not prepared["flat_renderable"] and prepared["template_target"]
    assert prepared["rendered_tactic"] == "rw [← «h0»] at «h1»"


def test_native_v1_templates_reproduce_source_transitions_and_coverage_boundaries():
    configured = os.environ.get("MATHS_AI_PANTOGRAPH_ROOT")
    if not configured:
        pytest.skip("set MATHS_AI_PANTOGRAPH_ROOT for pinned-Lean integration")
    from maths_ai.gnn_inference.atp_lean_gnn.sexpr_extraction import (
        PantographInvocationClient, _make_source_syntax_trace_record,
    )
    root = Path(configured).resolve()
    fixture = (Path(__file__).parent / "fixtures/template_replay.lean").read_bytes()
    supported = {
        "rw [h]", "rw [← h] at hb", "rw [h] at ha", "rw [h1, ← h2] at hx",
        "rw [h1, h2, h3] at hx", "rw [h,]", "rw [<- h]", "simp", "simp only []",
        "rw [h, ← h] at hx",
        "simp only [Nat.add_zero]", "simp only [h] at hx",
    }
    rejected = {"simp only [h] at hx ⊢", "rw [Nat.add_zero a]", "simp [*]"}
    with tempfile.TemporaryDirectory(prefix="TemplateSmoke_", dir=root) as temporary:
        source_path = Path(temporary) / "Example.lean"
        source_path.write_bytes(fixture)

        async def capture():
            client = PantographInvocationClient(source_root=root, pantograph_repl=root / ".lake/build/bin/repl",
                                                 capture_model_sexprs=True)
            try:
                return await client.process_file(str(source_path.relative_to(root)))
            finally:
                await client.close()

        units = asyncio.run(capture())
        assert all(not unit.get("messages") for unit in units), units
        requests, seen, actions = [], set(), {}
        for u, unit in enumerate(units):
            for v, invocation in enumerate(unit.get("invocations", [])):
                text = invocation["tactic"].strip()
                if text not in supported | rejected or text in seen:
                    continue
                seen.add(text)
                row_id = len(requests)
                row = DatasetRow(invocation["goalBefore"], "TemplateSmoke", text, "val", row_id)
                data = _make_source_syntax_trace_record(row, {"invocation": invocation}, {})
                prepared = prepare_gold_rendering(text, data, renderer="template-v1")
                if text in rejected:
                    assert prepared["rendered_tactic"] is None, prepared
                else:
                    assert prepared["rendered_tactic"] is not None, prepared
                    actions[text] = prepared["template_target"]
                syntax = data["source_syntax"]
                requests.append({"rowIndex": row_id, "unitIndex": u, "invocationIndex": v,
                                 "sourceStart": syntax["sourceStart"], "sourceEnd": syntax["sourceEnd"],
                                 "renderedTactic": prepared["rendered_tactic"]})
        assert seen == supported | rejected
        response = run_native_file(source_path, requests, source_root=root, pantograph_root=root, timeout=120)
    assert not response["frontend_errors"], response
    for request, native in zip(requests, response["rows"]):
        assert native["original"]["status"] == "executed", native
        assert goal_signature(native["original"]["goals"]) == goal_signature(native["source_after"])
        if request["renderedTactic"] is not None:
            assert native["rendered"]["status"] == "executed", native
            assert goal_signature(native["rendered"]["goals"]) == goal_signature(native["source_after"])
    assert actions["rw [h1, h2, h3] at hx"]["rule_count"] == 3
    assert actions["rw [h1, h2, h3] at hx"]["location"]["kind"] == "local"
    repeated = actions["rw [h, ← h] at hx"]["rules"]
    assert repeated[0]["reference"] == repeated[1]["reference"]
    assert not repeated[0]["backward"] and repeated[1]["backward"]
