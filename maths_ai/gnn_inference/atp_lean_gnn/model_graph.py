"""Normalized Lean input adapter using the shared DAG and tensorization code.

This is the accepted model's input format, not an action-generation template.
Legacy text graph construction remains available through graph.py.
"""

from dataclasses import dataclass

import torch
from torch_geometric.data import Batch

from .graph import DAGBuilder, BINDER_KIND_FORALL, BINDER_KIND_LAMBDA, BINDER_KIND_LET, root_state_node_id
from .native_session import LiveStateError
from .parser import parse_sexp_string
from .pyg import MODEL_NODE_TYPE_TO_ID, build_premise_mask, dag_to_pyg, transform_edge_index


def _payload(value):
    return "(" + " ".join(_payload(child) for child in value) + ")" if isinstance(value, list) else str(value)


def _walk(value, context, dag):
    if not isinstance(value, list):
        text = str(value)
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            index = int(text)
            return context[index] if 0 <= index < len(context) else dag.get_or_create(f"?db-{index}", (), node_type="var")
        return dag.get_or_create(text, (), node_type="sconst")
    if not value:
        return dag.get_or_create("()", (), node_type="sconst")
    head = value[0]
    if head in (":forall", ":lambda"):
        if len(value) not in (4, 5):
            raise ValueError("malformed binder S-expression")
        name, role = value[1], None
        if len(value) == 5 and str(value[2]).startswith(":"):
            role, typ, body = value[2:]
        else:
            typ, body = value[2:4]
            if len(value) == 5:
                role = value[4]
        variable = dag.create_bound_variable(str(name), binder_depth=len(context) + 1,
            binder_kind=BINDER_KIND_FORALL if head == ":forall" else BINDER_KIND_LAMBDA)
        children = [variable, _walk(typ, context, dag), _walk(body, [variable, *context], dag)]
        if role is not None:
            children.append(dag.get_or_create(f"BinderRole:{role}", (), node_type="sconst"))
        return dag.get_or_create(head, tuple(children), node_type="sbinder")
    if head == ":let":
        if len(value) != 5:
            raise ValueError("malformed let S-expression")
        _, name, typ, assigned, body = value
        variable = dag.create_bound_variable(str(name), binder_depth=len(context) + 1, binder_kind=BINDER_KIND_LET)
        return dag.get_or_create(head, (variable, _walk(typ, context, dag), _walk(assigned, context, dag),
            _walk(body, [variable, *context], dag)), node_type="sbinder")
    if head in (":c", ":fv", ":sort", ":lit"):
        if len(value) != 2:
            raise ValueError("malformed atomic S-expression")
        label, node_type = str(value[1]), {":c": "const", ":fv": "var", ":sort": "type", ":lit": "const"}[head]
        if head == ":sort":
            label = "Prop" if value[1] in ("0", "Prop") else "Type" if value[1] in ("1", "Type") else f"Sort-{value[1]}"
        elif head == ":lit":
            label = "Lit:" + _payload(value[1])
        return dag.get_or_create(label, (), node_type=node_type)
    if head == ":mdata":
        if len(value) != 3:
            raise ValueError("malformed metadata S-expression")
        metadata = dag.get_or_create("Metadata:" + _payload(value[1]), (), node_type="sconst")
        return dag.get_or_create(head, (metadata, _walk(value[2], context, dag)), node_type="meta")
    if head == ":proj":
        if len(value) != 4:
            raise ValueError("malformed projection S-expression")
        structure = dag.get_or_create(str(value[1]), (), node_type="const")
        index = dag.get_or_create(f"Field:{value[2]}", (), node_type="sconst")
        return dag.get_or_create(head, (structure, index, _walk(value[3], context, dag)), node_type="sapp")
    if head == ":arg":
        if len(value) not in (3, 4):
            raise ValueError("malformed argument S-expression")
        children = [dag.get_or_create(f"ArgRole:{value[1]}", (), node_type="sconst"),
                    dag.get_or_create(f"ArgPosition:{value[2]}", (), node_type="sconst")]
        if len(value) == 4:
            children.append(_walk(value[3], context, dag))
        return dag.get_or_create(head, tuple(children), node_type="sapp")
    if head in (":instance-of", ":proof-of"):
        if len(value) != 2:
            raise ValueError("malformed normalized placeholder")
        return dag.get_or_create(head, (_walk(value[1], context, dag),), node_type="meta")
    if head == ":metavar":
        if len(value) != 1:
            raise ValueError("malformed metavariable")
        return dag.get_or_create(head, (), node_type="meta")
    if head == ":app":
        if len(value) < 2:
            raise ValueError("model application needs a function")
        return dag.get_or_create(head, tuple(_walk(child, context, dag) for child in value[1:]), node_type="sapp")
    if isinstance(head, str) and head.startswith(":"):
        return dag.get_or_create(head, tuple(_walk(child, context, dag) for child in value[1:]), node_type="sconst")
    if len(value) >= 2:
        return dag.get_or_create("App", tuple(_walk(child, context, dag) for child in value), node_type="sapp")
    return dag.get_or_create(str(value), ())


def sexp_to_dag(source):
    dag = DAGBuilder()
    dag.expression_root_id = _walk(parse_sexp_string(source), [], dag)
    return dag


def normalized_state_to_dag(goal_sexp, hypotheses):
    dag = sexp_to_dag(goal_sexp)
    goal_id, roots = dag.expression_root_id, []
    for hypothesis in hypotheses:
        if isinstance(hypothesis, dict):
            name, sexp, index = hypothesis.get("name", "_"), hypothesis.get("sexp"), hypothesis.get("context_index")
            role = str(hypothesis.get("binder_role", ":explicit"))
            instance, assigned = bool(hypothesis.get("is_instance", False)), bool(hypothesis.get("is_let", False))
        else:
            name, sexp = hypothesis
            index, role, instance, assigned = None, ":explicit", False, False
        name_node = dag.get_or_create(name or "_", (), node_type="sconst" if isinstance(index, int) else "var")
        typ = _walk(parse_sexp_string(str(sexp)), [], dag) if sexp else dag.get_or_create("?", ())
        if isinstance(index, int):
            pointer = dag.get_or_create(f"FV{index}", (), node_type="var")
            role = "instance" if instance else "let" if assigned else role.removeprefix(":")
            role_node = dag.get_or_create(f"HypRole:{role}", (), node_type="sconst")
            children = (pointer, name_node, role_node, typ)
        else:
            children = (name_node, typ)
        roots.append(dag.get_or_create("Hyp", children))
    roots.append(dag.get_or_create("Goal", (goal_id,)))
    dag.get_or_create("State", tuple(roots))
    return dag


def _expression(value):
    if (not isinstance(value, dict) or type(value.get("modelSexpVersion")) is not int
            or value["modelSexpVersion"] != 1 or not isinstance(value.get("modelSexp"), str)
            or not value["modelSexp"].strip() or not isinstance(value.get("pp"), str)):
        raise LiveStateError("complete version-1 normalized expression and pretty text required")
    return value["modelSexp"]


def normalized_goal_record(goal):
    try:
        if not isinstance(goal, dict) or goal.get("isConversion") is not False:
            raise LiveStateError("ordinary tactic-mode goal required")
        target = _expression(goal.get("target"))
        if not isinstance(goal.get("vars"), list):
            raise LiveStateError("complete local context required")
        indices, hypotheses = set(), []
        for variable in goal["vars"]:
            index = variable.get("contextIndex")
            if (type(index) is not int or index < 0 or index in indices or not isinstance(variable.get("userName"), str)
                    or not isinstance(variable.get("binderRole"), str)):
                raise LiveStateError("unique input indices, names and binder roles required")
            indices.add(index)
            hypotheses.append({"name": variable["userName"], "internal_name": str(variable.get("name", "")),
                "context_index": index, "binder_role": variable["binderRole"],
                "is_instance": bool(variable.get("isInstance", False)), "is_let": bool(variable.get("isLet", False)),
                "sexp": _expression(variable.get("type"))})
        for source in [target, *(hypothesis["sexp"] for hypothesis in hypotheses)]:
            pending = [parse_sexp_string(source)]
            while pending:
                node = pending.pop()
                if not isinstance(node, list):
                    continue
                if node and node[0] == ":fv" and (len(node) != 2 or not isinstance(node[1], str)
                        or not node[1].startswith("FV") or not node[1][2:].isdecimal() or int(node[1][2:]) not in indices):
                    raise LiveStateError("model expression references an unknown input local")
                pending.extend(node)
        return {"goal_sexp": target, "hyp_sexps": hypotheses}
    except LiveStateError:
        raise
    except (ValueError, TypeError, AttributeError, RecursionError) as error:
        raise LiveStateError(f"invalid normalized input: {error}") from error


@dataclass
class LiveGraph:
    dag: DAGBuilder
    batch: Batch
    normalized: dict
    text: str
    local_node_context: dict[int, int]
    local_context: list[dict]


def prepare_live_goal(goal, node_vocab, *, edge_mode):
    try:
        record = normalized_goal_record(goal)
        lines = []
        for variable in goal["vars"]:
            line = f"{variable['userName'] or '_'} : {variable['type']['pp']}"
            if variable.get("isLet"):
                if not isinstance(variable.get("value"), dict) or not isinstance(variable["value"].get("pp"), str):
                    raise LiveStateError("pretty value required for local let")
                line += f" := {variable['value']['pp']}"
            lines.append(line)
        text = "\n".join(lines + [f"⊢ {goal['target']['pp']}"])
        dag = normalized_state_to_dag(record["goal_sexp"], record["hyp_sexps"])
        data = dag_to_pyg(dag, node_vocab, node_type_to_id=MODEL_NODE_TYPE_TO_ID)
        data.state_node_index = torch.tensor([root_state_node_id(dag)], dtype=torch.long)
        data.premise_mask = torch.tensor(build_premise_mask(dag), dtype=torch.bool)
        data.edge_index = transform_edge_index(data.edge_index, edge_mode=edge_mode, deduplicate=False)
        indices = {hypothesis["context_index"] for hypothesis in record["hyp_sexps"]}
        local_nodes = {node.id: int(node.label[2:]) for node in dag.nodes if node.node_type == "var" and node.is_bound == 0
            and node.label.startswith("FV") and node.label[2:].isdecimal() and int(node.label[2:]) in indices}
        for node in dag.nodes:
            if node.label == "Hyp" and len(node.children) == 4 and node.children[0] in local_nodes:
                local_nodes[node.id] = local_nodes[node.children[0]]
        context = [{"context_index": variable["contextIndex"], "user_name": variable["userName"],
                    "isInaccessible": bool(variable.get("isInaccessible", False))} for variable in goal["vars"]]
        return LiveGraph(dag, Batch.from_data_list([data]), record, text, local_nodes, context)
    except LiveStateError:
        raise
    except (ValueError, TypeError, AttributeError, RecursionError) as error:
        raise LiveStateError(f"normalized graph conversion failed: {error}") from error
