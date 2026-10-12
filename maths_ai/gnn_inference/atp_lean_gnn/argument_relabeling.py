"""Versioned flat-pointer targets, separate from all known lemma citations.

The decoder generates a tactic plus flat references, not arbitrary Lean syntax.
Compound terms, locations, modifiers and fresh names remain explicit unsupported
rows; they are never relabeled as valid stop-only actions.
"""

from __future__ import annotations

import re
from collections import defaultdict

import torch

from .action_targets import compile_source_syntax_trace
from .labels import clean_tactic_markup, parse_tactic_arguments


POLICY = "flat_reference_v2"
_PART = r"(?:[^\W\d][\w'′]*|«[^»]+»)"
_IDENTIFIER = re.compile(rf"{_PART}(?:\.{_PART})*\Z", re.UNICODE)
_LIST_TACTICS = {"rw", "rewrite", "rwa", "erw", "erewrite", "simp", "simp_rw",
                 "simpa", "simp_all", "dsimp", "norm_num", "rw_mod_cast",
                 "field_simp", "linarith", "nlinarith"}


def flat_reference_syntax(raw: str) -> tuple[list[str], str]:
    clean = clean_tactic_markup(raw).strip()
    parts = clean.split(maxsplit=1)
    if not parts:
        return [], "empty_tactic"
    if len(parts) == 1:
        return [], "supported"
    family, remainder = parts
    if remainder.startswith("[") and remainder.endswith("]") and family in _LIST_TACTICS:
        inner = remainder[1:-1].strip()
        tokens = [item.strip() for item in inner.split(",")] if inner else []
    else:
        tokens = remainder.split()
    if any(token in {"only", "at", "using", "with", "by"} for token in tokens):
        return [], "syntax_modifier_or_location"
    if not all(_IDENTIFIER.fullmatch(token) and token != "_" for token in tokens):
        return [], "compound_or_nonreference_syntax"
    # Names introduced by these tactics are not pointers into the input context.
    if family in {"intro", "intros", "rintro", "ext", "funext", "have", "let", "set", "obtain"}:
        return [], "fresh_name_or_binding_syntax"
    return tokens, "supported"


def _local_graph_names(data, node_vocab, id_to_label=None) -> dict[str, set[int]]:
    """Hyp children retain source names; map them to shared FV pointer nodes."""
    labels = id_to_label if id_to_label is not None else {int(value): name for name, value in node_vocab.items()}
    ids = data.x.tolist()
    incoming = defaultdict(list)
    for child, parent in data.edge_index.t().tolist():
        # Original DAG edges go from older children to newer parents. Ignore
        # reverse edges and retain child order within each Hyp node.
        if child < parent and child not in incoming[parent]:
            incoming[parent].append(child)
    result = defaultdict(set)
    for node, label_id in enumerate(ids):
        if labels.get(label_id) != "Hyp":
            continue
        children = incoming[node]
        if len(children) == 4:
            pointer, name, role, _ = children
            if not re.fullmatch(r"FV\d+", labels.get(ids[pointer], "")):
                continue
            if not labels.get(ids[role], "").startswith("HypRole:"):
                continue
        elif len(children) == 2:
            pointer = name = children[0]
        else:
            continue
        source_name = labels.get(ids[name], "")
        if source_name and bool(data.premise_mask[pointer]):
            result[source_name].add(pointer)
    return dict(result)


def relabel_arguments(data, *, node_vocab, name_to_id, max_args=3, trace=None, trace_error=None, id_to_label=None):
    if max_args < 1:
        raise ValueError("max_args must be positive")
    result = data.clone()
    tokens, reason = flat_reference_syntax(str(data.tactic_raw))
    local_names = _local_graph_names(data, node_vocab, id_to_label)
    annotation_by_source = defaultdict(set)
    context_names = {}
    citations = []
    outside_corpus = []
    if trace is not None:
        compile_source_syntax_trace(trace)  # verify roles/context indices, not just spelling
        context_names = {int(item["context_index"]): str(item.get("user_name", ""))
                         for item in trace["local_context"]}
        pending = [trace["source_syntax"]]
        while pending:
            node = pending.pop()
            pending.extend(reversed(node.get("children", [])))
            if node.get("tag") != "identifier":
                continue
            role = node.get("semanticRole")
            if role in {"global", "constructor"}:
                name = str(node["name"])
                annotation_by_source[node["source"]].add(("library", name))
                if name in name_to_id:
                    citations.append(name_to_id[name])
                else:
                    outside_corpus.append(name)
            elif role == "local":
                annotation_by_source[node["source"]].add(("local", int(node["contextIndex"])))
            elif role == "scoped_local":
                annotation_by_source[node["source"]].add(("scoped", node["source"]))
    else:
        _, reference_tokens = parse_tactic_arguments(str(data.tactic_raw))
        citations = [name_to_id[token] for token in reference_tokens
                     if token in name_to_id and token not in local_names]

    local_ids, lemma_ids = [], []
    label_ids = data.x.tolist()
    for token in tokens:
        local_id = lemma_id = -1
        annotations = annotation_by_source.get(token, set())
        if len(annotations) == 1:
            kind, identity = next(iter(annotations))
            if kind == "library":
                lemma_id = int(name_to_id.get(identity, -1))
            elif kind == "local":
                fv_label = node_vocab.get(f"FV{identity}")
                candidates = [i for i, label in enumerate(label_ids)
                              if label == fv_label and bool(data.premise_mask[i])]
                expected_name = context_names.get(identity, "")
                expected_nodes = local_names.get(expected_name, set())
                # A display name outside train vocabulary becomes <UNK>, but a
                # validated Lean context index still identifies its FV node.
                name_outside_vocab = expected_name not in node_vocab and bool(expected_name)
                if len(candidates) == 1 and (candidates[0] in expected_nodes or name_outside_vocab):
                    local_id = candidates[0]
        elif not annotations:
            candidates = local_names.get(token, set())
            if len(candidates) == 1:
                local_id = next(iter(candidates))
            elif not candidates:
                lemma_id = int(name_to_id.get(token, -1))
        local_ids.append(local_id)
        lemma_ids.append(lemma_id)
        if local_id < 0 and lemma_id < 0 and reason == "supported":
            reason = "unresolved_reference"
    if reason == "supported" and len(tokens) > max_args:
        reason = "over_decoder_budget"
    if trace_error is not None:
        reason = trace_error
    if reason == "supported":
        result.arg_node_indices = torch.tensor(local_ids, dtype=torch.long)
        result.arg_lemma_ids = torch.tensor(lemma_ids, dtype=torch.long)
        result.arg_count = len(tokens)
    else:
        # Keep true citations separately; unknown syntax is not zero-argument
        # stop supervision and must remain an explicit action-evaluation miss.
        result.arg_node_indices = torch.empty(0, dtype=torch.long)
        result.arg_lemma_ids = torch.empty(0, dtype=torch.long)
        result.arg_count = 0
    result.arg_label_policy = POLICY
    result.action_target_supported = reason == "supported"
    result.action_target_reason = reason
    result.action_target_max_args = max_args
    result.action_reference_count = len(tokens)
    result.citation_lemma_ids = torch.tensor(list(dict.fromkeys(citations)), dtype=torch.long)
    result.citation_count = int(result.citation_lemma_ids.numel())
    result.citation_outside_corpus_count = len(set(outside_corpus))
    result.citation_label_policy = "resolved_source_v3" if trace is not None else "clean_name_lookup_v2"
    return result
