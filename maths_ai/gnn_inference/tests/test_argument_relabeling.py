"""Citation markup is not supervision; unsupported syntax is not stop-only."""

import json

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.argument_relabeling import relabel_arguments
from maths_ai.gnn_inference.atp_lean_gnn.graph import proof_state_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.labels import parse_tactic_arguments
from maths_ai.gnn_inference.atp_lean_gnn.premise_retriever_training import extract_external_positive_ids
from maths_ai.gnn_inference.atp_lean_gnn.pyg import build_premise_mask, build_vocab, dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.argument_selector import compute_combined_loss
from maths_ai.gnn_inference.atp_lean_gnn.training import PreparedGraphDataset, load_prepared_metadata


def _data(tactic):
    dag = proof_state_to_dag(
        "h₁ : P\n⊢ P", goal_sexp="(:c P)",
        hyp_sexps=[{"name": "h₁", "context_index": 0, "sexp": "(:c P)"}],
    )
    vocab = build_vocab([dag])
    data = dag_to_pyg(dag, vocab, add_reverse_edges=True)
    data.premise_mask = torch.tensor(build_premise_mask(dag))
    data.y = torch.tensor([1])
    data.tactic_raw = tactic
    data.tactic_name = tactic.split()[0]
    data.arg_count = 0
    data.arg_node_indices = torch.empty(0, dtype=torch.long)
    data.arg_lemma_ids = torch.empty(0, dtype=torch.long)
    return data, vocab


def test_markup_is_removed_and_unicode_is_preserved():
    assert parse_tactic_arguments("rw [<a>pow_succ'</a>, <a>MulAction.mul_smul</a>]")[1] == [
        "pow_succ'", "MulAction.mul_smul",
    ]
    assert parse_tactic_arguments("exact h₁")[1] == ["h₁"]
    assert parse_tactic_arguments("have := h₁.mono_f")[1] == ["h₁.mono_f"]
    assert parse_tactic_arguments("rw [a, <a>Demo.foo</a>]")[1] == ["a", "Demo.foo"]


def test_legacy_recovery_keeps_cached_positions_until_relabeling():
    data, _ = _data("rw [<a>Demo.foo</a>]")
    data.arg_count = 3
    data.arg_node_indices = torch.tensor([-1, -1, -1])
    data.arg_lemma_ids = torch.tensor([-1, -1, -1])
    assert parse_tactic_arguments(data.tactic_raw, legacy=True)[1] == ["a", "Demo.foo", "a"]
    assert extract_external_positive_ids(Batch.from_data_list([data]), name_to_id={"Demo.foo": 100}) == [[100]]


def test_unicode_hypothesis_resolves_to_shared_fv_node_without_rebuilding_graph():
    data, vocab = _data("exact h₁")
    changed = relabel_arguments(data, node_vocab=vocab, name_to_id={})
    assert changed.action_target_supported
    assert changed.arg_count == 1
    node = int(changed.arg_node_indices[0])
    assert int(changed.x[node]) == vocab["FV0"]
    assert torch.equal(changed.edge_index, data.edge_index)
    assert torch.equal(changed.x, data.x)
    assert data.arg_count == 0


def test_two_highlighted_lemmas_are_two_arguments_not_six():
    data, vocab = _data("rw [<a>Demo.a</a>, <a>Demo.b</a>]")
    changed = relabel_arguments(data, node_vocab=vocab, name_to_id={"Demo.a": 10, "Demo.b": 11})
    assert changed.action_target_supported
    assert changed.arg_count == 2
    assert changed.arg_lemma_ids.tolist() == [10, 11]
    assert changed.citation_lemma_ids.tolist() == [10, 11]


def test_compound_terms_remain_unsupported_but_keep_external_citations():
    data, vocab = _data("simp [<a>Demo.a</a> (p.mpr hp)]")
    changed = relabel_arguments(data, node_vocab=vocab, name_to_id={"Demo.a": 10})
    assert not changed.action_target_supported
    assert changed.action_target_reason == "compound_or_nonreference_syntax"
    assert changed.arg_count == 0
    assert changed.citation_lemma_ids.tolist() == [10]
    assert extract_external_positive_ids(Batch.from_data_list([changed])) == [[10]]


def test_resolved_trace_supplies_canonical_alias_and_context_index():
    data, vocab = _data("rw [alias, h₁]")
    trace = {
        "syntax_args": [], "local_context": [{"context_index": 0, "user_name": "h₁"}],
        "source_syntax": {"tag": "node", "kind": "Lean.Parser.Tactic.rwSeq", "children": [
            {"tag": "identifier", "source": "alias", "semanticRole": "global", "name": "Demo.actual"},
            {"tag": "identifier", "source": "h₁", "semanticRole": "local", "contextIndex": 0},
        ]},
    }
    changed = relabel_arguments(data, node_vocab=vocab, name_to_id={"Demo.actual": 10}, trace=trace)
    assert changed.action_target_supported
    assert changed.arg_lemma_ids.tolist() == [10, -1]
    assert int(changed.x[changed.arg_node_indices[1]]) == vocab["FV0"]
    assert changed.citation_label_policy == "resolved_source_v3"


def test_resolved_context_index_survives_oov_display_name():
    data, vocab = _data("exact hα")
    data.x[data.x == vocab["h₁"]] = 0
    trace = {
        "syntax_args": [], "local_context": [{"context_index": 0, "user_name": "hα"}],
        "source_syntax": {"tag": "identifier", "source": "hα", "semanticRole": "local", "contextIndex": 0},
    }
    changed = relabel_arguments(data, node_vocab=vocab, name_to_id={}, trace=trace)
    assert changed.action_target_supported
    assert int(changed.x[changed.arg_node_indices[0]]) == vocab["FV0"]


def test_locations_and_over_budget_sequences_are_explicitly_unsupported():
    data, vocab = _data("rw [Demo.a] at h₁")
    changed = relabel_arguments(data, node_vocab=vocab, name_to_id={"Demo.a": 10})
    assert not changed.action_target_supported
    data.tactic_raw = "rw [Demo.a, Demo.a, Demo.a, Demo.a]"
    changed = relabel_arguments(data, node_vocab=vocab, name_to_id={"Demo.a": 10}, max_args=3)
    assert changed.action_target_reason == "over_decoder_budget"
    assert changed.citation_lemma_ids.tolist() == [10]


def test_unsupported_argument_rows_receive_no_stop_supervision():
    stop = [torch.zeros(2, requires_grad=True), torch.zeros(2, requires_grad=True)]
    loss, metrics = compute_combined_loss(
        torch.zeros(2, 4, requires_grad=True), [torch.zeros(2, 3, requires_grad=True)],
        torch.tensor([1, 1]), torch.tensor([[-1], [-1]]),
        torch.tensor([0, 0, 0, 1, 1, 1]), arg_count_per_sample=[0, 0],
        stop_logits_list=stop, argument_supervision_mask=torch.tensor([True, False]),
    )
    loss.backward()
    assert metrics["arg_sequence_count"] == 1
    assert all(float(logit.grad[1]) == 0.0 for logit in stop)
    assert all(float(logit.grad[0]) != 0.0 for logit in stop)


def test_relabel_cli_preserves_graphs_and_resume_checks_source_chunks(tmp_path):
    import faiss
    from maths_ai.gnn_inference.scripts.relabel_prepared_arguments import main

    source = tmp_path / "source"
    (source / "vocab").mkdir(parents=True)
    (source / "manifests").mkdir()
    data, vocab = _data("rw [<a>Demo.a</a>, <a>Demo.b</a>]")
    (source / "vocab/node_vocab.json").write_text(json.dumps(vocab))
    (source / "vocab/tactic_vocab.json").write_text(json.dumps({"<UNK_TACTIC>": 0, "rw": 1}))
    packed = source / "packed/bidirectional"
    split_info = {}
    for split in ("train", "val", "test"):
        (source / split / "pyg").mkdir(parents=True)
        (packed / split).mkdir(parents=True)
        example = data.clone()
        example.row_index = 7
        example.split = split
        example.theorem = "Demo.test"
        example.dataset_name = "demo"
        example.arg_count = 6
        example.arg_node_indices = torch.tensor([-1] * 6)
        example.arg_lemma_ids = torch.tensor([-1, 10, -1, -1, 11, -1])
        torch.save([example], packed / split / "chunk_000000.pt")
        (source / "manifests" / f"{split}.json").write_text(json.dumps({
            "success_count": 1, "artifact_paths": {"pyg_dir": f"{split}/pyg"},
        }))
        split_info[split] = {"count": 1, "chunks": ["chunk_000000.pt"]}
    (packed / "manifest.json").write_text(json.dumps({"splits": split_info, "edge_mode": "bidirectional"}))
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    matrix = np.eye(8, dtype=np.float32)[:2]
    faiss_index = faiss.IndexFlatIP(8)
    faiss_index.add(matrix)
    faiss.write_index(faiss_index, str(index_dir / "faiss.index"))
    np.save(index_dir / "lemma_vectors.npy", matrix)
    (index_dir / "lemma_ids.json").write_text("[10, 11]")
    (index_dir / "lemma_names.json").write_text('["Demo.a", "Demo.b"]')
    output = tmp_path / "repaired"
    argv = ["--source-root", str(source), "--output-root", str(output), "--index-path", str(index_dir)]
    main(argv)
    dataset = PreparedGraphDataset(load_prepared_metadata(output), split="val", cache_in_memory=True)
    repaired = dataset[0]
    assert repaired.arg_count == 2 and repaired.action_target_supported
    assert repaired.citation_lemma_ids.tolist() == [10, 11]
    assert torch.equal(repaired.x, data.x)
    assert torch.equal(repaired.edge_index, data.edge_index)
    assert torch.load(packed / "val/chunk_000000.pt", weights_only=False)[0].arg_count == 6
    main([*argv, "--resume"])
    changed = torch.load(packed / "val/chunk_000000.pt", weights_only=False)
    changed[0].tactic_raw = "rw [Demo.a]"
    torch.save(changed, packed / "val/chunk_000000.pt")
    with pytest.raises(ValueError, match="source/output chunk changed"):
        main([*argv, "--resume"])
