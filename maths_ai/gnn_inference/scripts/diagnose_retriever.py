"""Compare exact-index GNN, graph-label BM25 and optional state-text BM25.

Prepared PyG graphs do not store original proof-state text. In text mode,
recover it from validated extractor records or stream the original benchmark.
Both BM25 queries rank identical declaration name + type documents; neither
is a reproduction of ReProver's trained dense retriever.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import torch

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from maths_ai.gnn_inference.atp_lean_gnn.bundle import load_state_dict_checked
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME, iter_dataset_rows
from maths_ai.gnn_inference.atp_lean_gnn.graph import lemma_statement_to_dag
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import load_index_for_encoder
from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import (
    load_retriever_checkpoint,
    retrieval_metrics,
)
from maths_ai.gnn_inference.atp_lean_gnn.premise_retriever_training import (
    extract_external_positive_ids,
)
from maths_ai.gnn_inference.atp_lean_gnn.pyg import dag_to_pyg
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_diagnostics import (
    PremiseBM25,
    hit_overlap,
    hybrid_top_k,
    lexical_tokens,
    load_cached_state_text,
    reciprocal_rank_fusion,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    REQUIRED_POINTER_DATA_FIELDS,
    build_dataloaders,
    build_pointer_model,
    load_pointer_config,
    load_prepared_metadata,
    resolve_device,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "prepared-root", "checkpoint", "retriever-checkpoint", "index-path", "corpus-path", "output"):
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional first-N validation rows")
    parser.add_argument("--graph-samples", type=int, default=2048)
    parser.add_argument("--failure-examples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-source", choices=("none", "raw-cache", "upstream"), default="none",
        help="Compare original pre-tactic text against graph labels; raw-cache reads prepared/val/sexpr",
    )
    parser.add_argument(
        "--dataset-name", default=DATASET_NAME,
        help="Upstream benchmark to stream when --text-source upstream is selected",
    )
    return parser


def _oov_stats(graphs: list, *, unknown_id: int) -> dict[str, float | int]:
    nodes = sum(int(graph.x.numel()) for graph in graphs)
    unknown = sum(int((graph.x == unknown_id).sum()) for graph in graphs)
    return {
        "graphs": len(graphs), "nodes": nodes, "unknown_nodes": unknown,
        "unknown_fraction": unknown / max(nodes, 1),
        "graphs_with_unknown": sum(bool((graph.x == unknown_id).any()) for graph in graphs),
    }


def _diagnostic_config(config, prepared_root: str):
    # Graph-budget sampling needs the packed in-memory cache. Preserve both
    # settings to evaluate the same validation rows as retriever training.
    return replace(
        config,
        prepared_root=Path(prepared_root),
        training=replace(config.training, num_workers=0, pin_memory=False),
    ).normalized()


def _load_text_states(args, val_dataset) -> dict[int, str]:
    """Join original states to prepared examples by verified split/row identity."""
    if args.text_source == "none":
        return {}
    examples = {int(data.row_index): data for data in val_dataset}
    if len(examples) != len(val_dataset):
        raise ValueError("Validation prepared graphs contain repeated row_index values")
    if args.text_source == "raw-cache":
        return {
            row_index: load_cached_state_text(args.prepared_root, data)
            for row_index, data in examples.items()
        }
    states: dict[int, str] = {}
    for row in iter_dataset_rows(dataset_name=args.dataset_name, split="val"):
        data = examples.get(row.row_index)
        if data is None:
            continue
        if (row.theorem != str(data.theorem) or row.tactic != str(data.tactic_raw)
                or row.dataset_name != str(data.dataset_name)):
            raise ValueError(f"Upstream validation row {row.row_index} does not match prepared graph")
        if not row.state.strip():
            raise ValueError(f"Upstream validation row {row.row_index} has no state text")
        states[row.row_index] = row.state
        if len(states) == len(examples):
            break
    if len(states) != len(examples):
        raise ValueError(f"Upstream dataset has {len(states)}/{len(examples)} prepared validation rows")
    return states


def _original_row_index(batch, row: int) -> int:
    """Undo PyG's automatic node offset for attributes named '*index'."""
    # PyG treats `row_index` like a node index, adding the cumulative number
    # of preceding nodes. Prepared source row IDs are not graph node indices.
    return int(batch.row_index[row]) - int(batch.ptr[row])


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max-queries must be positive")
    if args.graph_samples < 1 or args.failure_examples < 0:
        raise ValueError("--graph-samples must be positive and --failure-examples nonnegative")
    device = resolve_device(args.device)
    config = load_pointer_config(args.config, device_override=args.device)
    config = _diagnostic_config(config, args.prepared_root)
    metadata = load_prepared_metadata(config.prepared_root)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pointer_state = saved.get("model_state_dict", saved)
    if any(str(key).startswith("module.") for key in pointer_state):
        pointer_state = {str(key).removeprefix("module."): value for key, value in pointer_state.items()}
    pointer = build_pointer_model(metadata, config).to(device)
    load_state_dict_checked(pointer, pointer_state)
    index = load_index_for_encoder(
        args.index_path, encoder_state_dict=pointer_state,
        node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        corpus_path=args.corpus_path, expected_edge_mode=config.edge_mode,
    )
    if not index.normalize_queries:
        raise ValueError("diagnostic requires normalized exact index")
    retriever = load_retriever_checkpoint(
        args.retriever_checkpoint, pointer_backbone=pointer.backbone,
        hidden_dim=config.model.hidden_dim, node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab, device=device,
    )
    del pointer, saved
    datasets, loaders = build_dataloaders(
        metadata, config, required_fields=REQUIRED_POINTER_DATA_FIELDS + ("arg_lemma_ids", "arg_count"),
    )
    text_states = _load_text_states(args, datasets["val"])
    records = load_lemma_corpus(args.corpus_path)
    indexed = [record for record in records if record.lemma_id in index.id_to_position]
    if len(indexed) != len(index.lemma_ids):
        raise ValueError("corpus and index record counts differ")
    print(f"Building lexical index over {len(indexed)} declarations...", flush=True)
    lexical = PremiseBM25(indexed)
    by_id = {record.lemma_id: record for record in indexed}
    rng = random.Random(args.seed)
    graph_records = rng.sample(indexed, min(args.graph_samples, len(indexed)))
    lemma_graphs = [dag_to_pyg(lemma_statement_to_dag(record.statement), metadata.node_vocab) for record in graph_records]
    unknown_id = metadata.node_vocab["<UNK>"]
    id_to_label = {value: label for label, value in metadata.node_vocab.items()}
    gnn_rows: list[list[int]] = []
    lexical_rows: list[list[int]] = []
    bm25_gold_ranks: list[int | None] = []
    text_rows: list[list[int]] = []
    text_gold_ranks: list[int | None] = []
    text_without_tokens = 0
    positive_rows: list[list[int]] = []
    failures: list[dict[str, object]] = []
    failure_count = 0
    state_nodes = state_unknown = labeled_nodes = labeled_unknown = 0
    row_count = 0
    next_progress = 500
    for batch in loaders["val"]:
        with torch.no_grad():
            queries = retriever.encode_states(batch.to(device))
            ranked, _, _ = index.search(queries, k=200)
        positives = extract_external_positive_ids(batch, name_to_id=index.name_to_id)
        for row, (gnn_ids, gold_ids) in enumerate(zip(ranked, positives)):
            if args.max_queries is not None and row_count >= args.max_queries:
                break
            source_row_index = _original_row_index(batch, row)
            node_ids = batch.x[batch.ptr[row]:batch.ptr[row + 1]].tolist()
            unknown_count = node_ids.count(unknown_id)
            state_nodes += len(node_ids)
            state_unknown += unknown_count
            if gold_ids:
                labeled_nodes += len(node_ids)
                labeled_unknown += unknown_count
            tokens = set()
            for node_id in node_ids:
                if node_id != unknown_id:
                    tokens.update(lexical_tokens(id_to_label[int(node_id)]))
            bm25_ids, gold_rank = lexical.search_with_gold_rank(
                tokens, gold_ids, k=200
            )
            if args.text_source != "none":
                text_tokens = set(lexical_tokens(text_states[source_row_index]))
                text_without_tokens += int(not text_tokens)
                text_ids, text_rank = lexical.search_with_gold_rank(
                    text_tokens, gold_ids, k=200
                )
                text_rows.append(text_ids)
                if gold_ids:
                    text_gold_ranks.append(text_rank)
            gnn_rows.append(gnn_ids)
            lexical_rows.append(bm25_ids)
            if gold_ids:
                bm25_gold_ranks.append(gold_rank)
            positive_rows.append(gold_ids)
            row_count += 1
            if gold_ids and not set(gold_ids).intersection(gnn_ids):
                failure_count += 1
                if not args.failure_examples:
                    continue
                # Reservoir sample: first 20 validation misses can be highly
                # correlated, especially when sorted by source theorem.
                slot = failure_count - 1
                if slot >= args.failure_examples:
                    slot = rng.randrange(failure_count)
                if slot >= args.failure_examples:
                    continue
                example = {
                    "row_index": source_row_index,
                    "theorem": str(batch.theorem[row]),
                    "tactic": str(batch.tactic_raw[row]),
                    "query_tokens": sorted(tokens)[:80],
                    "gold": [{"id": lemma_id, "name": by_id[lemma_id].name,
                              "type": by_id[lemma_id].statement[:250],
                              "gnn_rank_top_200": gnn_ids.index(lemma_id) + 1 if lemma_id in gnn_ids else None,
                              "bm25_rank_top_200": bm25_ids.index(lemma_id) + 1 if lemma_id in bm25_ids else None,
                              "text_bm25_rank_top_200": (
                                  text_ids.index(lemma_id) + 1 if lemma_id in text_ids else None
                              ) if args.text_source != "none" else None}
                             for lemma_id in gold_ids if lemma_id in by_id],
                    "gnn_top_5": [by_id[lemma_id].name for lemma_id in gnn_ids[:5] if lemma_id in by_id],
                    "bm25_top_5": [by_id[lemma_id].name for lemma_id in bm25_ids[:5] if lemma_id in by_id],
                    "text_bm25_top_5": [by_id[lemma_id].name for lemma_id in text_ids[:5] if lemma_id in by_id]
                    if args.text_source != "none" else None,
                }
                if slot == len(failures):
                    failures.append(example)
                else:
                    failures[slot] = example
        if row_count >= next_progress:
            print(f"Evaluated {row_count} validation rows...", flush=True)
            next_progress += 500
        if args.max_queries is not None and row_count >= args.max_queries:
            break
    gold_ids = list({lemma_id for row in positive_rows for lemma_id in row if lemma_id in by_id})
    sampled_gold = rng.sample(gold_ids, min(args.graph_samples, len(gold_ids)))
    gold_graphs = [
        dag_to_pyg(lemma_statement_to_dag(by_id[lemma_id].statement), metadata.node_vocab)
        for lemma_id in sampled_gold
    ]
    hybrid_150_50 = [
        hybrid_top_k(bm25, gnn, k=200, bm25_quota=150)
        for bm25, gnn in zip(lexical_rows, gnn_rows)
    ]
    hybrid_180_20 = [
        hybrid_top_k(bm25, gnn, k=200, bm25_quota=180)
        for bm25, gnn in zip(lexical_rows, gnn_rows)
    ]
    fused = [
        reciprocal_rank_fusion(bm25, gnn, k=200)
        for bm25, gnn in zip(lexical_rows, gnn_rows)
    ]
    union = [
        list(dict.fromkeys([*bm25, *gnn]))
        for bm25, gnn in zip(lexical_rows, gnn_rows)
    ]
    bm25_recall_curve = {
        f"recall_at_{k}": sum(
            rank is not None and rank <= k for rank in bm25_gold_ranks
        ) / max(len(bm25_gold_ranks), 1)
        for k in (1, 10, 50, 200, 1000, 5000)
    }
    report = {
        "config": {"index_path": args.index_path, "corpus_path": args.corpus_path,
                   "retriever_checkpoint": args.retriever_checkpoint, "seed": args.seed,
                   "max_queries": args.max_queries,
                   "text_source": args.text_source,
                   "text_dataset": args.dataset_name if args.text_source == "upstream" else None,
                   "bm25_query": "unique graph node-label tokens (no original state text stored)",
                   "text_bm25_query": (
                       "original pre-tactic proof-state text"
                       if args.text_source != "none" else None
                   ),
                   "bm25_documents": "declaration name + type, full exact-index corpus",
                   "bm25_k1": 1.2, "bm25_b": 0.75},
        "lemma_graph_oov_sample": _oov_stats(lemma_graphs, unknown_id=unknown_id),
        "gold_lemma_graph_oov_sample": _oov_stats(gold_graphs, unknown_id=unknown_id),
        "state_graph_oov": {"nodes": state_nodes, "unknown_nodes": state_unknown,
                            "unknown_fraction": state_unknown / max(state_nodes, 1)},
        "labeled_state_graph_oov": {"nodes": labeled_nodes, "unknown_nodes": labeled_unknown,
                                    "unknown_fraction": labeled_unknown / max(labeled_nodes, 1)},
        "gnn": retrieval_metrics(gnn_rows, positive_rows).as_dict(),
        "bm25": retrieval_metrics(lexical_rows, positive_rows).as_dict(),
        "bm25_gold_rank_curve": {
            "labeled_query_count": len(bm25_gold_ranks),
            **bm25_recall_curve,
            "gold_without_lexical_overlap": bm25_gold_ranks.count(None),
        },
        **({
            "text_bm25": retrieval_metrics(text_rows, positive_rows).as_dict(),
            "text_bm25_gold_rank_curve": {
                "labeled_query_count": len(text_gold_ranks),
                **{
                    f"recall_at_{k}": sum(
                        rank is not None and rank <= k for rank in text_gold_ranks
                    ) / max(len(text_gold_ranks), 1)
                    for k in (1, 10, 50, 200, 1000, 5000)
                },
                "gold_without_lexical_overlap": text_gold_ranks.count(None),
                "state_without_lexical_tokens": text_without_tokens,
            },
            "text_vs_graph_bm25_top_200_hit_overlap": hit_overlap(
                text_rows, lexical_rows, positive_rows
            ),
        } if args.text_source != "none" else {}),
        "hybrid_150_bm25_50_gnn": retrieval_metrics(hybrid_150_50, positive_rows).as_dict(),
        "hybrid_180_bm25_20_gnn": retrieval_metrics(hybrid_180_20, positive_rows).as_dict(),
        "hybrid_rrf_200": retrieval_metrics(fused, positive_rows).as_dict(),
        "union_up_to_400_upper_bound": retrieval_metrics(
            union, positive_rows, ks=(1, 10, 50, 200, 400)
        ).as_dict(),
        "top_200_labeled_hit_overlap": hit_overlap(lexical_rows, gnn_rows, positive_rows),
        "gnn_top_200_failure_count": failure_count,
        "gnn_top_200_failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_keys = (
            "lemma_graph_oov_sample", "gold_lemma_graph_oov_sample",
            "state_graph_oov", "gnn", "bm25", "bm25_gold_rank_curve",
            "hybrid_150_bm25_50_gnn",
            "hybrid_180_bm25_20_gnn", "hybrid_rrf_200",
            "union_up_to_400_upper_bound", "top_200_labeled_hit_overlap",
            "gnn_top_200_failure_count",
    )
    if args.text_source != "none":
        summary_keys += (
            "text_bm25", "text_bm25_gold_rank_curve",
            "text_vs_graph_bm25_top_200_hit_overlap",
        )
    print(f"Results saved to {output}\n" + json.dumps({
        key: report[key] for key in summary_keys
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
