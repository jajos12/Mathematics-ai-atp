"""Fit text-score prior strength on train; report oracle-prefix ranks on val.

The joint pointer, backbone, candidate lists and text encoder are frozen. This
is an argument-ranking diagnostic, not complete-action or Lean proof success.
No ground-truth lemma name enters the retrieval query or the score adjustment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from maths_ai.gnn_inference.atp_lean_gnn.bundle import load_state_dict_checked
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidateSource, build_unified_pools
from maths_ai.gnn_inference.atp_lean_gnn.premise_training import (
    _extract_arg_lemma_ids, _extract_arg_targets, _recover_lemma_targets,
)
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import CachedTextCandidates
from maths_ai.gnn_inference.atp_lean_gnn.text_prior_reranking import blend_library_scores, rank_of
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PointerConfig, PreparedGraphDataset, build_pointer_model,
    load_prepared_metadata, resolve_device,
)
from maths_ai.gnn_inference.atp_lean_gnn.unified_reranking import (
    resolve_ordered_pool_targets, verify_unified_reranker_dependencies,
)


def _collect(model, dataset, provider, index, device, *, k, max_library_positions=None, seed=42):
    candidates = [i for i, row in enumerate(dataset)
                  if bool(row.action_target_supported) and int(row.arg_count) > 0]
    random.Random(seed).shuffle(candidates)
    samples = []
    library_count = 0
    with torch.no_grad():
        for item in candidates:
            data = dataset[item]
            batch = Batch.from_data_list([data]).to(device)
            local = _extract_arg_targets(batch, model.max_args, device)
            library = _extract_arg_lemma_ids(batch, model.max_args, device)
            library = _recover_lemma_targets(batch, local, library, index)
            cached = provider.search(batch, k=k)
            nodes = model.backbone.encode_nodes(batch)
            state = model.backbone.readout(nodes, batch)
            pool = build_unified_pools(
                state, nodes, batch.premise_mask, batch.batch,
                library_candidates=cached,
            )[0]
            positions, _ = resolve_ordered_pool_targets(
                [pool], local, library, [int(data.arg_count)], max_args=model.max_args,
            )
            decoder = model.argument_selector.initial_state(
                state, model.tactic_embedding(batch.y.view(-1)),
            )
            for position in positions[0]:
                if position < 0:
                    break
                pointer = model.argument_selector.score_candidates(
                    decoder, pool.candidate_vectors
                ).squeeze(0).float().cpu().numpy()
                vectors = [(reference.source.value, reference.retrieval_score)
                           for reference in pool.candidates]
                samples.append((pointer, vectors, position))
                if pool.candidates[position].source is CandidateSource.LIBRARY:
                    library_count += 1
                decoder = model.argument_selector.gru(
                    pool.candidate_vectors[position].unsqueeze(0), decoder,
                )
            if max_library_positions is not None and library_count >= max_library_positions:
                break
    return samples


def _metrics(samples, alpha):
    result = {"library_positions": 0, "local_positions": 0,
              "library_only_top1": 0, "library_only_top5": 0,
              "mixed_library_top1": 0, "mixed_library_top5": 0,
              "mixed_local_top1": 0, "mixed_local_top5": 0,
              "correct_library_source": 0, "correct_local_source": 0,
              "text_only_library_top1": 0, "text_only_library_top5": 0}
    for pointer, vectors, gold in samples:
        from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidateRef
        references = [CandidateRef(source=CandidateSource.parse(source), stable_id=i, retrieval_score=score)
                      for i, (source, score) in enumerate(vectors)]
        scores = blend_library_scores(pointer, references, alpha=alpha)
        source = references[gold].source
        rank = rank_of(scores, gold)
        top_source = references[int(np.argmax(scores))].source
        if source is CandidateSource.LIBRARY:
            result["library_positions"] += 1
            library = [i for i, ref in enumerate(references) if ref.source is CandidateSource.LIBRARY]
            lib_rank = rank_of(scores, gold, eligible=library)
            text_values = np.asarray([
                ref.retrieval_score if ref.source is CandidateSource.LIBRARY else 0.0
                for ref in references
            ], dtype=np.float32)
            text_rank = rank_of(text_values, gold, eligible=library)
            result["library_only_top1"] += int(lib_rank == 1)
            result["library_only_top5"] += int(lib_rank <= 5)
            result["text_only_library_top1"] += int(text_rank == 1)
            result["text_only_library_top5"] += int(text_rank <= 5)
            result["mixed_library_top1"] += int(rank == 1)
            result["mixed_library_top5"] += int(rank <= 5)
            result["correct_library_source"] += int(top_source is source)
        else:
            result["local_positions"] += 1
            result["mixed_local_top1"] += int(rank == 1)
            result["mixed_local_top5"] += int(rank <= 5)
            result["correct_local_source"] += int(top_source is source)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--pointer-index", required=True, type=Path)
    parser.add_argument("--scorer-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-library-positions", type=int, default=1500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.train_library_positions < 1:
        raise ValueError("--train-library-positions must be positive")
    device = resolve_device(args.device)
    saved = torch.load(args.scorer_checkpoint, map_location="cpu", weights_only=False)
    if saved.get("phase") != "complete":
        raise ValueError("scorer checkpoint must have completed validation")
    config = PointerConfig.from_dict(saved["config"]).normalized()
    metadata = load_prepared_metadata(args.prepared_root)
    if metadata.node_vocab != saved["node_vocab"] or metadata.tactic_vocab != saved["tactic_vocab"]:
        raise ValueError("scorer and prepared vocabularies differ")
    model = build_pointer_model(metadata, config).to(device)
    load_state_dict_checked(model, saved["model_state_dict"])
    model.eval()
    index = LemmaIndex.load(args.pointer_index)
    provider = CachedTextCandidates(args.candidate_dir, index)
    verify_unified_reranker_dependencies(
        saved, index_manifest=index.manifest, retriever=None, candidate_provider=provider,
    )
    datasets = {
        split: PreparedGraphDataset(metadata, split=split, edge_mode=config.edge_mode, cache_in_memory=True)
        for split in ("train", "val")
    }
    for split, dataset in datasets.items():
        provider.validate_dataset(dataset, split=split)
    train = _collect(
        model, datasets["train"], provider, index, device, k=200,
        max_library_positions=args.train_library_positions, seed=args.seed,
    )
    validation = _collect(model, datasets["val"], provider, index, device, k=200, seed=args.seed)
    if not any(vectors[gold][0] == "library" for _, vectors, gold in train):
        raise ValueError("no available library labels in sampled training rows")
    alphas = (0, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
    train_scores = {alpha: _metrics(train, alpha) for alpha in alphas}
    selected = max(alphas, key=lambda alpha: (
        train_scores[alpha]["library_only_top1"] / train_scores[alpha]["library_positions"],
        -alpha,
    ))
    report = {
        "selected_alpha_from_train": selected,
        "train": {str(alpha): result for alpha, result in train_scores.items()},
        "validation": {
            "pointer_only": _metrics(validation, 0),
            "train_selected_alpha": _metrics(validation, selected),
        },
        "interpretation": "oracle tactic/prefix; available targets only; no complete-action or proof result",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected_alpha_from_train": selected,
                      "validation": report["validation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
