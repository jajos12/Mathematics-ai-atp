"""Train flat local/library/stop decisions over frozen text-retrieved candidates.

The normalized baseline initializes the GNN. Training never substitutes old
text-parser GNN lemma vectors; library GRU inputs are adapted frozen text keys.
Unsupported Lean syntax and absent retrieved gold remain explicit misses.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    JointSourceDecoder, candidate_groups_from_batch, gold_arguments_from_batch,
)
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import load_frozen_text_retriever
from maths_ai.gnn_inference.atp_lean_gnn.text_library_candidates import CachedTextLibraryCandidates
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    REQUIRED_POINTER_DATA_FIELDS, build_dataloaders, build_pointer_model,
    initialize_pointer_from_baseline_checkpoint, load_pointer_config,
    load_prepared_metadata,
)


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _metric_counts(totals: Counter, *, split: str) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = dict(totals)
    for key in ("reachable_rows", "reachable_nonempty_rows", "missing_library_rows",
                "missing_source_supervised_rows", "reachable_positions"):
        result.setdefault(key, 0)
    def rate(numerator, denominator):
        return totals[numerator] / max(totals[denominator], 1)

    for name, numerator, denominator in (
        ("tactic_top1", "tactic_hits", "known_tactics"),
        ("retrieval_available_rate", "reachable_rows", "supported_rows"),
        ("nonempty_retrieval_available_rate", "reachable_nonempty_rows", "supported_nonempty_rows"),
        ("argument_count_accuracy", "count_hits", "supported_rows"),
        ("source_top1", "source_hits", "reachable_positions"),
        ("source_top1_both_available", "both_source_hits", "both_source_positions"),
        ("local_source_recall", "local_source_hits", "reachable_local_positions"),
        ("library_source_recall", "library_source_hits", "reachable_library_positions"),
        ("stop_accuracy_teacher_forced", "stop_hits", "stop_decisions"),
        ("local_top1_given_local", "local_hits", "reachable_local_positions"),
        ("library_top1_given_library", "library_hits", "reachable_library_positions"),
        ("frozen_text_top1_given_library", "frozen_text_hits", "reachable_library_positions"),
        ("joint_argument_top1", "joint_hits", "reachable_positions"),
    ):
        result[name] = rate(numerator, denominator)
    recalls = []
    for count in range(4):
        target = totals[f"gold_count_{count}"]
        if target:
            recall = totals[f"count_hit_{count}"] / target
            recalls.append(recall)
            result[f"gold_count_{count}_recall"] = recall
    result["count_macro_recall"] = sum(recalls) / max(len(recalls), 1)
    if split == "val":
        for name, numerator, denominator in (
            ("supported_action_exact", "predicted_action_exact", "supported_rows"),
            ("oracle_tactic_action_exact", "oracle_action_exact", "supported_rows"),
            ("nonempty_oracle_action_exact", "oracle_nonempty_exact", "supported_nonempty_rows"),
        ):
            result[name] = rate(numerator, denominator)
    result["split"] = split
    result["mean_reachable_loss"] = totals["loss_sum"] / max(totals["reachable_rows"], 1)
    result["mean_argument_count_loss"] = totals["count_loss_sum"] / max(totals["supported_rows"], 1)
    return result


def _supervision_weights(dataset, *, max_args: int):
    """Count all supported labels, not only text-reachable training states."""
    lengths = Counter()
    sources = Counter()
    for graph in dataset:
        if not bool(graph.action_target_supported):
            continue
        length = int(graph.arg_count)
        if not 0 <= length <= max_args:
            raise ValueError("supported action length exceeds decoder budget")
        lengths[length] += 1
        sources["local"] += int((graph.arg_node_indices >= 0).sum())
        sources["library"] += int((graph.arg_lemma_ids >= 0).sum())
    if not lengths or not sources["local"] or not sources["library"]:
        raise ValueError("joint training requires supported local and library targets")
    weights = [
        math.sqrt(sum(lengths.values()) / ((max_args + 1) * lengths[count]))
        if lengths[count] else 1.0
        for count in range(max_args + 1)
    ]
    local_weight = min(3.0, math.sqrt(sources["library"] / sources["local"]))
    return weights, (local_weight, 1.0), {"lengths": dict(lengths), "sources": dict(sources)}


def _run_epoch(model, loader, provider, *, split, device, k, optimizer=None,
               start_batch=0, max_batches=0, on_progress=None, initial_counts=None,
               count_weights=None, source_weights=(1.0, 1.0)):
    training = optimizer is not None
    model.train(training)
    model.pointer.backbone.eval()  # Freeze graph and tactic baseline in pilot.
    totals = Counter(initial_counts or {})
    weights = torch.as_tensor(
        [1.0] * (model.pointer.max_args + 1) if count_weights is None else count_weights,
        dtype=torch.float32, device=device,
    )
    if weights.numel() != model.pointer.max_args + 1:
        raise ValueError("argument-count weights must match decoder classes")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        if batch_index < start_batch:
            continue
        batch = batch.to(device)
        lemma_ids, lemma_vecs, lemma_scores = provider.search(batch, k=k)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            nodes = model.pointer.backbone.encode_nodes(batch)
            states = model.pointer.backbone.readout(nodes, batch)
            tactic_logits = model.pointer.backbone.classifier(states)
        candidates = candidate_groups_from_batch(batch, nodes, lemma_ids, lemma_vecs, lemma_scores)
        gold_rows = gold_arguments_from_batch(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        with torch.set_grad_enabled(training), torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            for row, (pool, gold) in enumerate(zip(candidates, gold_rows)):
                tactic_id = int(batch.y[row].item())
                if tactic_id != 0:  # unknown tactic ID is validated below by caller.
                    totals["known_tactics"] += 1
                    totals["tactic_hits"] += int(tactic_logits[row].argmax().item() == tactic_id)
                totals["states"] += 1
                if not bool(batch.action_target_supported[row]):
                    continue
                totals["supported_rows"] += 1
                totals["supported_nonempty_rows"] += int(bool(gold))
                length = len(gold)
                count_logits = model.count_logits(states[row], tactic_id)
                count_loss = weights[length] * torch.nn.functional.cross_entropy(
                    count_logits.float().view(1, -1),
                    torch.tensor([length], device=device),
                )
                losses.append(count_loss)
                totals["count_loss_sum"] += float(count_loss.detach().item())
                totals[f"gold_count_{length}"] += 1
                count_hit = int(count_logits.argmax().item() == length)
                totals[f"count_hit_{length}"] += count_hit
                totals["count_hits"] += count_hit
                result = model.supervised_row(states[row], tactic_id, pool, gold,
                                              source_weights=source_weights)
                if result is None:
                    totals["missing_library_rows"] += 1
                    prefix_loss = model.supervised_source_prefix(
                        states[row], tactic_id, pool, gold, source_weights=source_weights,
                    )
                    if prefix_loss is not None:
                        losses.append(prefix_loss)
                        totals["missing_source_supervised_rows"] += 1
                else:
                    loss, metrics = result
                    losses.append(loss)
                    totals["reachable_rows"] += 1
                    totals["reachable_nonempty_rows"] += int(bool(gold))
                    totals["reachable_positions"] += metrics["positions"]
                    totals["reachable_local_positions"] += metrics["local_positions"]
                    totals["reachable_library_positions"] += metrics["library_positions"]
                    totals["source_hits"] += metrics["source_top1"]
                    totals["local_source_hits"] += metrics["local_source_top1"]
                    totals["library_source_hits"] += metrics["library_source_top1"]
                    totals["both_source_hits"] += metrics["both_source_hits"]
                    totals["both_source_positions"] += metrics["both_source_positions"]
                    totals["stop_hits"] += metrics["stop_hits"]
                    totals["stop_decisions"] += metrics["stop_decisions"]
                    totals["local_hits"] += metrics["local_top1"]
                    totals["library_hits"] += metrics["library_top1"]
                    totals["frozen_text_hits"] += metrics["frozen_text_top1"]
                    totals["joint_hits"] += metrics["joint_top1"]
                    totals["loss_sum"] += float(loss.detach().item())
                if not training:
                    predicted_tactic = int(tactic_logits[row].argmax())
                    predicted, stopped = model.decode(states[row], predicted_tactic, pool)
                    totals["predicted_action_exact"] += int(
                        stopped and predicted_tactic == tactic_id and list(predicted) == gold
                    )
                    oracle, stopped = model.decode(states[row], tactic_id, pool)
                    correct = int(stopped and list(oracle) == gold)
                    totals["oracle_action_exact"] += correct
                    totals["oracle_nonempty_exact"] += correct * int(bool(gold))
            if training and losses:
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite joint loss in batch {batch_index}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad], 1.0,
                )
                optimizer.step()
        totals["batches"] += 1
        if training and on_progress is not None and (batch_index + 1) % 100 == 0:
            on_progress(batch_index + 1, dict(totals))
            print(f"trained {batch_index + 1}/{len(loader)} batches; reachable={totals['reachable_rows']}", flush=True)
        elif not training and (batch_index + 1) % 25 == 0:
            print(f"validated {batch_index + 1}/{len(loader)} batches", flush=True)
    return _metric_counts(totals, split=split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "candidate-dir", "text-checkpoint", "text-index",
                 "corpus-path", "baseline-checkpoint", "pointer-config", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.k) < 1 or args.max_train_batches < 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, K, and learning rate must be valid")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    metadata = load_prepared_metadata(args.prepared_root)
    config = load_pointer_config(
        args.pointer_config, prepared_root_override=args.prepared_root,
        device_override=args.device,
        training_overrides={
            "batch_size": args.batch_size, "max_batch_nodes": 30000,
            "max_batch_edges": 150000, "oversize_graph_policy": "skip",
            "cache_in_memory": True, "num_workers": 0,
        },
    )
    config = replace(
        config, initialization_checkpoint=args.baseline_checkpoint,
        training=replace(config.training, epochs=args.epochs),
    ).normalized()
    text_model, _, text_index, saved = load_frozen_text_retriever(
        args.text_checkpoint, args.text_index, args.corpus_path, device=torch.device("cpu"),
    )
    del text_model  # Text encoder is frozen; its shared index supplies keys.
    provider = CachedTextLibraryCandidates(
        args.candidate_dir, text_index, checkpoint_path=args.text_checkpoint,
    )
    manifest = provider.manifest
    for field, path in (
        ("argument_label_binding_sha256", args.prepared_root / "argument_labels.json"),
        ("prepared_packed_manifest_sha256", args.prepared_root / "packed/bidirectional/manifest.json"),
        ("node_vocab_sha256", args.prepared_root / "vocab/node_vocab.json"),
        ("tactic_vocab_sha256", args.prepared_root / "vocab/tactic_vocab.json"),
    ):
        if manifest.get(field) != file_sha256(path):
            raise ValueError(f"candidate source binding changed: {field}")
    if (manifest["prepared_root"] != str(args.prepared_root.resolve())
            or manifest["k"] < args.k or saved["epoch"] != manifest["text_epoch"]
            or manifest["corpus_sha256"] != saved["corpus_sha256"]
            or set(provider.tables) != {"train", "val"}):
        raise ValueError("candidate pool does not match corrected source, text epoch, or K")
    datasets, loaders = build_dataloaders(metadata, config, required_fields=REQUIRED_POINTER_DATA_FIELDS)
    for split in ("train", "val"):
        provider.validate_dataset(datasets[split], split=split)
    if any(getattr(loaders["val"].batch_sampler, "oversize_indices", [])):
        raise ValueError("validation graph budget excludes source rows; enlarge budget")
    count_weights, source_weights, label_stats = _supervision_weights(
        datasets["train"], max_args=config.max_args,
    )
    print(f"Training labels: {json.dumps(label_stats)}; count weights={count_weights}; "
          f"source weights={source_weights}", flush=True)
    pointer = build_pointer_model(metadata, config)
    initialization = initialize_pointer_from_baseline_checkpoint(
        pointer, config=config, metadata=metadata, checkpoint_path=args.baseline_checkpoint,
    )
    pointer.backbone.requires_grad_(False)
    model = JointSourceDecoder(pointer, text_dim=text_index.lemma_vectors.shape[1]).to(device)
    optimizer = AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                      lr=args.learning_rate)
    config_policy = config.to_dict()
    config_policy["training"].pop("epochs")  # A resumed run may extend total epochs.
    binding = {
        "model_type": "joint_source_flat_decoder_v2",
        "candidate_manifest_sha256": file_sha256(args.candidate_dir / "manifest.json"),
        "baseline_checkpoint_sha256": initialization["source_checkpoint_sha256"],
        "text_checkpoint_sha256": file_sha256(args.text_checkpoint),
        "prepared_root": str(args.prepared_root.resolve()),
        "text_index_manifest": text_index.manifest,
        "config": config_policy, "k": args.k,
        "batch_size": args.batch_size, "max_train_batches": args.max_train_batches,
        "learning_rate": args.learning_rate, "seed": args.seed,
        "frozen_backbone": True, "library_ranking": "text_prior_with_contextual_residual",
        "count_policy": "explicit_arity_sqrt_inverse_frequency_v1",
        "count_weights": count_weights, "source_weights": list(source_weights),
    }
    root = args.run_dir
    if root.exists() and not args.resume:
        raise FileExistsError(f"joint run exists: {root}; pass --resume")
    root.mkdir(parents=True, exist_ok=True)
    binding_path = root / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("joint decoder resume uses different data or training policy")
    _atomic_json(binding_path, binding)
    epoch, phase, offset = 1, "ready", 0
    progress_counts = None
    completed_train_metrics = None
    best = (-1.0, -1.0, float("-inf"))
    last_path = root / "last.pt"
    if args.resume and last_path.exists():
        saved_state = torch.load(last_path, map_location=device, weights_only=False)
        if saved_state["binding"] != binding:
            raise ValueError("joint decoder checkpoint has different provenance")
        model.load_state_dict(saved_state["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved_state["optimizer_state_dict"])
        epoch, phase, offset = saved_state["epoch"], saved_state["phase"], saved_state["next_batch"]
        progress_counts = saved_state.get("train_counts") if phase == "training" else None
        completed_train_metrics = saved_state.get("train_metrics") if phase == "train_done" else None
        if phase == "complete":
            epoch += 1
        best_path = root / "best.pt"
        if best_path.exists():
            best_metrics = torch.load(best_path, map_location="cpu", weights_only=False)["val_metrics"]
            best = (best_metrics["nonempty_oracle_action_exact"],
                    best_metrics["count_macro_recall"], -best_metrics["mean_reachable_loss"])

    def save(stage, *, next_batch=0, metrics=None, train_metrics=None, train_counts=None):
        payload = {
            "model_type": "joint_source_flat_decoder_v2", "binding": binding,
            "epoch": epoch, "phase": stage, "next_batch": next_batch,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": metrics, "train_metrics": train_metrics, "train_counts": train_counts,
            "config": config.to_dict(),
        }
        temporary = last_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(last_path)
        return payload

    while epoch <= args.epochs:
        if phase != "train_done":
            sampler = getattr(loaders["train"], "batch_sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_metrics = _run_epoch(
                model, loaders["train"], provider, split="train", device=device, k=args.k,
                optimizer=optimizer, start_batch=offset, max_batches=args.max_train_batches,
                initial_counts=progress_counts,
                count_weights=count_weights, source_weights=source_weights,
                on_progress=lambda next_batch, counts: save("training", next_batch=next_batch, train_counts=counts),
            )
            print(f"epoch {epoch} train: {json.dumps(train_metrics)}", flush=True)
            save("train_done", train_metrics=train_metrics)
        else:
            train_metrics = completed_train_metrics
        with torch.no_grad():
            val_metrics = _run_epoch(model, loaders["val"], provider,
                                     split="val", device=device, k=args.k,
                                     count_weights=count_weights, source_weights=source_weights)
        if not val_metrics.get("reachable_nonempty_rows", 0):
            raise ValueError("no nonempty validation actions reachable in the frozen candidate pool")
        print(f"epoch {epoch} val: {json.dumps(val_metrics)}", flush=True)
        _atomic_json(root / f"metrics_epoch_{epoch}.json", {"train": train_metrics, "val": val_metrics})
        payload = save("complete", metrics=val_metrics, train_metrics=train_metrics)
        score = (val_metrics["nonempty_oracle_action_exact"],
                 val_metrics["count_macro_recall"], -val_metrics["mean_reachable_loss"])
        if score > best:
            best = score
            torch.save(payload, root / "best.pt")
        epoch, phase, offset, progress_counts, completed_train_metrics = epoch + 1, "ready", 0, None, None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
