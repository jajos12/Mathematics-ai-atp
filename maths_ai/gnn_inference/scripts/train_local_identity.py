"""Isolate local-reference ranking with frozen normalized graph representations.

Train only the existing local pointer projection against all flat-supported
local positions. Use gold tactic and preceding arguments. Frozen text-index
vectors supply GRU context for preceding library references, even when that
lemma missed top-200 retrieval; the matched validation cohort is reported
separately and never receives an injected candidate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    JointSourceDecoder, gold_arguments_from_batch,
)
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidateSource
from maths_ai.gnn_inference.atp_lean_gnn.text_library_candidates import CachedTextLibraryCandidates
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    GraphBudgetBatchSampler, PointerConfig, PreparedGraphDataset,
    build_pointer_model, load_prepared_metadata,
)


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _local_rows(dataset) -> list[int]:
    return [index for index, graph in enumerate(dataset)
            if bool(graph.action_target_supported) and bool((graph.arg_node_indices >= 0).any())]


def _loader(dataset, indices, *, batch_size: int, seed: int, train: bool):
    sizes = dataset.graph_sizes()
    selected = Subset(dataset, indices)
    sampler = GraphBudgetBatchSampler(
        [sizes[index] for index in indices], max_graphs=batch_size,
        max_nodes=30000, max_edges=150000, oversize_policy="skip",
        shuffle=train, seed=seed,
    )
    return DataLoader(selected, batch_sampler=sampler, num_workers=0), sampler


def _metrics(totals: Counter, *, split: str):
    result = dict(totals)
    result["split"] = split
    for cohort in ("all", "text_pool_reachable"):
        denominator = totals[f"{cohort}_local_positions"]
        result[f"{cohort}_top1"] = totals[f"{cohort}_top1_hits"] / max(denominator, 1)
        result[f"{cohort}_top5"] = totals[f"{cohort}_top5_hits"] / max(denominator, 1)
    result["mean_local_loss"] = totals["loss_sum"] / max(totals["all_local_positions"], 1)
    return result


def _run_epoch(model, loader, sampler, index, *, device, epoch, split,
               candidate_table=None, optimizer=None, start_batch=0,
               initial_counts=None, on_progress=None):
    """Train local projection only; report all and exact V2-reachable cohorts."""
    model.eval()  # Frozen GNN/GRU dropout must stay off even during training.
    sampler.set_epoch(epoch)
    totals = Counter(initial_counts or {})
    training = optimizer is not None
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for batch_number, batch in enumerate(loader):
        if batch_number < start_batch:
            continue
        batch = batch.to(device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            nodes = model.pointer.backbone.encode_nodes(batch)
            states = model.pointer.backbone.readout(nodes, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        with torch.set_grad_enabled(training), torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            for row, gold in enumerate(gold_arguments_from_batch(batch)):
                assert bool(batch.action_target_supported[row]) and any(
                    argument.source is CandidateSource.LOCAL for argument in gold
                )
                start, end = int(batch.ptr[row]), int(batch.ptr[row + 1])
                local_nodes = batch.premise_mask[start:end].nonzero(as_tuple=False).view(-1)
                local_vectors = nodes[start:end].index_select(0, local_nodes)
                node_positions = {int(node): position for position, node in enumerate(local_nodes.tolist())}
                lemma_pool = None
                if candidate_table is not None:
                    row_index = int(batch.row_index[row]) - start  # PyG offsets keys containing 'index'.
                    row_position = int(np.searchsorted(candidate_table["rows"], row_index))
                    if (row_position >= len(candidate_table["rows"])
                            or int(candidate_table["rows"][row_position]) != row_index):
                        raise ValueError(f"candidate cache absent for val row {row_index}")
                    lemma_pool = set(int(value) for value in candidate_table["ids"][row_position])
                matched = lemma_pool is not None and all(
                    argument.source is CandidateSource.LOCAL or argument.stable_id in lemma_pool
                    for argument in gold
                )
                totals["rows_with_local"] += 1
                totals["text_pool_reachable_rows"] += matched
                hidden = model.initial_state(states[row], int(batch.y[row]))
                for argument in gold:
                    if argument.source is CandidateSource.LOCAL:
                        position = node_positions.get(argument.stable_id)
                        if position is None:
                            raise ValueError("corrected local target outside the selectable graph pool")
                        logits = model.pointer.argument_selector.score_candidates(
                            hidden, local_vectors,
                        ).view(-1)
                        target = torch.tensor([position], device=device)
                        losses.append(F.cross_entropy(logits.float().view(1, -1), target))
                        hit = int(logits.argmax().item() == position)
                        top5 = int(position in logits.topk(min(5, logits.numel())).indices.tolist())
                        totals["all_local_positions"] += 1
                        totals["all_top1_hits"] += hit
                        totals["all_top5_hits"] += top5
                        if matched:
                            totals["text_pool_reachable_local_positions"] += 1
                            totals["text_pool_reachable_top1_hits"] += hit
                            totals["text_pool_reachable_top5_hits"] += top5
                        next_vector = local_vectors[position].view(1, -1)
                    else:
                        position = index.id_to_position.get(argument.stable_id)
                        if position is None:
                            raise ValueError(f"corrected library lemma {argument.stable_id} absent from text index")
                        text_vector = torch.as_tensor(
                            index.lemma_vectors[position], dtype=states.dtype, device=device,
                        ).view(1, -1)
                        next_vector = model.lemma_adapter(text_vector)
                    hidden = model.pointer.argument_selector.gru(next_vector, hidden)
            if training and losses:
                loss = torch.stack(losses).mean()
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"nonfinite local identity loss in batch {batch_number}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.pointer.argument_selector.out_proj.parameters(), 1.0)
                optimizer.step()
                totals["loss_sum"] += float(sum(item.detach().item() for item in losses))
            elif losses:
                totals["loss_sum"] += float(sum(item.detach().item() for item in losses))
        totals["batches"] += 1
        if training and on_progress is not None and (batch_number + 1) % 50 == 0:
            on_progress(batch_number + 1, dict(totals))
        if (batch_number + 1) % 50 == 0 or batch_number + 1 == len(loader):
            print(f"{split} epoch {epoch}: {batch_number + 1}/{len(loader)} batches; "
                  f"local_positions={totals['all_local_positions']}", flush=True)
    return _metrics(totals, split=split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "joint-checkpoint", "text-checkpoint", "text-index",
                 "candidate-dir", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-val-local", type=int)
    parser.add_argument("--expected-val-reachable", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if any(value is not None and value < 1 for value in
           (args.expected_val_local, args.expected_val_reachable)):
        raise ValueError("expected validation counts must be positive")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    metadata = load_prepared_metadata(args.prepared_root, splits=("train", "val"))
    saved = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False)
    if saved.get("model_type") != "joint_source_flat_decoder_v2":
        raise ValueError("local experiment requires the trained V2 joint decoder")
    joint_binding = saved["binding"]
    if joint_binding["prepared_root"] != str(args.prepared_root.resolve()):
        raise ValueError("joint checkpoint and local prepared data differ")
    text_index = LemmaIndex.load(args.text_index)
    if joint_binding["text_index_manifest"] != text_index.manifest:
        raise ValueError("joint checkpoint and text-index vectors differ")
    if joint_binding["text_checkpoint_sha256"] != file_sha256(args.text_checkpoint):
        raise ValueError("joint checkpoint and text snapshot differ")
    candidate_provider = CachedTextLibraryCandidates(
        args.candidate_dir, text_index, checkpoint_path=args.text_checkpoint,
    )
    if joint_binding["candidate_manifest_sha256"] != file_sha256(args.candidate_dir / "manifest.json"):
        raise ValueError("joint checkpoint and validation candidate policy differ")
    config = PointerConfig.from_dict(saved["config"])
    pointer = build_pointer_model(metadata, config)
    model = JointSourceDecoder(pointer, text_dim=text_index.lemma_vectors.shape[1])
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.pointer.argument_selector.out_proj.requires_grad_(True)
    model = model.to(device).eval()
    optimizer = AdamW(model.pointer.argument_selector.out_proj.parameters(), lr=args.learning_rate)
    datasets = {
        split: PreparedGraphDataset(metadata, split=split, cache_in_memory=True)
        for split in ("train", "val")
    }
    candidate_provider.validate_dataset(datasets["val"], split="val")
    indices = {split: _local_rows(dataset) for split, dataset in datasets.items()}
    loaders = {}
    for split in ("train", "val"):
        loaders[split] = _loader(datasets[split], indices[split], batch_size=args.batch_size,
                                 seed=args.seed, train=split == "train")
    if not indices["train"] or not indices["val"] or loaders["val"][1].oversize_indices:
        raise ValueError("no local targets or oversized validation local graphs")
    missing_train_positions = sum(
        int((datasets["train"][indices["train"][position]].arg_node_indices >= 0).sum())
        for position in loaders["train"][1].oversize_indices
    )
    print(f"Local rows: train={len(indices['train'])}, val={len(indices['val'])}; "
          f"oversized train local positions excluded={missing_train_positions}", flush=True)
    binding = {
        "model_type": "frozen_graph_local_identity_v1",
        "prepared_root": str(args.prepared_root.resolve()),
        "joint_checkpoint_sha256": file_sha256(args.joint_checkpoint),
        "text_index_manifest": text_index.manifest,
        "text_checkpoint_sha256": file_sha256(args.text_checkpoint),
        "candidate_manifest_sha256": file_sha256(args.candidate_dir / "manifest.json"),
        "training_rows_with_local": len(indices["train"]),
        "validation_rows_with_local": len(indices["val"]),
        "excluded_train_local_positions": missing_train_positions,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "seed": args.seed, "frozen_components": "graph/tactic/GRU/text_adapter",
        "trained_components": "local_pointer_out_proj_only",
    }
    run_dir = args.run_dir
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"local scorer run exists: {run_dir}; pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    binding_path = run_dir / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("local scorer resume uses different data or training policy")
    _write_json(binding_path, binding)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    epoch, phase, offset, train_counts = 1, "ready", 0, None
    best = (-1.0, -1.0, -1.0)
    saved_best_metrics = None
    if args.resume and last_path.exists():
        previous = torch.load(last_path, map_location=device, weights_only=False)
        if previous["binding"] != binding:
            raise ValueError("saved local scorer uses different provenance")
        model.load_state_dict(previous["model_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        epoch, phase, offset = previous["epoch"], previous["phase"], previous["next_batch"]
        train_counts = previous.get("train_counts") if phase == "training" else None
        if phase in {"complete", "initial"}:
            epoch += 1
        if best_path.exists():
            metrics = torch.load(best_path, map_location="cpu", weights_only=False)["val_metrics"]
            saved_best_metrics = metrics
            best = (metrics["text_pool_reachable_top1"], metrics["all_top1"], metrics["text_pool_reachable_top5"])

    def score(metrics):
        return (metrics["text_pool_reachable_top1"], metrics["all_top1"],
                metrics["text_pool_reachable_top5"])

    def validate_cohort(metrics):
        for actual, expected, name in (
            (metrics["all_local_positions"], args.expected_val_local, "all val local positions"),
            (metrics.get("text_pool_reachable_local_positions", 0),
             args.expected_val_reachable, "matched reachable val local positions"),
        ):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} changed: {actual}, expected {expected}")

    def save(stage, *, next_batch=0, counts=None, metrics=None, checkpoint_epoch=None):
        payload = {
            "model_type": "frozen_graph_local_identity_v1", "binding": binding,
            "epoch": epoch if checkpoint_epoch is None else checkpoint_epoch,
            "phase": stage, "next_batch": next_batch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_counts": counts, "val_metrics": metrics,
        }
        temporary = last_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(last_path)
        return payload

    if saved_best_metrics is not None:
        validate_cohort(saved_best_metrics)
    if not last_path.exists():
        baseline_metrics = _run_epoch(
            model, *loaders["val"], text_index, device=device, epoch=0,
            split="val", candidate_table=candidate_provider.tables["val"],
        )
        validate_cohort(baseline_metrics)
        print(f"frozen V2 local baseline: {json.dumps(baseline_metrics)}", flush=True)
        _write_json(run_dir / "metrics_epoch_0.json", baseline_metrics)
        initial = save("initial", metrics=baseline_metrics, checkpoint_epoch=0)
        torch.save(initial, best_path)
        best = score(baseline_metrics)

    while epoch <= args.epochs:
        if phase != "train_done":
            train_metrics = _run_epoch(
                model, *loaders["train"], text_index, device=device, epoch=epoch,
                split="train", optimizer=optimizer, start_batch=offset,
                initial_counts=train_counts,
                on_progress=lambda next_batch, counts: save("training", next_batch=next_batch, counts=counts),
            )
            print(f"train epoch {epoch}: {json.dumps(train_metrics)}", flush=True)
            save("train_done", counts=dict(train_metrics))
        val_metrics = _run_epoch(
            model, *loaders["val"], text_index, device=device, epoch=epoch,
            split="val", candidate_table=candidate_provider.tables["val"],
        )
        validate_cohort(val_metrics)
        if val_metrics.get("text_pool_reachable_local_positions", 0) == 0:
            raise ValueError("no local validation targets in frozen text-pool cohort")
        print(f"val epoch {epoch}: {json.dumps(val_metrics)}", flush=True)
        _write_json(run_dir / f"metrics_epoch_{epoch}.json", val_metrics)
        payload = save("complete", metrics=val_metrics)
        val_score = score(val_metrics)
        if val_score > best:
            best = val_score
            torch.save(payload, best_path)
        epoch, phase, offset, train_counts = epoch + 1, "ready", 0, None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
