"""Train an independent library query inside the existing joint decoder.

Freeze graph, local query, tactic/count/source heads, GRU and text adapter.
Never inject missing gold into top-k candidates. Full-index gold vectors may
provide teacher-forced prefix context, not extra selectable identities.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.joint_source_decoder import (
    JointSourceDecoder, candidate_groups_from_batch, gold_arguments_from_batch,
)
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
from maths_ai.gnn_inference.atp_lean_gnn.premise_pool import CandidateSource
from maths_ai.gnn_inference.atp_lean_gnn.text_library_candidates import CachedTextLibraryCandidates
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PointerConfig, PreparedGraphDataset, build_pointer_model, load_prepared_metadata,
)
from maths_ai.gnn_inference.scripts.train_local_identity import _loader, _write_json


MODEL_TYPE = "frozen_graph_library_identity_v1"


def _library_rows(dataset):
    return [i for i, graph in enumerate(dataset)
            if bool(graph.action_target_supported) and bool((graph.arg_lemma_ids >= 0).any())]


def _metrics(totals, split):
    result = dict(totals)
    result["split"] = split
    for cohort in ("retrieved", "matched"):
        count = totals[f"{cohort}_positions"]
        for metric in ("top1", "top5", "mrr", "text_top1"):
            result[f"{cohort}_{metric}"] = totals[f"{cohort}_{metric}_sum"] / max(count, 1)
    result["mean_library_loss"] = totals["loss_sum"] / max(totals["retrieved_positions"], 1)
    return result


def _run_epoch(model, loader, sampler, provider, index, *, device, epoch, split,
               k, optimizer=None, max_batches=0, start_batch=0,
               initial_counts=None, on_progress=None):
    model.eval()  # All frozen dropout stays off, including during query training.
    sampler.set_epoch(epoch)
    totals = Counter(initial_counts or {})
    training = optimizer is not None
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for batch_number, batch in enumerate(loader):
        if max_batches and batch_number >= max_batches:
            break
        if batch_number < start_batch:
            continue
        batch = batch.to(device)
        ids, vectors, scores = provider.search(batch, k=k)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            nodes = model.pointer.backbone.encode_nodes(batch)
            states = model.pointer.backbone.readout(nodes, batch)
        pools = candidate_groups_from_batch(batch, nodes, ids, vectors, scores)
        if training:
            optimizer.zero_grad(set_to_none=True)
        losses = []
        with torch.set_grad_enabled(training), torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            for row, (pool, gold) in enumerate(zip(pools, gold_arguments_from_batch(batch))):
                local_positions = {node: i for i, node in enumerate(pool.local_ids)}
                library_positions = {lemma: i for i, lemma in enumerate(pool.lemma_ids)}
                matched = all(argument.stable_id in (
                    local_positions if argument.source is CandidateSource.LOCAL else library_positions
                ) for argument in gold)
                with torch.no_grad():
                    hidden = model.initial_state(states[row], int(batch.y[row]))
                    adapted = model.lemma_adapter(pool.lemma_vectors)
                totals["rows_with_library"] += 1
                for step, argument in enumerate(gold):
                    if argument.source is CandidateSource.LIBRARY:
                        totals["all_library_positions"] += 1
                        position = library_positions.get(argument.stable_id)
                        if position is None:
                            totals["missing_library_positions"] += 1
                        else:
                            _, _, logits = model.score_step(hidden, pool, adapted=adapted, step=step)
                            target = torch.tensor([position], device=device)
                            losses.append(F.cross_entropy(logits.float().view(1, -1), target))
                            # Stable order breaks ties consistently with argmax.
                            order = torch.argsort(logits, descending=True, stable=True)
                            rank = int((order == position).nonzero(as_tuple=False)[0, 0]) + 1
                            for cohort in (("retrieved", "matched") if matched else ("retrieved",)):
                                totals[f"{cohort}_positions"] += 1
                                totals[f"{cohort}_top1_sum"] += int(rank == 1)
                                totals[f"{cohort}_top5_sum"] += int(rank <= 5)
                                totals[f"{cohort}_mrr_sum"] += 1.0 / rank
                                totals[f"{cohort}_text_top1_sum"] += int(
                                    pool.lemma_scores.argmax().item() == position
                                )
                            totals[f"slot_{step + 1}_retrieved_positions"] += 1
                            totals[f"slot_{step + 1}_top1"] += int(rank == 1)
                        # Missing gold supplies context only, never a selectable key.
                        with torch.no_grad():
                            if position is not None:
                                next_vector = adapted[position].view(1, -1)
                            else:
                                index_position = index.id_to_position.get(argument.stable_id)
                                if index_position is None:
                                    raise ValueError(f"gold lemma {argument.stable_id} absent from full text index")
                                vector = torch.as_tensor(index.lemma_vectors[index_position],
                                                         device=device, dtype=states.dtype).view(1, -1)
                                next_vector = model.lemma_adapter(vector)
                    else:
                        position = local_positions.get(argument.stable_id)
                        if position is None:
                            raise ValueError("gold local outside selectable pool")
                        next_vector = pool.local_vectors[position].view(1, -1)
                    with torch.no_grad():
                        hidden = model.pointer.argument_selector.gru(next_vector, hidden)
            if losses:
                loss = torch.stack(losses).mean()
                if not bool(torch.isfinite(loss).item()):
                    raise RuntimeError(f"nonfinite library identity loss at batch {batch_number}")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.library_query.parameters(), 1.0)
                    optimizer.step()
                totals["loss_sum"] += sum(float(item.detach()) for item in losses)
        totals["batches"] += 1
        if training and on_progress is not None and (batch_number + 1) % 50 == 0:
            on_progress(batch_number + 1, dict(totals))
        if (batch_number + 1) % 50 == 0 or batch_number + 1 == len(loader):
            print(f"{split} epoch {epoch}: batch {batch_number + 1}/{len(loader)}; "
                  f"retrieved_positions={totals['retrieved_positions']}", flush=True)
    return _metrics(totals, split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared-root", "joint-checkpoint", "local-checkpoint", "text-checkpoint",
                 "text-index", "candidate-dir", "run-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-batches", type=int, default=500)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--library-key-space", choices=("adapted", "text"), default="adapted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-val-library", type=int)
    parser.add_argument("--expected-val-matched", type=int)
    parser.add_argument("--expected-baseline-top1", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (min(args.epochs, args.batch_size, args.k) < 1 or args.max_train_batches < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        raise ValueError("invalid epochs, batch size, k, batch cap or learning rate")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    joint = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False)
    local = torch.load(args.local_checkpoint, map_location="cpu", weights_only=False)
    if (joint.get("model_type") != "joint_source_flat_decoder_v2"
            or local.get("model_type") != "frozen_graph_local_identity_v1"
            or local["binding"]["joint_checkpoint_sha256"] != file_sha256(args.joint_checkpoint)):
        raise ValueError("library probe requires the bound V2 joint and local checkpoints")
    index = LemmaIndex.load(args.text_index)
    manifest_sha = file_sha256(args.candidate_dir / "manifest.json")
    text_sha = file_sha256(args.text_checkpoint)
    for saved in (joint, local):
        binding = saved["binding"]
        if (binding["prepared_root"] != str(args.prepared_root.resolve())
                or binding["text_index_manifest"] != index.manifest
                or binding["text_checkpoint_sha256"] != text_sha
                or binding["candidate_manifest_sha256"] != manifest_sha):
            raise ValueError("checkpoint data, candidate or text binding changed")
    metadata = load_prepared_metadata(args.prepared_root, splits=("train", "val"))
    config = PointerConfig.from_dict(joint["config"])
    model = JointSourceDecoder(build_pointer_model(metadata, config), text_dim=index.lemma_vectors.shape[1])
    model.load_state_dict(local["model_state_dict"], strict=True)
    model.enable_library_identity(key_space=args.library_key_space)
    model.requires_grad_(False)
    model.library_query.requires_grad_(True)
    model = model.to(device).eval()
    optimizer = AdamW(model.library_query.parameters(), lr=args.learning_rate)
    provider = CachedTextLibraryCandidates(args.candidate_dir, index, checkpoint_path=args.text_checkpoint)
    for field, path in (
        ("argument_label_binding_sha256", args.prepared_root / "argument_labels.json"),
        ("prepared_packed_manifest_sha256", args.prepared_root / "packed/bidirectional/manifest.json"),
        ("node_vocab_sha256", args.prepared_root / "vocab/node_vocab.json"),
        ("tactic_vocab_sha256", args.prepared_root / "vocab/tactic_vocab.json"),
    ):
        if provider.manifest.get(field) != file_sha256(path):
            raise ValueError(f"candidate source binding changed: {field}")
    if args.k > provider.manifest["k"]:
        raise ValueError("requested k exceeds frozen candidate budget")
    datasets, loaders, indices = {}, {}, {}
    for split in ("train", "val"):
        datasets[split] = PreparedGraphDataset(metadata, split=split, edge_mode=config.edge_mode,
                                              cache_in_memory=True)
        provider.validate_dataset(datasets[split], split=split)
        indices[split] = _library_rows(datasets[split])
        loaders[split] = _loader(datasets[split], indices[split], batch_size=args.batch_size,
                                 seed=args.seed, train=split == "train")
    if not all(indices.values()) or loaders["val"][1].oversize_indices:
        raise ValueError("no library labels or oversized validation library graphs")
    excluded = sum(int((datasets["train"][indices["train"][i]].arg_lemma_ids >= 0).sum())
                   for i in loaders["train"][1].oversize_indices)
    binding = {
        "model_type": MODEL_TYPE, "prepared_root": str(args.prepared_root.resolve()),
        "joint_checkpoint_sha256": file_sha256(args.joint_checkpoint),
        "local_checkpoint_sha256": file_sha256(args.local_checkpoint),
        "candidate_manifest_sha256": manifest_sha, "text_checkpoint_sha256": text_sha,
        "text_index_manifest": index.manifest, "k": args.k, "seed": args.seed,
        "batch_size": args.batch_size, "max_train_batches": args.max_train_batches,
        "learning_rate": args.learning_rate, "excluded_train_library_positions": excluded,
        "trained_components": "library_query_only", "source_evidence": "frozen_legacy_library_scores",
        "prefix_policy": "gold_full_index_context_no_candidate_injection",
    }
    # Preserve first-pilot resume bindings; text-key runs get a distinct policy.
    if args.library_key_space != "adapted":
        binding["library_key_space"] = args.library_key_space
    root = args.run_dir
    if root.exists() and not args.resume:
        raise FileExistsError(f"library run exists: {root}; pass --resume")
    root.mkdir(parents=True, exist_ok=True)
    binding_path = root / "binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("library resume uses changed data or training policy")
    _write_json(binding_path, binding)
    print(f"Library rows: train={len(indices['train'])}, val={len(indices['val'])}; "
          f"excluded train library positions={excluded}; key_space={args.library_key_space}", flush=True)
    last_path, best_path = root / "last.pt", root / "best.pt"
    epoch, phase, offset, counts = 1, "ready", 0, None

    def validate(metrics):
        for key, expected in (("all_library_positions", args.expected_val_library),
                              ("matched_positions", args.expected_val_matched)):
            if expected is not None and metrics.get(key, 0) != expected:
                raise ValueError(f"validation cohort changed: {key}={metrics.get(key, 0)}, expected {expected}")

    def score(metrics):
        return metrics["matched_top1"], metrics["matched_mrr"], metrics["matched_top5"]

    best = (-1.0, -1.0, -1.0)

    def save(stage, *, next_batch=0, train_counts=None, metrics=None, checkpoint_epoch=None):
        payload = {
            "model_type": MODEL_TYPE, "binding": binding, "config": config.to_dict(),
            "epoch": epoch if checkpoint_epoch is None else checkpoint_epoch,
            "phase": stage, "next_batch": next_batch, "train_counts": train_counts,
            "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": metrics,
        }
        temporary = last_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(last_path)
        return payload

    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["binding"] != binding:
            raise ValueError("saved library checkpoint binding changed")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        epoch, phase, offset = saved["epoch"], saved["phase"], saved["next_batch"]
        counts = saved.get("train_counts") if phase == "training" else None
        if phase in {"complete", "initial"}:
            epoch += 1
    if best_path.exists():
        metrics = torch.load(best_path, map_location="cpu", weights_only=False)["val_metrics"]
        validate(metrics)
        best = score(metrics)
    if not last_path.exists():
        baseline = _run_epoch(model, *loaders["val"], provider, index, device=device,
                              epoch=0, split="val", k=args.k)
        validate(baseline)
        if (args.expected_baseline_top1 is not None
                and baseline.get("matched_top1_sum", 0) != args.expected_baseline_top1):
            raise ValueError("epoch-zero library top1 differs from expected baseline")
        if not baseline.get("matched_positions", 0):
            raise ValueError("no matched library validation positions")
        print(f"epoch 0 baseline: {json.dumps(baseline)}", flush=True)
        _write_json(root / "metrics_epoch_0.json", baseline)
        initial = save("initial", metrics=baseline, checkpoint_epoch=0)
        torch.save(initial, best_path)
        best = score(baseline)
    while epoch <= args.epochs:
        if phase != "train_done":
            train = _run_epoch(
                model, *loaders["train"], provider, index, device=device, epoch=epoch, split="train",
                k=args.k, optimizer=optimizer, max_batches=args.max_train_batches,
                start_batch=offset, initial_counts=counts,
                on_progress=lambda next_batch, totals: save("training", next_batch=next_batch, train_counts=totals),
            )
            print(f"epoch {epoch} train: {json.dumps(train)}", flush=True)
            _write_json(root / f"train_metrics_epoch_{epoch}.json", train)
            save("train_done")
        val = _run_epoch(model, *loaders["val"], provider, index, device=device,
                         epoch=epoch, split="val", k=args.k)
        validate(val)
        print(f"epoch {epoch} val: {json.dumps(val)}", flush=True)
        _write_json(root / f"val_metrics_epoch_{epoch}.json", val)
        payload = save("complete", metrics=val)
        if score(val) > best:
            best = score(val)
            torch.save(payload, best_path)
        epoch, phase, offset, counts = epoch + 1, "ready", 0, None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
