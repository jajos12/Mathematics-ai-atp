"""Train unified action decoder against fixed text-retrieved candidate IDs.

Only train/validation are used here. Text selects library IDs, while original
pointer-index vectors represent them to the decoder. Training can inject known
positives for loss; validation cannot. Checkpoint before validation preserves
training progress even if complete-action evaluation is interrupted.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256, load_state_dict_checked
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import load_index_for_encoder
from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer
from maths_ai.gnn_inference.atp_lean_gnn.premise_training import (
    evaluate_model_with_premises, train_one_epoch_with_premises,
)
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import CachedTextCandidates
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    REQUIRED_POINTER_DATA_FIELDS, _amp_dtype, build_dataloaders,
    build_pointer_model, load_pointer_config, load_prepared_metadata, resolve_device,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "prepared-root", "checkpoint", "index-path", "corpus-path",
                "candidate-dir", "run-dir"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.k) < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, K and learning rate must be positive")
    config = load_pointer_config(args.config, device_override=args.device)
    config = replace(
        config, prepared_root=args.prepared_root,
        training=replace(config.training, batch_size=args.batch_size, num_workers=0),
    ).normalized()
    device = resolve_device(args.device)
    metadata = load_prepared_metadata(args.prepared_root)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    if any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    model = build_pointer_model(metadata, config)
    load_state_dict_checked(model, state)
    index = load_index_for_encoder(
        args.index_path, encoder_state_dict=state,
        node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        corpus_path=args.corpus_path, expected_edge_mode=config.edge_mode,
    )
    provider = CachedTextCandidates(args.candidate_dir, index)
    if args.k > provider.manifest["k"]:
        raise ValueError("scorer K exceeds frozen retrieval budget")
    datasets, loaders = build_dataloaders(
        metadata, config, required_fields=REQUIRED_POINTER_DATA_FIELDS + ("arg_lemma_ids", "arg_count"),
    )
    for split in ("train", "val"):
        provider.validate_dataset(datasets[split], split=split)
    model.backbone.requires_grad_(False)
    model = model.to(device)
    scorer = PremiseScorer(config.model.hidden_dim).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    amp_dtype = _amp_dtype(device, config)
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    binding = {
        "pointer_checkpoint_sha256": file_sha256(args.checkpoint),
        "candidate_manifest_sha256": file_sha256(args.candidate_dir / "manifest.json"),
        "k": args.k, "batch_size": args.batch_size,
        "edge_mode": config.edge_mode,
    }
    root = args.run_dir
    if root.exists() and not args.resume:
        raise FileExistsError(f"scorer run exists: {root}; pass --resume")
    root.mkdir(parents=True, exist_ok=True)
    epoch, phase, next_batch = 1, "ready", 0
    best = -1.0
    resume_rng = None
    resume_cuda_rng = None
    if args.resume and (root / "last.pt").exists():
        saved = torch.load(root / "last.pt", map_location=device, weights_only=False)
        if saved["run_binding"] != binding:
            raise ValueError("resume scorer candidate policy or pointer differs")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        epoch, phase = int(saved["epoch"]), saved["phase"]
        next_batch = int(saved.get("next_batch", 0)) if phase == "training" else 0
        resume_rng = saved.get("torch_rng_state")
        resume_cuda_rng = saved.get("cuda_rng_states")
        if phase == "complete":
            epoch += 1
        if (root / "best.pt").exists():
            best = float(torch.load(root / "best.pt", map_location="cpu", weights_only=False)
                         ["val_metrics"]["complete_action_exact_match_predicted_tactic"])

    def save(phase, *, batch=0, metrics=None):
        payload = {
            "model_type": "unified_action_reranker", "epoch": epoch,
            "phase": phase, "next_batch": batch,
            "model_state_dict": model.state_dict(),
            "scorer_state_dict": scorer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config.to_dict(), "run_binding": binding,
            "node_vocab": metadata.node_vocab, "tactic_vocab": metadata.tactic_vocab,
            "lemma_index_manifest": index.manifest,
            "candidate_manifest": provider.manifest,
            "retriever_state_sha256": None, "val_metrics": metrics,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }
        torch.save(payload, root / "last.tmp")
        (root / "last.tmp").replace(root / "last.pt")
        return payload

    while epoch <= args.epochs:
        torch.manual_seed(config.seed + epoch)
        if resume_rng is not None:
            torch.set_rng_state(resume_rng.cpu())
        if resume_cuda_rng is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in resume_cuda_rng])
        sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if phase != "train_done":
            save("training", batch=next_batch)
            metrics = train_one_epoch_with_premises(
                model, scorer, loaders["train"], index, candidate_provider=provider,
                optimizer=optimizer, grad_scaler=scaler, device=device,
                grad_clip=1.0, unknown_tactic_id=metadata.unknown_tactic_id,
                arg_loss_weight=config.arg_loss_weight, premise_loss_weight=1.0,
                k=args.k, epoch=epoch, total_epochs=args.epochs,
                log_every_batches=50, use_amp=amp_dtype is not None, amp_dtype=amp_dtype,
                start_batch=next_batch, checkpoint_callback=lambda batch: save("training", batch=batch),
            )
            print(f"Epoch {epoch} scorer training: {metrics}", flush=True)
            save("train_done")
        metrics = evaluate_model_with_premises(
            model, scorer, loaders["val"], index, candidate_provider=provider,
            device=device, unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight, premise_loss_weight=1.0,
            tactic_vocab=metadata.tactic_vocab, k=args.k, split_name="val",
            log_every_batches=1, use_amp=amp_dtype is not None, amp_dtype=amp_dtype,
        )
        if metrics["evaluated_count"] != provider.manifest["splits"]["val"]["count"]:
            raise ValueError("scorer validation skipped frozen source states")
        print(f"Epoch {epoch} scorer validation: {json.dumps(metrics)}", flush=True)
        (root / f"metrics_epoch_{epoch}.json").write_text(json.dumps(metrics, indent=2))
        payload = save("complete", metrics=metrics)
        exact = float(metrics["complete_action_exact_match_predicted_tactic"])
        if exact > best:
            best = exact
            torch.save(payload, root / "best.pt")
        epoch, phase, next_batch = epoch + 1, "ready", 0
        resume_rng = resume_cuda_rng = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
