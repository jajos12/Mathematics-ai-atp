"""Train first-stage external-premise retrieval against a frozen lemma index.

The lemma tower and its normalized index stay frozen for the run. The state
tower starts from the same pointer backbone and learns queries using trace-cited
declarations, in-batch negatives, random declarations from the served Mathlib
environment, and nearest-neighbor hard negatives from the epoch-zero index.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

from maths_ai.gnn_inference.atp_lean_gnn.bundle import load_state_dict_checked
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import (
    load_index_for_encoder,
    read_index_manifest,
    state_dict_sha256,
)
from maths_ai.gnn_inference.atp_lean_gnn.logger import TrainingLogger
from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import DualEncoderRetriever
from maths_ai.gnn_inference.atp_lean_gnn.premise_retriever_training import (
    LemmaGraphStore,
    evaluate_retriever,
    train_retriever_epoch,
)
from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    REQUIRED_POINTER_DATA_FIELDS,
    build_dataloaders,
    build_pointer_model,
    load_pointer_config,
    load_prepared_metadata,
    resolve_device,
)


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a proof-state/lemma dual-encoder retriever."
    )
    parser.add_argument("--config", required=True, help="Pointer model config")
    parser.add_argument("--checkpoint", required=True, help="Pointer best.pt")
    parser.add_argument(
        "--index-path",
        required=True,
        help="Normalized index built from the pointer checkpoint",
    )
    parser.add_argument("--corpus-path", required=True, help="Mathlib lemma corpus")
    parser.add_argument(
        "--run-root", default="runs/premise_retriever", help="Output run directory"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--accessible-negatives", type=int, default=32)
    parser.add_argument("--hard-negatives", type=int, default=32)
    parser.add_argument("--cache-size", type=int, default=4096)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.accessible_negatives < 0 or args.hard_negatives < 0:
        raise ValueError("negative counts must be non-negative")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    config = load_pointer_config(args.config, device_override=args.device)
    metadata = load_prepared_metadata(config.prepared_root)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pointer_state = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if any(str(key).startswith("module.") for key in pointer_state):
        pointer_state = {
            str(key).removeprefix("module."): value
            for key, value in pointer_state.items()
        }
    pointer = build_pointer_model(metadata, config).to(device)
    load_state_dict_checked(pointer, pointer_state)

    # Verify before training. The index is intentionally frozen; changing the
    # lemma tower would require rebuilding all corpus vectors before evaluation.
    index = load_index_for_encoder(
        args.index_path,
        encoder_state_dict=pointer_state,
        node_vocab=metadata.node_vocab,
        tactic_vocab=metadata.tactic_vocab,
        corpus_path=args.corpus_path,
        expected_edge_mode=config.edge_mode,
    )
    if not index.normalize_queries:
        raise ValueError(
            "retriever requires a normalized index; rebuild with "
            "build_lemma_index.py --normalize"
        )

    model = DualEncoderRetriever(
        copy.deepcopy(pointer.backbone),
        copy.deepcopy(pointer.backbone),
        hidden_dim=config.model.hidden_dim,
        temperature=args.temperature,
    ).to(device)
    model.freeze_lemma_tower()
    del pointer

    required_fields = REQUIRED_POINTER_DATA_FIELDS + ("arg_lemma_ids", "arg_count")
    _, loaders = build_dataloaders(
        metadata, config, required_fields=required_fields
    )
    lemma_store = LemmaGraphStore.from_corpus(
        args.corpus_path,
        node_vocab=metadata.node_vocab,
        edge_mode=config.edge_mode,
        cache_size=args.cache_size,
    )
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_dir = _create_run_dir(Path(args.run_root))
    logger = TrainingLogger(run_dir)
    run_config = {
        "pointer_config": config.to_dict(),
        "pointer_checkpoint": Path(args.checkpoint).name,
        "frozen_index": Path(args.index_path).name,
        "lemma_corpus": Path(args.corpus_path).name,
        "index_manifest": read_index_manifest(args.index_path),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "temperature": args.temperature,
        "accessible_negative_count": args.accessible_negatives,
        "hard_negative_count": args.hard_negatives,
        # Corpus was enumerated from a Lean process launched with `import
        # Mathlib`; every sampled declaration is available in that served
        # environment. Per-source-file import visibility is not present in the
        # benchmark and is not claimed here.
        "accessible_negative_policy": "import_Mathlib_environment",
        "index_policy": "frozen_lemma_tower",
        "seed": args.seed,
    }
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8"
    )

    baseline_metrics = evaluate_retriever(
        model, loaders["val"], index, device=device
    )
    (run_dir / "pointer_index_baseline.json").write_text(
        json.dumps(baseline_metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    console_print(
        "Pointer-index baseline | "
        f"coverage={baseline_metrics['label_coverage']:.4f} "
        f"R@1={baseline_metrics['recall_at_1']:.4f} "
        f"R@10={baseline_metrics['recall_at_10']:.4f} "
        f"R@50={baseline_metrics['recall_at_50']:.4f} "
        f"R@200={baseline_metrics['recall_at_200']:.4f} "
        f"MRR={baseline_metrics['mrr']:.4f}"
    )

    best_mrr = -1.0
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_retriever_epoch(
            model,
            loaders["train"],
            lemma_store,
            index,
            optimizer=optimizer,
            device=device,
            accessible_negative_count=args.accessible_negatives,
            hard_negative_count=args.hard_negatives,
            grad_clip=args.grad_clip,
            rng=rng,
        )
        val_metrics = evaluate_retriever(
            model, loaders["val"], index, device=device
        )
        console_print(
            f"Epoch {epoch:02d} | loss={train_metrics['loss']:.4f} "
            f"coverage={val_metrics['label_coverage']:.4f} "
            f"R@1={val_metrics['recall_at_1']:.4f} "
            f"R@10={val_metrics['recall_at_10']:.4f} "
            f"R@50={val_metrics['recall_at_50']:.4f} "
            f"R@200={val_metrics['recall_at_200']:.4f} "
            f"MRR={val_metrics['mrr']:.4f} | "
            f"negatives(in_batch={train_metrics['in_batch_negative_count']}, "
            f"accessible={train_metrics['accessible_negative_count']}, "
            f"hard={train_metrics['hard_negative_count']})"
        )
        log_metrics = {
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "baseline_mrr": baseline_metrics["mrr"],
        }
        logger.log_epoch(epoch, log_metrics)

        if float(val_metrics["mrr"]) > best_mrr:
            best_mrr = float(val_metrics["mrr"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": "dual_encoder_retriever",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": run_config,
                    "val_metrics": val_metrics,
                    "pointer_index_baseline": baseline_metrics,
                    "state_encoder_sha256": state_dict_sha256(
                        model.state_encoder.state_dict()
                    ),
                    "lemma_encoder_sha256": state_dict_sha256(
                        model.lemma_encoder.state_dict()
                    ),
                    "node_vocab": metadata.node_vocab,
                    "tactic_vocab": metadata.tactic_vocab,
                },
                run_dir / "best.pt",
            )

    best_run_link = Path(args.run_root) / "best_run"
    if best_run_link.exists() or best_run_link.is_symlink():
        best_run_link.unlink()
    best_run_link.symlink_to(run_dir.name)
    console_print(f"Best validation MRR: {best_mrr:.4f}; run saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
