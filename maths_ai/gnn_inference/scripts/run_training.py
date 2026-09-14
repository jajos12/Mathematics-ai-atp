#!/usr/bin/env python3
"""Unified training pipeline for GNN-based tactic prediction.

Runs the full training loop: dataset preparation → baseline model → pointer model
→ premise scorer, with configurable options at every stage.

Usage:
    # Run full pipeline with default config
    python -m maths_ai.gnn_inference.scripts.run_training

    # Run with a custom config
    python -m maths_ai.gnn_inference.scripts.run_training --config my_config.json

    # Run only specific stages
    python -m maths_ai.gnn_inference.scripts.run_training --stages prepare,baseline

    # Resume from a checkpoint
    python -m maths_ai.gnn_inference.scripts.run_training --resume

    # Override individual parameters from CLI
    python -m maths_ai.gnn_inference.scripts.run_training \
        --model.hidden_dim 256 \
        --training.epochs 10 \
        --training.batch_size 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from maths_ai.gnn_inference.atp_lean_gnn.reporting import console_print


# ──────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_name": "default",
    "stages": ["prepare", "baseline", "pointer", "scorer"],
    "prepared_root": "maths_ai/gnn_inference/artifacts/prepared/v1",
    "run_root": "maths_ai/gnn_inference/runs",

    "prepare": {
        "dataset_name": "cat-searcher/leandojo-benchmark-4-random",
        "splits": "train,val,test",
        "sample_per_split": None,
        "selection_manifest": None,
        "force": False,
        "use_sexpr": False,
        "sexpr_cache_root": None,
        "sexpr_variant": "model",
        "project_path": "maths_ai/lean_mathlib",
    },

    "baseline": {
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_size": 256,
        "max_batch_nodes": 0,
        "max_batch_edges": 0,
        "oversize_graph_policy": "singleton",
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "num_workers": 12,
        "use_amp": True,
        "cache_in_memory": False,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0001,
    },

    "pointer": {
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.2,
        "heads": 8,
        "readout": "state",
        "max_args": 3,
        "arg_loss_weight": 0.5,
        "initialization_checkpoint": None,
        "batch_size": 256,
        "max_batch_nodes": 0,
        "max_batch_edges": 0,
        "oversize_graph_policy": "singleton",
        "epochs": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "num_workers": 8,
        "use_amp": True,
        "cache_in_memory": False,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0001,
    },

    "scorer": {
        "premise_config": "maths_ai/gnn_inference/configs/premise_scoring.json",
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.2,
        "batch_size": 256,
        "max_batch_nodes": 0,
        "max_batch_edges": 0,
        "oversize_graph_policy": "singleton",
        "epochs": 10,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "grad_clip": 1.0,
        "num_workers": 8,
        "use_amp": True,
        "cache_in_memory": False,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0001,
        "k": 200,
        "premise_loss_weight": 1.0,
        "scoring_mode": "dot",
    },

    "seed": 42,
    "device": "auto",
    "gnn_type": "sage",
}


# ──────────────────────────────────────────────────────────────────────────
# Pipeline state tracking
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Tracks which stages have completed and where their outputs are."""
    completed_stages: list[str] = field(default_factory=list)
    stage_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    start_time: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["elapsed_seconds"] = time.time() - self.start_time if self.start_time else 0
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def mark_complete(self, stage: str, output: dict[str, Any]) -> None:
        self.completed_stages.append(stage)
        self.stage_outputs[stage] = output

    def is_complete(self, stage: str) -> bool:
        return stage in self.completed_stages

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            completed_stages=data.get("completed_stages", []),
            stage_outputs=data.get("stage_outputs", {}),
            start_time=data.get("start_time", 0.0),
        )


# ──────────────────────────────────────────────────────────────────────────
# Stage runners
# ──────────────────────────────────────────────────────────────────────────

def run_prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 1: Prepare the dataset (build vocabularies, convert to PyG)."""
    from maths_ai.gnn_inference.atp_lean_gnn.preprocess import (
        PreprocessConfig,
        run_preprocessing,
    )

    prepare_cfg = config["prepare"]
    output_root = Path(config["prepared_root"]).resolve()

    console_print("\n" + "=" * 60)
    console_print("  STAGE 1: PREPARE DATASET")
    console_print("=" * 60)
    console_print(f"  Dataset   : {prepare_cfg['dataset_name']}")
    console_print(f"  Output    : {output_root}")
    console_print(f"  Splits    : {prepare_cfg['splits']}")
    console_print(f"  Sample    : {prepare_cfg['sample_per_split'] or 'all'}")
    console_print(f"  Force     : {prepare_cfg['force']}")
    console_print("")

    # Check if output already exists with valid data
    if output_root.exists():
        manifests_dir = output_root / "manifests"
        has_data = manifests_dir.exists() and any(manifests_dir.glob("*.json"))
        if has_data:
            if prepare_cfg["force"]:
                console_print("  Output exists but --force is set. Rebuilding...")
            else:
                console_print(f"  Output already exists at {output_root}")
                console_print("  Skipping prepare stage. Use --prepare.force true to rebuild.")
                return {
                    "summary": {"skipped": True, "reason": "output exists"},
                    "output_root": str(output_root),
                }
        else:
            console_print(f"  Output directory exists but is empty/incomplete. Proceeding...")

    cfg = PreprocessConfig(
        dataset_name=prepare_cfg["dataset_name"],
        splits=tuple(s.strip() for s in prepare_cfg["splits"].split(",")),
        output_root=output_root,
        sample_per_split=prepare_cfg["sample_per_split"],
        force=prepare_cfg["force"],
        use_sexpr=prepare_cfg.get("use_sexpr", False),
        sexpr_cache_root=(
            Path(prepare_cfg["sexpr_cache_root"])
            if prepare_cfg.get("sexpr_cache_root")
            else None
        ),
        sexpr_variant=prepare_cfg.get("sexpr_variant", "model"),
        project_path=prepare_cfg.get("project_path", "maths_ai/lean_mathlib"),
        selection_manifest=(
            Path(prepare_cfg["selection_manifest"])
            if prepare_cfg.get("selection_manifest")
            else None
        ),
    )

    summary = run_preprocessing(cfg)
    return {"summary": summary, "output_root": str(output_root)}


def _resolve_stage_model(baseline_cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge inline and nested ``model``/``training`` hyperparams for a stage.

    ``DEFAULT_CONFIG`` stores stage hyperparams inline (e.g. ``baseline.hidden_dim``),
    while ``--config`` preset files follow ``BaselineConfig.to_dict()`` and nest them
    under ``baseline.model`` / ``baseline.training``. This resolves both shapes,
    preferring the nested form when present.
    """
    inline_model_keys = ("hidden_dim", "num_layers", "dropout", "heads", "readout")
    inline_training_keys = (
        "batch_size", "max_batch_nodes", "max_batch_edges", "oversize_graph_policy", "epochs", "learning_rate", "weight_decay",
        "grad_clip", "num_workers", "use_amp", "cache_in_memory",
        "early_stopping_patience", "early_stopping_min_delta",
    )
    model = {k: baseline_cfg[k] for k in inline_model_keys if k in baseline_cfg}
    model.update(baseline_cfg.get("model", {}) or {})
    training = {k: baseline_cfg[k] for k in inline_training_keys if k in baseline_cfg}
    training.update(baseline_cfg.get("training", {}) or {})
    return model, training


def run_baseline(config: dict[str, Any], resume_run_dir: str | None = None) -> dict[str, Any]:
    """Stage 2: Train baseline GNN classifier."""
    from maths_ai.gnn_inference.atp_lean_gnn.training import (
        BaselineConfig,
        load_baseline_config,
        train_baseline,
        _create_run_dir,
        _write_json,
        _safe_num_workers,
    )

    baseline_cfg = config["baseline"]
    prepared_root = Path(config["prepared_root"]).resolve()
    run_root = (Path(config["run_root"]) / "baseline_gnn").resolve()

    console_print("\n" + "=" * 60)
    console_print("  STAGE 2: TRAIN BASELINE MODEL")
    console_print("=" * 60)
    console_print(f"  Prepared  : {prepared_root}")
    console_print(f"  Run root  : {run_root}")
    _m, _t = _resolve_stage_model(baseline_cfg)
    console_print(f"  hidden    : {_m.get('hidden_dim')}")
    console_print(f"  layers    : {_m.get('num_layers')}")
    console_print(f"  dropout   : {_m.get('dropout')}")
    if config.get("gnn_type") == "gat":
        console_print(f"  readout   : {_m.get('readout', 'state')}")
    console_print(f"  batch     : {baseline_cfg['batch_size']}")
    console_print(f"  epochs    : {baseline_cfg['epochs']}")
    console_print(f"  lr        : {baseline_cfg['learning_rate']}")
    _eff_workers, _ = _safe_num_workers(baseline_cfg["num_workers"], pin_memory=True)
    console_print(f"  workers   : {_eff_workers} (requested {baseline_cfg['num_workers']})")
    console_print("")

    model_overrides, training_overrides = _resolve_stage_model(baseline_cfg)
    gnn_type = baseline_cfg.get("gnn_type", config["gnn_type"])

    if resume_run_dir is not None:
        # The run-local config is authoritative for architecture compatibility;
        # only the prepared cache location and total epoch target may change.
        cfg = load_baseline_config(
            Path(resume_run_dir) / "config.json",
            prepared_root_override=prepared_root,
            epochs_override=int(training_overrides["epochs"]),
            device_override=config["device"],
            training_overrides=dict(training_overrides),
        )
    else:
        # Rebuild config from dict to ensure proper normalization.
        cfg = BaselineConfig.from_dict({
            "prepared_root": str(prepared_root),
            "run_root": str(run_root),
            "seed": config["seed"],
            "device": config["device"],
            "edge_mode": baseline_cfg.get("edge_mode", "bidirectional"),
            "use_node_type": baseline_cfg.get("use_node_type", True),
            "gnn_type": gnn_type,
            "model": dict(model_overrides),
            "training": {
                "log_every_batches": 100,
                "pin_memory": True,
                "persistent_workers": training_overrides.get("num_workers", 0) > 0,
                "prefetch_factor": 2,
                **training_overrides,
            },
        })

    summary = train_baseline(cfg, resume_run_dir=resume_run_dir)
    best_checkpoint = summary.get("best_checkpoint", "")

    # Create best_run symlink
    best_run_link = run_root / "best_run"
    run_dir = Path(summary["run_dir"])
    if best_run_link.exists() or best_run_link.is_symlink():
        best_run_link.unlink()
    best_run_link.symlink_to(run_dir.name)

    return {
        "summary": summary,
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_run_link": str(best_run_link),
    }


def run_pointer(config: dict[str, Any], resume_run_dir: str | None = None) -> dict[str, Any]:
    """Stage 3: Train pointer-based argument selection model."""
    from maths_ai.gnn_inference.atp_lean_gnn.training import (
        PointerConfig,
        load_pointer_config,
        train_pointer,
    )

    pointer_cfg = config["pointer"]
    prepared_root = Path(config["prepared_root"]).resolve()
    run_root = (Path(config["run_root"]) / "pointer_gnn").resolve()

    console_print("\n" + "=" * 60)
    console_print("  STAGE 3: TRAIN POINTER MODEL")
    console_print("=" * 60)
    console_print(f"  Prepared  : {prepared_root}")
    console_print(f"  Run root  : {run_root}")
    _pm, _pt = _resolve_stage_model(pointer_cfg)
    console_print(f"  hidden    : {_pm.get('hidden_dim')}")
    console_print(f"  layers    : {_pm.get('num_layers')}")
    console_print(f"  dropout   : {_pm.get('dropout')}")
    console_print(f"  batch     : {_pt.get('batch_size')}")
    console_print(f"  epochs    : {_pt.get('epochs')}")
    console_print(f"  max_args  : {pointer_cfg.get('max_args', 3)}")
    console_print(f"  arg_wt    : {pointer_cfg.get('arg_loss_weight', 0.5)}")
    console_print(
        "  init ckpt : "
        f"{pointer_cfg.get('initialization_checkpoint') or 'random initialization'}"
    )
    console_print("")

    model_overrides, training_overrides = _resolve_stage_model(pointer_cfg)
    gnn_type = pointer_cfg.get("gnn_type", config["gnn_type"])

    if resume_run_dir is not None:
        cfg = load_pointer_config(
            Path(resume_run_dir) / "config.json",
            prepared_root_override=prepared_root,
            epochs_override=int(training_overrides["epochs"]),
            device_override=config["device"],
            training_overrides=dict(training_overrides),
        )
    else:
        cfg = PointerConfig.from_dict({
            "prepared_root": str(prepared_root),
            "run_root": str(run_root),
            "seed": config["seed"],
            "device": config["device"],
            "edge_mode": pointer_cfg.get("edge_mode", "bidirectional"),
            "use_node_type": pointer_cfg.get("use_node_type", True),
            "gnn_type": gnn_type,
            "max_args": pointer_cfg.get("max_args", 3),
            "arg_loss_weight": pointer_cfg.get("arg_loss_weight", 0.5),
            "initialization_checkpoint": pointer_cfg.get("initialization_checkpoint"),
            "model": dict(model_overrides),
            "training": {
                "log_every_batches": 50,
                "pin_memory": True,
                "persistent_workers": training_overrides.get("num_workers", 0) > 0,
                "prefetch_factor": 2,
                **training_overrides,
            },
        })

    summary = train_pointer(cfg, resume_run_dir=resume_run_dir)
    best_checkpoint = summary.get("best_checkpoint", "")

    # Create best_run symlink
    best_run_link = run_root / "best_run"
    run_dir = Path(summary["run_dir"])
    if best_run_link.exists() or best_run_link.is_symlink():
        best_run_link.unlink()
    best_run_link.symlink_to(run_dir.name)

    return {
        "summary": summary,
        "run_dir": str(run_dir),
        "best_checkpoint": best_checkpoint,
        "best_run_link": str(best_run_link),
    }


def run_scorer(config: dict[str, Any]) -> dict[str, Any]:
    """Stage 4: Train premise scorer on top of frozen pointer model."""
    scorer_cfg = config["scorer"]
    pointer_cfg = config["pointer"]
    prepared_root = Path(config["prepared_root"]).resolve()
    run_root = (Path(config["run_root"]) / "premise_gnn").resolve()

    console_print("\n" + "=" * 60)
    console_print("  STAGE 4: TRAIN PREMISE SCORER")
    console_print("=" * 60)

    # Find pointer checkpoint
    pointer_run_root = Path(config["run_root"]) / "pointer_gnn"
    best_run_link = pointer_run_root / "best_run"
    if best_run_link.is_symlink():
        pointer_run_dir = best_run_link.resolve()
    else:
        # Find latest run
        runs = sorted(pointer_run_root.glob("run_*"))
        if not runs:
            console_print("  ERROR: No pointer model runs found. Train pointer first.")
            return {"error": "no pointer checkpoint found"}
        pointer_run_dir = runs[-1]

    pointer_checkpoint = pointer_run_dir / "best.pt"
    if not pointer_checkpoint.exists():
        console_print(f"  ERROR: Pointer checkpoint not found at {pointer_checkpoint}")
        return {"error": "pointer checkpoint not found"}

    console_print(f"  Pointer ckpt: {pointer_checkpoint}")
    console_print(f"  Run root    : {run_root}")
    console_print(f"  hidden      : {scorer_cfg['hidden_dim']}")
    console_print(f"  scoring     : {scorer_cfg['scoring_mode']}")
    console_print(f"  k           : {scorer_cfg['k']}")
    console_print(f"  epochs      : {scorer_cfg['epochs']}")
    console_print("")

    from maths_ai.gnn_inference.scripts.train_scorer import _create_run_dir
    from maths_ai.gnn_inference.atp_lean_gnn.training import (
        REQUIRED_POINTER_DATA_FIELDS, _amp_dtype, build_dataloaders,
        build_pointer_model, load_pointer_config, load_prepared_metadata, resolve_device,
    )
    from maths_ai.gnn_inference.atp_lean_gnn.premise_scoring import PremiseScorer, PremiseScorerConfig
    from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
    from maths_ai.gnn_inference.atp_lean_gnn.logger import TrainingLogger
    from maths_ai.gnn_inference.atp_lean_gnn.premise_training import evaluate_model_with_premises, train_one_epoch_with_premises

    import torch
    from dataclasses import replace
    from torch.optim import AdamW

    device = resolve_device(str(config["device"]))

    # Load pointer config to get model architecture
    pointer_config_path = pointer_run_dir / "config.json"
    if not pointer_config_path.exists():
        pointer_config_path = Path("maths_ai/gnn_inference/configs/pointer_graphsage_state.json")
    p_config = load_pointer_config(pointer_config_path)
    scorer_training = replace(
        p_config.training,
        batch_size=int(scorer_cfg["batch_size"]),
        max_batch_nodes=int(scorer_cfg.get("max_batch_nodes", 0)),
        max_batch_edges=int(scorer_cfg.get("max_batch_edges", 0)),
        oversize_graph_policy=str(scorer_cfg.get("oversize_graph_policy", "singleton")),
        epochs=int(scorer_cfg["epochs"]),
        learning_rate=float(scorer_cfg["learning_rate"]),
        weight_decay=float(scorer_cfg["weight_decay"]),
        grad_clip=float(scorer_cfg["grad_clip"]),
        num_workers=int(scorer_cfg["num_workers"]),
        use_amp=bool(scorer_cfg["use_amp"]),
        cache_in_memory=bool(scorer_cfg.get("cache_in_memory", False)),
        early_stopping_patience=int(scorer_cfg.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(scorer_cfg.get("early_stopping_min_delta", 0.0)),
    )
    p_config = replace(
        p_config, prepared_root=prepared_root, device=str(config["device"]),
        training=scorer_training,
    ).normalized()
    amp_dtype = _amp_dtype(device, p_config)
    use_amp = amp_dtype is not None
    metadata = load_prepared_metadata(prepared_root)

    # Load premise scoring config
    premise_config_path = Path(scorer_cfg["premise_config"])
    if premise_config_path.exists():
        with open(premise_config_path) as f:
            p_cfg_dict = json.load(f)
        # Override with our config values
        p_cfg_dict["scoring_mode"] = scorer_cfg["scoring_mode"]
        p_cfg_dict["k"] = scorer_cfg["k"]
        p_cfg_dict["premise_loss_weight"] = scorer_cfg["premise_loss_weight"]
        p_cfg_dict["hidden_dim"] = p_config.model.hidden_dim
    else:
        p_cfg_dict = {
            "scoring_mode": scorer_cfg["scoring_mode"],
            "k": scorer_cfg["k"],
            "premise_loss_weight": scorer_cfg["premise_loss_weight"],
            "hidden_dim": p_config.model.hidden_dim,
        }
    p_config_obj = PremiseScorerConfig(**p_cfg_dict)

    run_dir = _create_run_dir(run_root)
    logger = TrainingLogger(run_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "pointer_config": p_config.to_dict(),
            "scorer_config": p_config_obj.to_dict(),
        }, handle, indent=2, sort_keys=True)

    # Try to load lemma index
    lemma_index_path = Path(config["run_root"]) / "lemma_index_v1" / "best_run"
    if lemma_index_path.is_symlink():
        lemma_index_dir = lemma_index_path.resolve()
    else:
        lemma_index_dir = Path(config["run_root"]) / "lemma_index_v1"

    lemma_index_file = None
    for candidate in [
        lemma_index_dir / "faiss.index",
        lemma_index_dir / "lemma_index.faiss",
        lemma_index_dir / "index" / "faiss.index",
        lemma_index_dir / "index" / "lemma_index.faiss",
    ]:
        if candidate.exists():
            lemma_index_file = candidate
            break

    # Also try the directory itself (LemmaIndex.load accepts directories)
    if lemma_index_file is None and lemma_index_dir.exists():
        if (lemma_index_dir / "faiss.index").exists():
            lemma_index_file = lemma_index_dir

    if lemma_index_file is None:
        console_print("  WARNING: No lemma index found. Scorer training will use local candidates only.")
        lemma_index = None
    else:
        console_print(f"  Lemma index: {lemma_index_file}")
        # The index must have been built by *this* pointer's backbone before
        # retrieval trains against it. A mismatched index scores plausible
        # nonsense (same dimensions, different vector space), which is worse
        # than no index at all -- the old behavior loaded it unconditionally.
        # The binding is over the encoder's weight tensors, so an index built
        # from the baseline this pointer was initialized from is refused too
        # once the backbone has trained; rebuild the index from the pointer
        # checkpoint before the scorer stage.
        from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import (
            load_index_for_encoder,
        )

        index_probe = torch.load(pointer_checkpoint, map_location="cpu", weights_only=False)
        index_probe_state = (
            index_probe["model_state_dict"]
            if "model_state_dict" in index_probe else index_probe
        )
        try:
            lemma_index = load_index_for_encoder(
                lemma_index_file,
                encoder_state_dict=index_probe_state,
                node_vocab=metadata.node_vocab,
                tactic_vocab=metadata.tactic_vocab,
            )
        except ValueError as exc:
            console_print(f"  ERROR: {exc}")
            return {"error": "lemma index does not match its encoder"}
        del index_probe, index_probe_state

    datasets, loaders = build_dataloaders(
        metadata, p_config, required_fields=REQUIRED_POINTER_DATA_FIELDS
    )

    # Build model
    model = build_pointer_model(metadata, p_config)

    ckpt = torch.load(pointer_checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    has_trained_tactic_embedding = any(k.startswith("tactic_embedding.") for k in state_dict)
    if not has_trained_tactic_embedding:
        with torch.no_grad():
            model.tactic_embedding.weight.copy_(model.backbone.classifier.weight)

    for param in model.backbone.parameters():
        param.requires_grad = False
    model = model.to(device)

    scorer = PremiseScorer(hidden_dim=p_config.model.hidden_dim, mode=p_config_obj.scoring_mode).to(device)

    trainable_params = (
        list(model.tactic_embedding.parameters())
        + list(model.argument_selector.parameters())
        + list(scorer.parameters())
    )
    optimizer = AdamW(trainable_params, lr=p_config.training.learning_rate, weight_decay=p_config.training.weight_decay)
    grad_scaler = torch.amp.GradScaler(
        device.type, enabled=(amp_dtype == torch.float16)
    )

    best_val_mrr = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    last_checkpoint_path = run_dir / "last.pt"
    best_checkpoint_path = run_dir / "best.pt"
    for epoch in range(1, p_config.training.epochs + 1):
        batch_sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch_with_premises(
            model=model, scorer=scorer, loader=loaders["train"],
            lemma_index=lemma_index, optimizer=optimizer, grad_scaler=grad_scaler,
            device=device, grad_clip=p_config.training.grad_clip,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=p_config.arg_loss_weight if hasattr(p_config, "arg_loss_weight") else 0.5,
            premise_loss_weight=p_config_obj.premise_loss_weight,
            k=p_config_obj.k, epoch=epoch, total_epochs=p_config.training.epochs,
            log_every_batches=p_config.training.log_every_batches,
            use_amp=use_amp, amp_dtype=amp_dtype,
            pin_memory=p_config.training.pin_memory,
        )

        val_metrics = evaluate_model_with_premises(
            model=model, scorer=scorer, loader=loaders["val"],
            lemma_index=lemma_index, device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=p_config.arg_loss_weight if hasattr(p_config, "arg_loss_weight") else 0.5,
            premise_loss_weight=p_config_obj.premise_loss_weight,
            k=p_config_obj.k, split_name="val",
            log_every_batches=p_config.training.log_every_batches,
            use_amp=use_amp, amp_dtype=amp_dtype,
            pin_memory=p_config.training.pin_memory,
        )

        console_print(
            f"  Epoch {epoch:02d}/{p_config.training.epochs:02d} | "
            f"Val MRR: {val_metrics['premise_mrr']:.4f} | "
            f"Hit@1: {val_metrics['premise_top1_accuracy']:.4f} | "
            f"Hit@5: {val_metrics['premise_top5_accuracy']:.4f}"
        )

        if (
            val_metrics["premise_mrr"]
            > best_val_mrr + p_config.training.early_stopping_min_delta
        ):
            best_val_mrr = val_metrics["premise_mrr"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "scorer_state_dict": scorer.state_dict(),
                "val_metrics": val_metrics,
            }, best_checkpoint_path)
        else:
            epochs_without_improvement += 1

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "scorer_state_dict": scorer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
        }, last_checkpoint_path)

        logger.log_epoch(epoch, {
            "train_tactic_loss": float(train_metrics["tactic_loss"]),
            "train_arg_loss": float(train_metrics["arg_loss"]),
            "train_premise_loss": float(train_metrics["premise_loss"]),
            "train_combined_loss": float(train_metrics["combined_loss"]),
            "val_premise_mrr": float(val_metrics["premise_mrr"]),
            "val_premise_top1_accuracy": float(val_metrics["premise_top1_accuracy"]),
            "val_premise_top5_accuracy": float(val_metrics["premise_top5_accuracy"]),
        })

        if (
            p_config.training.early_stopping_patience > 0
            and epochs_without_improvement >= p_config.training.early_stopping_patience
        ):
            console_print(
                "  Early stopping: validation MRR did not improve for "
                f"{epochs_without_improvement} epochs (best epoch {best_epoch})."
            )
            break

    best_checkpoint = torch.load(
        best_checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    scorer.load_state_dict(best_checkpoint["scorer_state_dict"], strict=True)
    test_metrics = evaluate_model_with_premises(
        model=model, scorer=scorer, loader=loaders["test"],
        lemma_index=lemma_index, device=device,
        unknown_tactic_id=metadata.unknown_tactic_id,
        arg_loss_weight=p_config.arg_loss_weight,
        premise_loss_weight=p_config_obj.premise_loss_weight,
        k=p_config_obj.k, split_name="test",
        log_every_batches=p_config.training.log_every_batches,
        use_amp=use_amp, amp_dtype=amp_dtype,
        pin_memory=p_config.training.pin_memory,
    )
    with (run_dir / "eval_val.json").open("w", encoding="utf-8") as handle:
        json.dump(best_checkpoint["val_metrics"], handle, indent=2, sort_keys=True)
    with (run_dir / "eval_test.json").open("w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2, sort_keys=True)
    summary = {
        "run_dir": str(run_dir),
        "pointer_checkpoint": str(pointer_checkpoint),
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation": best_checkpoint["val_metrics"],
        "test_evaluation": test_metrics,
        "device": str(device),
        "amp_dtype": str(amp_dtype).removeprefix("torch.") if amp_dtype else "float32",
        "lemma_index": str(lemma_index_file) if lemma_index_file else None,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    # Create best_run symlink
    best_run_link = run_root / "best_run"
    if best_run_link.exists() or best_run_link.is_symlink():
        best_run_link.unlink()
    best_run_link.symlink_to(run_dir.name)

    return {
        "run_dir": str(run_dir),
        "best_val_mrr": best_val_mrr,
        "best_checkpoint": str(best_checkpoint_path),
        "summary": summary,
        "best_run_link": str(best_run_link),
    }


# ──────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────────────

def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a nested config value from a dotted key like 'model.hidden_dim'."""
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    # Try to cast to the right type (check bool before int since bool is subclass of int)
    existing = d.get(parts[-1])
    if isinstance(existing, bool):
        d[parts[-1]] = value.lower() in ("true", "1", "yes")
    elif isinstance(existing, int):
        d[parts[-1]] = int(value)
    elif isinstance(existing, float):
        d[parts[-1]] = float(value)
    elif existing is None:
        # Try to infer from value
        if value.lower() in ("true", "false"):
            d[parts[-1]] = value.lower() == "true"
        else:
            try:
                d[parts[-1]] = int(value)
            except ValueError:
                try:
                    d[parts[-1]] = float(value)
                except ValueError:
                    d[parts[-1]] = value
    else:
        d[parts[-1]] = value


def _apply_cli_override(config: dict[str, Any], key: str, value: str) -> None:
    """Apply an inline stage override to its authoritative nested preset too."""
    _set_nested(config, key, value)
    parts = key.split(".")
    if len(parts) != 2:
        return
    stage, field = parts
    stage_config = config.get(stage)
    if not isinstance(stage_config, dict):
        return
    for section in ("model", "training"):
        nested = stage_config.get(section)
        if isinstance(nested, dict) and field in nested:
            _set_nested(config, f"{stage}.{section}.{field}", value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified training pipeline for GNN tactic prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults
  python -m maths_ai.gnn_inference.scripts.run_training

  # Custom config file
  python -m maths_ai.gnn_inference.scripts.run_training --config my_config.json

  # Only run specific stages
  python -m maths_ai.gnn_inference.scripts.run_training --stages prepare,baseline

  # Override parameters
  python -m maths_ai.gnn_inference.scripts.run_training \\
      --baseline.hidden_dim 256 \\
      --baseline.epochs 10 \\
      --baseline.batch_size 128

  # Resume from last checkpoint
  python -m maths_ai.gnn_inference.scripts.run_training --resume

  # Force re-prepare dataset
  python -m maths_ai.gnn_inference.scripts.run_training --prepare.force true
        """,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file (overrides defaults)")
    parser.add_argument("--stages", type=str, default=None, help="Comma-separated stages to run: prepare,baseline,pointer,scorer")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (requires pipeline_state.json)")
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Extend a specific completed baseline or pointer run from its last.pt checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Target total epochs when using --resume-run-dir",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit without running")
    parser.add_argument("--experiment-name", type=str, default=None, help="Name for this experiment run")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device: auto, cpu, cuda")
    parser.add_argument("--gnn-type", type=str, default=None, choices=["sage", "gat"], help="GNN backbone: sage (GraphSAGE) or gat (GATv2)")
    parser.add_argument("--prepared-root", type=str, default=None, help="Path to prepared dataset root")
    parser.add_argument("--run-root", type=str, default=None, help="Path to runs output directory")

    # Stage-specific overrides
    for stage in ["baseline", "pointer", "scorer", "prepare"]:
        group = parser.add_argument_group(f"{stage} overrides")
        for key, value in DEFAULT_CONFIG.get(stage, {}).items():
            if isinstance(value, dict):
                for k2, v2 in value.items():
                    group.add_argument(f"--{stage}.{k2}", type=str, default=None,
                                       help=f"{stage}.{k2} = {v2} (default)")
            else:
                group.add_argument(f"--{stage}.{key}", type=str, default=None,
                                   help=f"{stage}.{key} = {value} (default)")

    return parser


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load base config
    config = DEFAULT_CONFIG.copy()
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            console_print(f"  ERROR: Config file not found: {config_path}")
            return 1
        with open(config_path) as f:
            user_config = json.load(f)
        # Deep merge
        for key, value in user_config.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value

        # A preset file (e.g. baseline_gat_state.json) is a standalone
        # BaselineConfig/PointerConfig with top-level ``model``/``training``
        # rather than wrapped in a ``baseline``/``pointer`` section. Fold those
        # fields into the matching stage so run_baseline/run_pointer see them.
        if "model" in user_config or "training" in user_config:
            stage_key = "pointer" if "max_args" in user_config else "baseline"
            target = config.setdefault(stage_key, {})
            for fld in (
                "model", "training", "gnn_type", "prepared_root", "run_root",
                "device", "edge_mode", "use_node_type", "seed", "max_args",
                "arg_loss_weight", "initialization_checkpoint",
            ):
                if fld in user_config:
                    if fld in ("model", "training"):
                        target[fld] = {**target.get(fld, {}), **user_config[fld]}
                    else:
                        target[fld] = user_config[fld]
            # Also surface gnn_type at top level for the banner/CLI path.
            if "gnn_type" in user_config:
                config["gnn_type"] = user_config["gnn_type"]

    # Apply CLI overrides
    if args.stages:
        config["stages"] = [s.strip() for s in args.stages.split(",")]
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.gnn_type:
        config["gnn_type"] = args.gnn_type
    if args.prepared_root:
        config["prepared_root"] = args.prepared_root
    if args.run_root:
        config["run_root"] = args.run_root

    # Apply nested overrides
    explicit_stage_overrides = {
        key: value
        for key, value in vars(args).items()
        if "." in key and value is not None
    }
    for key, value in vars(args).items():
        if "." in key and value is not None:
            _apply_cli_override(config, key, value)

    if args.resume_run_dir:
        if args.resume:
            console_print("  ERROR: Use either --resume or --resume-run-dir, not both.")
            return 1
        if args.epochs is None:
            console_print("  ERROR: --resume-run-dir requires --epochs with a new total epoch target.")
            return 1
        if args.epochs < 1:
            console_print("  ERROR: --epochs must be positive.")
            return 1
        resume_run_dir = Path(args.resume_run_dir).resolve()
        resume_config_path = resume_run_dir / "config.json"
        if not resume_config_path.exists():
            console_print(f"  ERROR: Resume run is missing config: {resume_config_path}")
            return 1
        with resume_config_path.open(encoding="utf-8") as handle:
            resume_config = json.load(handle)

        config["prepared_root"] = args.prepared_root or resume_config["prepared_root"]
        config["run_root"] = str(resume_run_dir.parent.parent)
        config["seed"] = int(resume_config.get("seed", config["seed"]))
        config["device"] = args.device or str(resume_config.get("device", config["device"]))
        config["gnn_type"] = str(resume_config.get("gnn_type", config["gnn_type"]))
        resume_stage = "pointer" if "max_args" in resume_config else "baseline"
        config["stages"] = [resume_stage]
        config[resume_stage] = {
            **config[resume_stage],
            "gnn_type": config["gnn_type"],
            "edge_mode": resume_config.get("edge_mode", "bidirectional"),
            "use_node_type": bool(resume_config.get("use_node_type", True)),
            "model": dict(resume_config.get("model", {})),
            "training": {
                **dict(resume_config.get("training", {})),
                "epochs": args.epochs,
            },
            "epochs": args.epochs,
        }
        if resume_stage == "pointer":
            config["pointer"].update({
                "max_args": int(resume_config.get("max_args", 3)),
                "arg_loss_weight": float(
                    resume_config.get("arg_loss_weight", 0.5)
                ),
                "initialization_checkpoint": resume_config.get(
                    "initialization_checkpoint"
                ),
            })
        # Resume configuration supplies compatibility-critical architecture,
        # then explicit CLI training overrides take final precedence.
        for key, value in explicit_stage_overrides.items():
            _apply_cli_override(config, key, value)

    # Print config and exit if dry run
    if args.dry_run:
        console_print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    # Print banner
    console_print("=" * 60)
    console_print("  GNN TRAINING PIPELINE")
    console_print("=" * 60)
    console_print(f"  Experiment : {config['experiment_name']}")
    console_print(f"  Stages     : {', '.join(config['stages'])}")
    console_print(f"  Seed       : {config['seed']}")
    console_print(f"  Device     : {config['device']}")
    console_print(f"  GNN type   : {config['gnn_type']}")
    console_print(f"  Prepared   : {config['prepared_root']}")
    console_print(f"  Runs       : {config['run_root']}")
    console_print("")

    # Load or create pipeline state
    pipeline_dir = Path(config["run_root"]) / config["experiment_name"]
    pipeline_state_path = pipeline_dir / "pipeline_state.json"
    state = PipelineState.load(pipeline_state_path)
    state.start_time = time.time()

    # Determine which stages to run
    all_stages = ["prepare", "baseline", "pointer", "scorer"]
    stages_to_run = config["stages"]
    if args.resume:
        stages_to_run = [s for s in stages_to_run if not state.is_complete(s)]
        if not stages_to_run:
            console_print("  All stages already complete. Use --stages to force re-run.")
            return 0
        console_print(f"  Resuming — will run: {', '.join(stages_to_run)}")

    results = {}
    for stage in stages_to_run:
        if stage not in all_stages:
            console_print(f"  WARNING: Unknown stage '{stage}', skipping.")
            continue

        stage_start = time.time()
        try:
            if stage == "prepare":
                results[stage] = run_prepare(config)
            elif stage == "baseline":
                resume_dir = args.resume_run_dir
                if resume_dir is None and args.resume and state.stage_outputs.get("baseline", {}).get("run_dir"):
                    resume_dir = state.stage_outputs["baseline"]["run_dir"]
                results[stage] = run_baseline(config, resume_run_dir=resume_dir)
            elif stage == "pointer":
                resume_dir = args.resume_run_dir
                if resume_dir is None and args.resume and state.stage_outputs.get("pointer", {}).get("run_dir"):
                    resume_dir = state.stage_outputs["pointer"]["run_dir"]
                results[stage] = run_pointer(config, resume_run_dir=resume_dir)
            elif stage == "scorer":
                results[stage] = run_scorer(config)

            state.mark_complete(stage, results[stage])
            elapsed = time.time() - stage_start
            console_print(f"\n  [DONE] {stage} completed in {elapsed:.1f}s\n")
        except Exception as exc:
            elapsed = time.time() - stage_start
            console_print(f"\n  [FAILED] {stage} failed after {elapsed:.1f}s: {exc}\n")
            import traceback
            traceback.print_exc()
            # Save state and exit
            state.save(pipeline_state_path)
            return 1

        # Save state after each stage
        state.save(pipeline_state_path)

    # Print summary
    total_elapsed = time.time() - state.start_time
    console_print("\n" + "=" * 60)
    console_print("  PIPELINE COMPLETE")
    console_print("=" * 60)
    console_print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}m)")
    console_print(f"  Stages: {', '.join(state.completed_stages)}")
    for stage, result in results.items():
        if "run_dir" in result:
            console_print(f"  {stage}: {result['run_dir']}")
        elif "summary" in result and "run_dir" in result["summary"]:
            console_print(f"  {stage}: {result['summary']['run_dir']}")
    console_print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
