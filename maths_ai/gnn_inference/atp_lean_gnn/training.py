from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, Sampler
from torch_geometric.loader import DataLoader

from .argument_training import (
    evaluate_model_with_args,
    train_one_epoch_with_args,
)
from .argument_selector import TacticWithArgsClassifier, TacticWithArgsConfig
from .dataset import CANONICAL_SPLITS, canonicalize_split_name
from .labels import UNKNOWN_TACTIC, get_tactic_arity
from .model import (
    GATv2ClassifierConfig,
    GATv2StateClassifier,
    GraphSAGEClassifierConfig,
    GraphSAGEStateClassifier,
    VALID_READOUTS,
)
from .pyg import NODE_TYPE_TO_ID
from .reporting import console_print
from torch_geometric.nn import DataParallel as PyGDataParallel


VALID_GNN_TYPES = ("sage", "gat")

DEFAULT_BASELINE_CONFIG_PATH = Path("configs") / "baseline_graphsage_state.json"
DEFAULT_POINTER_CONFIG_PATH = Path("configs") / "pointer_graphsage_state.json"
REQUIRED_DATA_FIELDS = ("x", "node_type", "edge_index", "y", "split", "row_index", "tactic_name")
REQUIRED_POINTER_DATA_FIELDS = REQUIRED_DATA_FIELDS + ("premise_mask", "arg_node_indices")


@dataclass(frozen=True)
class TrainingLoopConfig:
    batch_size: int = 32
    max_batch_nodes: int = 0
    max_batch_edges: int = 0
    oversize_graph_policy: str = "singleton"
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_every_batches: int = 100
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    use_amp: bool = True
    cache_in_memory: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "max_batch_nodes": self.max_batch_nodes,
            "max_batch_edges": self.max_batch_edges,
            "oversize_graph_policy": self.oversize_graph_policy,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "log_every_batches": self.log_every_batches,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "use_amp": self.use_amp,
            "cache_in_memory": self.cache_in_memory,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
        }


@dataclass(frozen=True)
class BaselineConfig:
    prepared_root: Path
    run_root: Path
    seed: int = 42
    device: str = "auto"
    edge_mode: str = "bidirectional"
    use_node_type: bool = True
    gnn_type: str = "sage"
    model: GraphSAGEClassifierConfig | GATv2ClassifierConfig = field(default_factory=GraphSAGEClassifierConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineConfig":
        if "prepared_root" not in payload:
            raise ValueError("Training config is missing the required 'prepared_root' field.")

        model_payload = payload.get("model", {})
        training_payload = payload.get("training", {})
        gnn_type = str(payload.get("gnn_type", "sage")).lower().strip()
        if gnn_type not in VALID_GNN_TYPES:
            raise ValueError(f"Training config field 'gnn_type' must be one of: {', '.join(VALID_GNN_TYPES)}.")
        return cls(
            prepared_root=Path(payload["prepared_root"]),
            run_root=Path(payload.get("run_root", "runs/baseline_gnn")),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "auto")),
            edge_mode=str(payload.get("edge_mode", "bidirectional")),
            use_node_type=bool(payload.get("use_node_type", True)),
            gnn_type=gnn_type,
            model=(GATv2ClassifierConfig(
                hidden_dim=int(model_payload.get("hidden_dim", 256)),
                num_layers=int(model_payload.get("num_layers", 4)),
                dropout=float(model_payload.get("dropout", 0.2)),
                heads=int(model_payload.get("heads", 8)),
                readout=str(model_payload.get("readout", "state")),
            ) if gnn_type == "gat" else GraphSAGEClassifierConfig(
                hidden_dim=int(model_payload.get("hidden_dim", 128)),
                num_layers=int(model_payload.get("num_layers", 4)),
                dropout=float(model_payload.get("dropout", 0.2)),
            )),
            training=TrainingLoopConfig(
                batch_size=int(training_payload.get("batch_size", 32)),
                max_batch_nodes=int(training_payload.get("max_batch_nodes", 0)),
                max_batch_edges=int(training_payload.get("max_batch_edges", 0)),
                oversize_graph_policy=str(training_payload.get("oversize_graph_policy", "singleton")).lower().strip(),
                epochs=int(training_payload.get("epochs", 20)),
                learning_rate=float(training_payload.get("learning_rate", 1e-3)),
                weight_decay=float(training_payload.get("weight_decay", 1e-4)),
                grad_clip=float(training_payload.get("grad_clip", 1.0)),
                log_every_batches=int(training_payload.get("log_every_batches", 100)),
                num_workers=int(training_payload.get("num_workers", 2)),
                pin_memory=bool(training_payload.get("pin_memory", True)),
                persistent_workers=bool(training_payload.get("persistent_workers", True)),
                prefetch_factor=int(training_payload.get("prefetch_factor", 2)),
                use_amp=bool(training_payload.get("use_amp", True)),
                cache_in_memory=bool(training_payload.get("cache_in_memory", False)),
                early_stopping_patience=int(training_payload.get("early_stopping_patience", 0)),
                early_stopping_min_delta=float(training_payload.get("early_stopping_min_delta", 0.0)),
            ),
        ).normalized()

    def normalized(self) -> "BaselineConfig":
        edge_mode = self.edge_mode.lower().strip()
        if edge_mode not in {"forward", "bidirectional"}:
            raise ValueError("Training config field 'edge_mode' must be either 'forward' or 'bidirectional'.")

        device = self.device.lower().strip()
        if not _is_valid_device(device):
            raise ValueError("Training config field 'device' must be one of: auto, cpu, cuda, or cuda:<index>.")

        gnn_type = self.gnn_type.lower().strip()
        if gnn_type not in VALID_GNN_TYPES:
            raise ValueError(f"Training config field 'gnn_type' must be one of: {', '.join(VALID_GNN_TYPES)}.")

        if self.model.hidden_dim < 1:
            raise ValueError("Training config field 'model.hidden_dim' must be positive.")
        if self.model.num_layers < 1:
            raise ValueError("Training config field 'model.num_layers' must be positive.")
        normalized_model = self.model
        if gnn_type == "gat":
            readout = self.model.readout.lower().strip()
            if readout not in VALID_READOUTS:
                raise ValueError(
                    "Training config field 'model.readout' must be one of: "
                    f"{', '.join(VALID_READOUTS)}."
                )
            normalized_model = GATv2ClassifierConfig(
                hidden_dim=self.model.hidden_dim,
                num_layers=self.model.num_layers,
                dropout=self.model.dropout,
                heads=self.model.heads,
                readout=readout,
            )
        if self.training.batch_size < 1:
            raise ValueError("Training config field 'training.batch_size' must be positive.")
        if self.training.max_batch_nodes < 0:
            raise ValueError("Training config field 'training.max_batch_nodes' cannot be negative.")
        if self.training.max_batch_edges < 0:
            raise ValueError("Training config field 'training.max_batch_edges' cannot be negative.")
        if self.training.oversize_graph_policy not in {"singleton", "skip", "error"}:
            raise ValueError(
                "Training config field 'training.oversize_graph_policy' must be one of: "
                "singleton, skip, error."
            )
        if self.training.epochs < 1:
            raise ValueError("Training config field 'training.epochs' must be positive.")
        if self.training.learning_rate <= 0:
            raise ValueError("Training config field 'training.learning_rate' must be positive.")
        if self.training.weight_decay < 0:
            raise ValueError("Training config field 'training.weight_decay' cannot be negative.")
        if self.training.grad_clip <= 0:
            raise ValueError("Training config field 'training.grad_clip' must be positive.")
        if self.training.log_every_batches < 1:
            raise ValueError("Training config field 'training.log_every_batches' must be positive.")
        if self.training.num_workers < 0:
            raise ValueError("Training config field 'training.num_workers' cannot be negative.")
        if self.training.prefetch_factor < 1:
            raise ValueError("Training config field 'training.prefetch_factor' must be positive.")
        if self.training.early_stopping_patience < 0:
            raise ValueError("Training config field 'training.early_stopping_patience' cannot be negative.")
        if self.training.early_stopping_min_delta < 0:
            raise ValueError("Training config field 'training.early_stopping_min_delta' cannot be negative.")

        return BaselineConfig(
            prepared_root=self.prepared_root.resolve(),
            run_root=self.run_root.resolve(),
            seed=self.seed,
            device=device,
            edge_mode=edge_mode,
            use_node_type=self.use_node_type,
            gnn_type=gnn_type,
            model=normalized_model,
            training=self.training,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prepared_root": str(self.prepared_root),
            "run_root": str(self.run_root),
            "seed": self.seed,
            "device": self.device,
            "edge_mode": self.edge_mode,
            "use_node_type": self.use_node_type,
            "gnn_type": self.gnn_type,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class PointerConfig:
    """Config for pointer-based argument selection model."""
    prepared_root: Path
    run_root: Path
    seed: int = 42
    device: str = "auto"
    edge_mode: str = "bidirectional"
    use_node_type: bool = True
    gnn_type: str = "sage"
    max_args: int = 3
    arg_loss_weight: float = 0.5
    initialization_checkpoint: Path | None = None
    model: TacticWithArgsConfig = field(default_factory=TacticWithArgsConfig)
    training: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PointerConfig":
        if "prepared_root" not in payload:
            raise ValueError("Training config is missing the required 'prepared_root' field.")

        model_payload = payload.get("model", {})
        training_payload = payload.get("training", {})
        gnn_type = str(payload.get("gnn_type", "sage")).lower().strip()
        if gnn_type not in VALID_GNN_TYPES:
            raise ValueError(f"Training config field 'gnn_type' must be one of: {', '.join(VALID_GNN_TYPES)}.")
        return cls(
            prepared_root=Path(payload["prepared_root"]),
            run_root=Path(payload.get("run_root", "runs/pointer_gnn")),
            seed=int(payload.get("seed", 42)),
            device=str(payload.get("device", "auto")),
            edge_mode=str(payload.get("edge_mode", "bidirectional")),
            use_node_type=bool(payload.get("use_node_type", True)),
            gnn_type=gnn_type,
            max_args=int(payload.get("max_args", 3)),
            arg_loss_weight=float(payload.get("arg_loss_weight", 0.5)),
            initialization_checkpoint=(
                Path(str(payload["initialization_checkpoint"]))
                if payload.get("initialization_checkpoint")
                else None
            ),
            model=TacticWithArgsConfig(
                hidden_dim=int(model_payload.get("hidden_dim", 128)),
                num_layers=int(model_payload.get("num_layers", 4)),
                dropout=float(model_payload.get("dropout", 0.2)),
                max_args=int(model_payload.get("max_args", 3)),
                arg_loss_weight=float(model_payload.get("arg_loss_weight", 0.5)),
                heads=int(model_payload.get("heads", 8)),
                readout=str(model_payload.get("readout", "state")),
                teacher_forcing=bool(model_payload.get("teacher_forcing", True)),
            ),
            training=TrainingLoopConfig(
                batch_size=int(training_payload.get("batch_size", 32)),
                max_batch_nodes=int(training_payload.get("max_batch_nodes", 0)),
                max_batch_edges=int(training_payload.get("max_batch_edges", 0)),
                oversize_graph_policy=str(training_payload.get("oversize_graph_policy", "singleton")).lower().strip(),
                epochs=int(training_payload.get("epochs", 20)),
                learning_rate=float(training_payload.get("learning_rate", 1e-3)),
                weight_decay=float(training_payload.get("weight_decay", 1e-4)),
                grad_clip=float(training_payload.get("grad_clip", 1.0)),
                log_every_batches=int(training_payload.get("log_every_batches", 100)),
                num_workers=int(training_payload.get("num_workers", 2)),
                pin_memory=bool(training_payload.get("pin_memory", True)),
                persistent_workers=bool(training_payload.get("persistent_workers", True)),
                prefetch_factor=int(training_payload.get("prefetch_factor", 2)),
                use_amp=bool(training_payload.get("use_amp", True)),
                cache_in_memory=bool(training_payload.get("cache_in_memory", False)),
                early_stopping_patience=int(training_payload.get("early_stopping_patience", 0)),
                early_stopping_min_delta=float(training_payload.get("early_stopping_min_delta", 0.0)),
            ),
        ).normalized()

    def normalized(self) -> "PointerConfig":
        edge_mode = self.edge_mode.lower().strip()
        if edge_mode not in {"forward", "bidirectional"}:
            raise ValueError("Training config field 'edge_mode' must be either 'forward' or 'bidirectional'.")

        device = self.device.lower().strip()
        if not _is_valid_device(device):
            raise ValueError("Training config field 'device' must be one of: auto, cpu, cuda, or cuda:<index>.")

        gnn_type = self.gnn_type.lower().strip()
        if gnn_type not in VALID_GNN_TYPES:
            raise ValueError(f"Training config field 'gnn_type' must be one of: {', '.join(VALID_GNN_TYPES)}.")

        if self.model.hidden_dim < 1:
            raise ValueError("Training config field 'model.hidden_dim' must be positive.")
        if self.model.num_layers < 1:
            raise ValueError("Training config field 'model.num_layers' must be positive.")
        if self.max_args < 1:
            raise ValueError("Training config field 'max_args' must be positive.")
        if self.arg_loss_weight < 0:
            raise ValueError("Training config field 'arg_loss_weight' cannot be negative.")
        readout = self.model.readout.lower().strip()
        if gnn_type == "gat" and readout not in VALID_READOUTS:
            raise ValueError(
                "Training config field 'model.readout' must be one of: "
                f"{', '.join(VALID_READOUTS)}."
            )
        if self.training.batch_size < 1:
            raise ValueError("Training config field 'training.batch_size' must be positive.")
        if self.training.max_batch_nodes < 0:
            raise ValueError("Training config field 'training.max_batch_nodes' cannot be negative.")
        if self.training.max_batch_edges < 0:
            raise ValueError("Training config field 'training.max_batch_edges' cannot be negative.")
        if self.training.oversize_graph_policy not in {"singleton", "skip", "error"}:
            raise ValueError(
                "Training config field 'training.oversize_graph_policy' must be one of: "
                "singleton, skip, error."
            )
        if self.training.epochs < 1:
            raise ValueError("Training config field 'training.epochs' must be positive.")
        if self.training.learning_rate <= 0:
            raise ValueError("Training config field 'training.learning_rate' must be positive.")
        if self.training.weight_decay < 0:
            raise ValueError("Training config field 'training.weight_decay' cannot be negative.")
        if self.training.early_stopping_patience < 0:
            raise ValueError("Training config field 'training.early_stopping_patience' cannot be negative.")
        if self.training.early_stopping_min_delta < 0:
            raise ValueError("Training config field 'training.early_stopping_min_delta' cannot be negative.")
        if self.training.grad_clip <= 0:
            raise ValueError("Training config field 'training.grad_clip' must be positive.")

        return PointerConfig(
            prepared_root=self.prepared_root.resolve(),
            run_root=self.run_root.resolve(),
            seed=self.seed,
            device=device,
            edge_mode=edge_mode,
            use_node_type=self.use_node_type,
            gnn_type=gnn_type,
            max_args=self.max_args,
            arg_loss_weight=self.arg_loss_weight,
            initialization_checkpoint=(
                self.initialization_checkpoint.expanduser().resolve()
                if self.initialization_checkpoint is not None
                else None
            ),
            model=TacticWithArgsConfig(
                hidden_dim=self.model.hidden_dim,
                num_layers=self.model.num_layers,
                dropout=self.model.dropout,
                max_args=self.model.max_args,
                arg_loss_weight=self.model.arg_loss_weight,
                heads=self.model.heads,
                readout=readout,
                teacher_forcing=self.model.teacher_forcing,
            ),
            training=self.training,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prepared_root": str(self.prepared_root),
            "run_root": str(self.run_root),
            "seed": self.seed,
            "device": self.device,
            "edge_mode": self.edge_mode,
            "use_node_type": self.use_node_type,
            "gnn_type": self.gnn_type,
            "max_args": self.max_args,
            "arg_loss_weight": self.arg_loss_weight,
            "initialization_checkpoint": (
                str(self.initialization_checkpoint)
                if self.initialization_checkpoint is not None
                else None
            ),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class PreparedMetadata:
    root: Path
    node_vocab: dict[str, int]
    tactic_vocab: dict[str, int]
    manifests: dict[str, dict[str, object]]
    state_label_id: int
    unknown_tactic_id: int

    @classmethod
    def from_vocabs(
        cls,
        *,
        node_vocab: dict[str, int],
        tactic_vocab: dict[str, int],
        root: Path | None = None,
    ) -> "PreparedMetadata":
        """Build metadata from vocabularies alone, with no dataset on disk.

        Inference needs only the two vocabularies: they fix the sizes of the
        label embedding and the tactic classifier, and both derived IDs below
        come from them.  ``manifests`` is read only by :meth:`split_manifest`
        and :meth:`split_pyg_dir`, which are training paths, so leaving it empty
        is what lets a published model be served without shipping the corpus it
        was trained on.
        """
        if "State" not in node_vocab:
            raise ValueError("Node vocab does not contain the required 'State' token.")
        if UNKNOWN_TACTIC not in tactic_vocab:
            raise ValueError(f"Tactic vocab does not contain '{UNKNOWN_TACTIC}'.")
        return cls(
            root=Path(root) if root is not None else Path("."),
            node_vocab=dict(node_vocab),
            tactic_vocab=dict(tactic_vocab),
            manifests={},
            state_label_id=node_vocab["State"],
            unknown_tactic_id=tactic_vocab[UNKNOWN_TACTIC],
        )

    def split_manifest(self, split: str) -> dict[str, object]:
        canonical_split = canonicalize_split_name(split)
        return self.manifests[canonical_split]

    def split_pyg_dir(self, split: str) -> Path:
        manifest = self.split_manifest(split)
        artifact_paths = manifest.get("artifact_paths", {})
        pyg_dir_rel = artifact_paths.get("pyg_dir")
        if not pyg_dir_rel:
            raise ValueError(f"Manifest for split '{split}' is missing 'artifact_paths.pyg_dir'.")
        return self.root / str(pyg_dir_rel)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _append_jsonl(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
    return path


def load_baseline_config(
    config_path: str | Path = DEFAULT_BASELINE_CONFIG_PATH,
    *,
    prepared_root_override: str | Path | None = None,
    run_root_override: str | Path | None = None,
    epochs_override: int | None = None,
    device_override: str | None = None,
    training_overrides: dict[str, Any] | None = None,
) -> BaselineConfig:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Training config file '{config_file}' does not exist.")

    payload = _read_json(config_file)
    if prepared_root_override is not None:
        payload["prepared_root"] = str(prepared_root_override)
    if run_root_override is not None:
        payload["run_root"] = str(run_root_override)
    if device_override is not None:
        payload["device"] = device_override
    if training_overrides:
        payload.setdefault("training", {}).update(training_overrides)
    if epochs_override is not None:
        payload.setdefault("training", {})["epochs"] = epochs_override
    return BaselineConfig.from_dict(payload)


def load_pointer_config(
    config_path: str | Path = DEFAULT_POINTER_CONFIG_PATH,
    *,
    prepared_root_override: str | Path | None = None,
    run_root_override: str | Path | None = None,
    epochs_override: int | None = None,
    device_override: str | None = None,
    training_overrides: dict[str, Any] | None = None,
) -> PointerConfig:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Training config file '{config_file}' does not exist.")

    payload = _read_json(config_file)
    if prepared_root_override is not None:
        payload["prepared_root"] = str(prepared_root_override)
    if run_root_override is not None:
        payload["run_root"] = str(run_root_override)
    if device_override is not None:
        payload["device"] = device_override
    if training_overrides:
        payload.setdefault("training", {}).update(training_overrides)
    if epochs_override is not None:
        payload.setdefault("training", {})["epochs"] = epochs_override
    return PointerConfig.from_dict(payload)


def load_prepared_metadata(
    prepared_root: str | Path,
    *,
    splits: Sequence[str] | None = None,
) -> PreparedMetadata:
    """Load vocabularies and split manifests from a prepared dataset root.

    Training requires all three canonical splits, which is the default.  Tools
    that legitimately inspect a single split -- an audit of a partially rebuilt
    corpus, for instance -- pass ``splits`` so a missing sibling manifest is not
    an error and no placeholder manifest has to be fabricated to satisfy one.
    """
    root = Path(prepared_root)
    if not root.exists():
        raise FileNotFoundError(f"Prepared dataset root '{root}' does not exist.")
    if not root.is_dir():
        raise FileNotFoundError(f"Prepared dataset root '{root}' is not a directory.")

    node_vocab_path = root / "vocab" / "node_vocab.json"
    tactic_vocab_path = root / "vocab" / "tactic_vocab.json"
    missing_paths = [path for path in (node_vocab_path, tactic_vocab_path) if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Prepared dataset is missing required vocab files: {missing_text}")

    node_vocab = {str(key): int(value) for key, value in _read_json(node_vocab_path).items()}
    tactic_vocab = {str(key): int(value) for key, value in _read_json(tactic_vocab_path).items()}

    if "State" not in node_vocab:
        raise ValueError(
            f"Prepared dataset node vocab '{node_vocab_path}' does not contain the required 'State' token."
        )
    if UNKNOWN_TACTIC not in tactic_vocab:
        raise ValueError(
            f"Prepared dataset tactic vocab '{tactic_vocab_path}' does not contain '{UNKNOWN_TACTIC}'."
        )

    manifests: dict[str, dict[str, object]] = {}
    required_splits = (
        CANONICAL_SPLITS
        if splits is None
        else tuple(canonicalize_split_name(split) for split in splits)
    )
    if not required_splits:
        raise ValueError("At least one split is required.")
    for split in required_splits:
        manifest_path = root / "manifests" / f"{split}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Prepared dataset is missing manifest '{manifest_path}'.")
        manifest = _read_json(manifest_path)
        artifact_paths = manifest.get("artifact_paths", {})
        pyg_dir_rel = artifact_paths.get("pyg_dir")
        if not pyg_dir_rel:
            raise ValueError(f"Manifest '{manifest_path}' is missing 'artifact_paths.pyg_dir'.")
        pyg_dir = root / str(pyg_dir_rel)
        if not pyg_dir.exists():
            raise FileNotFoundError(
                f"Manifest '{manifest_path}' points to missing PyG artifact directory '{pyg_dir}'."
            )
        manifests[split] = manifest

    return PreparedMetadata(
        root=root.resolve(),
        node_vocab=node_vocab,
        tactic_vocab=tactic_vocab,
        manifests=manifests,
        state_label_id=node_vocab["State"],
        unknown_tactic_id=tactic_vocab[UNKNOWN_TACTIC],
    )


def transform_edge_index(edge_index: torch.Tensor, *, edge_mode: str) -> torch.Tensor:
    if edge_mode == "forward":
        return edge_index.to(dtype=torch.long).contiguous()
    if edge_mode != "bidirectional":
        raise ValueError(f"Unsupported edge mode '{edge_mode}'.")
    if edge_index.numel() == 0:
        return edge_index.to(dtype=torch.long).contiguous()

    forward = edge_index.to(dtype=torch.long)
    reverse = forward[[1, 0], :]
    # Prepared graphs are DAGs whose edges were already deduplicated. They
    # cannot contain reciprocal pairs or self-loops, so concatenation is both
    # sufficient and much cheaper than sorting every graph with torch.unique
    # on every epoch.
    return torch.cat([forward, reverse], dim=1).contiguous()


def validate_prepared_data(data, *, path: Path, split: str, required_fields: tuple[str, ...]) -> None:
    missing = [field for field in required_fields if not hasattr(data, field)]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Prepared example '{path}' is missing required fields: {missing_text}")

    if not torch.is_tensor(data.x) or data.x.dim() != 1:
        raise ValueError(f"Prepared example '{path}' has an invalid 'x' tensor shape.")
    if not torch.is_tensor(data.node_type) or data.node_type.dim() != 1:
        raise ValueError(f"Prepared example '{path}' has an invalid 'node_type' tensor shape.")
    if not torch.is_tensor(data.edge_index) or data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValueError(f"Prepared example '{path}' has an invalid 'edge_index' tensor shape.")
    if not torch.is_tensor(data.y) or data.y.numel() != 1:
        raise ValueError(f"Prepared example '{path}' must store exactly one target label in 'y'.")
    if str(data.split) != split:
        raise ValueError(
            f"Prepared example '{path}' belongs to split '{data.split}', expected '{split}'."
        )


def infer_state_node_index(data, *, state_label_id: int, path: Path) -> torch.Tensor:
    state_matches = (data.x == state_label_id).nonzero(as_tuple=False).view(-1)
    if state_matches.numel() == 0:
        raise ValueError(
            f"Prepared example '{path}' does not contain the required 'State' node label."
        )

    source_nodes = {int(node_id) for node_id in data.edge_index[0].tolist()}
    root_candidates = [
        int(node_id)
        for node_id in state_matches.tolist()
        if int(node_id) not in source_nodes
    ]
    if len(root_candidates) == 1:
        return torch.tensor(root_candidates, dtype=torch.long)
    if state_matches.numel() == 1:
        return state_matches.to(dtype=torch.long)

    raise ValueError(
        f"Prepared example '{path}' must contain exactly one root 'State' node, "
        f"found {state_matches.numel()} 'State'-labeled nodes and {len(root_candidates)} root candidates."
    )


class PreparedGraphDataset(Dataset):
    def __init__(
        self,
        metadata: PreparedMetadata,
        *,
        split: str,
        edge_mode: str = "bidirectional",
        required_fields: tuple[str, ...] = REQUIRED_DATA_FIELDS,
        io_threads: int = 0,
        cache_in_memory: bool = False,
    ) -> None:
        self.metadata = metadata
        self.split = canonicalize_split_name(split)
        self.edge_mode = edge_mode
        self.required_fields = required_fields
        self.io_threads = max(0, int(io_threads))
        self.cache_in_memory = bool(cache_in_memory)
        self._thread_pool: ThreadPoolExecutor | None = None
        self.pyg_dir = metadata.split_pyg_dir(self.split)
        expected_count = int(metadata.split_manifest(self.split).get("success_count", 0))
        self._length = expected_count
        packed_available = self.cache_in_memory and self._packed_cache_is_complete()
        self.files = [] if packed_available else sorted(self.pyg_dir.glob("*.pt"))
        if not packed_available and not self.files:
            raise RuntimeError(
                f"Prepared split '{self.split}' has neither individual PyG examples "
                f"under '{self.pyg_dir}' nor a complete packed cache."
            )
        if not packed_available and expected_count != len(self.files):
            raise ValueError(
                f"Prepared split '{self.split}' manifest reports {expected_count} examples, "
                f"but '{self.pyg_dir}' contains {len(self.files)} '.pt' files."
            )
        self._cache = [None] * self._length if self.cache_in_memory else None
        self.packed_cache_loaded = False
        if self._cache is not None:
            self.packed_cache_loaded = self._load_packed_cache()

    def _packed_manifest_path(self) -> Path:
        return self.metadata.root / "packed" / self.edge_mode / "manifest.json"

    def _packed_cache_is_complete(self) -> bool:
        manifest_path = self._packed_manifest_path()
        if not manifest_path.exists():
            return False
        manifest = _read_json(manifest_path)
        split_payload = dict(manifest.get("splits", {})).get(self.split)
        if not isinstance(split_payload, dict):
            return False
        if int(split_payload.get("count", -1)) != self._length:
            return False
        chunk_names = split_payload.get("chunks", [])
        if not isinstance(chunk_names, list) or not chunk_names:
            return False
        packed_root = manifest_path.parent / self.split
        return all((packed_root / str(chunk_name)).exists() for chunk_name in chunk_names)

    def _load_packed_cache(self) -> bool:
        manifest_path = self._packed_manifest_path()
        if not manifest_path.exists():
            return False

        manifest = _read_json(manifest_path)
        split_payload = dict(manifest.get("splits", {})).get(self.split)
        if not isinstance(split_payload, dict):
            return False
        if int(split_payload.get("count", -1)) != len(self):
            return False

        chunk_names = split_payload.get("chunks", [])
        if not isinstance(chunk_names, list) or not chunk_names:
            return False

        packed_root = manifest_path.parent / self.split
        offset = 0
        for chunk_name in chunk_names:
            chunk_path = packed_root / str(chunk_name)
            if not chunk_path.exists():
                return False
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if not isinstance(chunk, list):
                raise ValueError(f"Packed graph chunk '{chunk_path}' must contain a list.")
            end = offset + len(chunk)
            if end > len(self._cache):
                raise ValueError(f"Packed graph chunk '{chunk_path}' exceeds the split size.")
            self._cache[offset:end] = chunk
            offset = end

        if offset != len(self._cache):
            raise ValueError(
                f"Packed cache for split '{self.split}' loaded {offset} examples, "
                f"expected {len(self._cache)}."
            )
        return True

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        if self._cache is not None and self._cache[index] is not None:
            return self._cache[index]

        if not self.files:
            raise RuntimeError(
                f"Packed cache entry {index} for split '{self.split}' was not loaded "
                "and individual PyG files are unavailable."
            )

        path = self.files[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        validate_prepared_data(data, path=path, split=self.split, required_fields=self.required_fields)

        data.x = data.x.to(dtype=torch.long)
        data.node_type = data.node_type.to(dtype=torch.long)
        if not hasattr(data, "state_node_index"):
            data.state_node_index = infer_state_node_index(
                data,
                state_label_id=self.metadata.state_label_id,
                path=path,
            )
        data.edge_index = transform_edge_index(data.edge_index, edge_mode=self.edge_mode)
        data.y = data.y.view(-1).to(dtype=torch.long)
        if self._cache is not None:
            self._cache[index] = data
        return data

    def __getitems__(self, indices: list[int]):
        """Load one batch concurrently when process workers cannot be used.

        PyTorch's map-style fetcher calls ``__getitems__`` with the complete
        batch of indices.  A small thread pool overlaps high-latency reads from
        network filesystems without allocating multiprocessing shared memory.
        """
        if self.io_threads <= 1 or len(indices) <= 1:
            return [self[index] for index in indices]
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.io_threads,
                thread_name_prefix=f"gnn-{self.split}-io",
            )
        return list(self._thread_pool.map(self.__getitem__, indices))

    def graph_sizes(self) -> list[tuple[int, int]]:
        """Return (nodes, edges) without reopening thousands of graph files."""
        if self._cache is None or any(data is None for data in self._cache):
            raise RuntimeError(
                "Node/edge-budget batching requires cache_in_memory=true and a complete "
                "packed cache. Build the packed cache before enabling max_batch_nodes or "
                "max_batch_edges."
            )
        return [
            (int(data.num_nodes), int(data.edge_index.size(1)))
            for data in self._cache
        ]


class GraphBudgetBatchSampler(Sampler[list[int]]):
    """Greedily batch graphs under graph-count, node, and edge limits."""

    def __init__(
        self,
        graph_sizes: list[tuple[int, int]],
        *,
        max_graphs: int,
        max_nodes: int = 0,
        max_edges: int = 0,
        oversize_policy: str = "singleton",
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.graph_sizes = graph_sizes
        self.max_graphs = int(max_graphs)
        self.max_nodes = int(max_nodes)
        self.max_edges = int(max_edges)
        self.oversize_policy = str(oversize_policy).lower().strip()
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.max_graphs < 1:
            raise ValueError("max_graphs must be positive.")
        if self.max_nodes < 0 or self.max_edges < 0:
            raise ValueError("Graph budgets cannot be negative.")
        if self.oversize_policy not in {"singleton", "skip", "error"}:
            raise ValueError("oversize_policy must be one of: singleton, skip, error.")
        self.oversize_indices = [
            index
            for index, (nodes, edges) in enumerate(self.graph_sizes)
            if (self.max_nodes > 0 and nodes > self.max_nodes)
            or (self.max_edges > 0 and edges > self.max_edges)
        ]
        if self.oversize_indices and self.oversize_policy == "error":
            max_nodes = max(self.graph_sizes[index][0] for index in self.oversize_indices)
            max_edges = max(self.graph_sizes[index][1] for index in self.oversize_indices)
            raise ValueError(
                f"{len(self.oversize_indices)} graphs exceed the configured batch budget "
                f"(largest: {max_nodes} nodes, {max_edges} edges)."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[int]]:
        indices = list(range(len(self.graph_sizes)))
        if self.oversize_policy == "skip":
            oversize = set(self.oversize_indices)
            indices = [index for index in indices if index not in oversize]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches: list[list[int]] = []
        batch: list[int] = []
        nodes = edges = 0
        for index in indices:
            graph_nodes, graph_edges = self.graph_sizes[index]
            exceeds = bool(batch) and (
                len(batch) >= self.max_graphs
                or (self.max_nodes > 0 and nodes + graph_nodes > self.max_nodes)
                or (self.max_edges > 0 and edges + graph_edges > self.max_edges)
            )
            if exceeds:
                batches.append(batch)
                batch = []
                nodes = edges = 0
            batch.append(index)
            nodes += graph_nodes
            edges += graph_edges
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return len(self._batches())


def _shm_bytes() -> int:
    """Return available shared-memory size in bytes (0 if unknown)."""
    try:
        stats = os.statvfs("/dev/shm")
        return stats.f_bavail * stats.f_frsize
    except OSError:
        return 0


def _safe_num_workers(requested: int, *, pin_memory: bool) -> tuple[int, str | None]:
    """Cap DataLoader workers to avoid exhausting ``/dev/shm``.

    PyTorch's DataLoader workers pass tensors between processes via shared
    memory; in containers with a tiny ``/dev/shm`` (common in Jupyter/Docker)
    this causes "Bus error ... out of shared memory". When ``/dev/shm`` is
    small we cap workers to a safe value and report a warning message.
    """
    if requested <= 0:
        return 0, None
    shm = _shm_bytes()
    # 256 MiB headroom per worker is a conservative floor for graph batches.
    safe = max(0, min(requested, (shm // (256 * 1024 * 1024)) if shm else requested))
    if safe == 0 and shm:
        return 0, (
            f"/dev/shm is too small ({shm // (1024 * 1024)} MiB); "
            "using num_workers=0 to avoid shared-memory exhaustion."
        )
    if safe < requested:
        return safe, (
            f"/dev/shm is small ({shm // (1024 * 1024)} MiB); "
            f"capped DataLoader workers to {safe} (requested {requested})."
        )
    return requested, None


def build_dataloaders(
    metadata: PreparedMetadata,
    config: BaselineConfig | PointerConfig,
    required_fields: tuple[str, ...] = REQUIRED_DATA_FIELDS,
) -> tuple[dict[str, PreparedGraphDataset], dict[str, DataLoader]]:
    requested_workers = config.training.num_workers
    num_workers, shm_warning = _safe_num_workers(
        requested_workers, pin_memory=config.training.pin_memory
    )
    if shm_warning:
        console_print(f"  [warn] {shm_warning}")
    use_workers = num_workers > 0
    io_threads = requested_workers if requested_workers > 0 and not use_workers else 0
    if io_threads:
        console_print(
            f"  [info] using {io_threads} in-process I/O threads as the "
            "shared-memory-safe DataLoader fallback."
        )
    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": config.training.pin_memory,
    }
    if use_workers:
        loader_kwargs["persistent_workers"] = config.training.persistent_workers
        loader_kwargs["prefetch_factor"] = config.training.prefetch_factor

    datasets = {
        split: PreparedGraphDataset(
            metadata,
            split=split,
            edge_mode=config.edge_mode,
            required_fields=required_fields,
            io_threads=io_threads,
            cache_in_memory=config.training.cache_in_memory,
        )
        for split in CANONICAL_SPLITS
    }
    use_graph_budget = bool(
        config.training.max_batch_nodes or config.training.max_batch_edges
    )
    if use_graph_budget:
        samplers = {
            split: GraphBudgetBatchSampler(
                dataset.graph_sizes(),
                max_graphs=config.training.batch_size,
                max_nodes=config.training.max_batch_nodes,
                max_edges=config.training.max_batch_edges,
                oversize_policy=config.training.oversize_graph_policy,
                shuffle=(split == "train"),
                seed=config.seed,
            )
            for split, dataset in datasets.items()
        }
        for split, sampler in samplers.items():
            if sampler.oversize_indices:
                largest_nodes = max(
                    sampler.graph_sizes[index][0] for index in sampler.oversize_indices
                )
                largest_edges = max(
                    sampler.graph_sizes[index][1] for index in sampler.oversize_indices
                )
                console_print(
                    f"  [warn] {split}: {len(sampler.oversize_indices)} oversized graphs; "
                    f"policy={sampler.oversize_policy}, largest={largest_nodes} nodes/"
                    f"{largest_edges} edges."
                )
        loaders = {
            split: DataLoader(dataset, batch_sampler=samplers[split], **loader_kwargs)
            for split, dataset in datasets.items()
        }
    else:
        loaders = {
            split: DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=(split == "train"),
                **loader_kwargs,
            )
            for split, dataset in datasets.items()
        }
    return datasets, loaders


def compute_eval_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    unknown_tactic_id: int,
) -> dict[str, float | int]:
    if logits.dim() != 2:
        raise ValueError("Expected logits to have shape [batch_size, num_classes].")
    if targets.dim() != 1:
        raise ValueError("Expected targets to have shape [batch_size].")
    if logits.size(0) != targets.size(0):
        raise ValueError("Logits and targets batch sizes do not match.")

    unknown_mask = targets == unknown_tactic_id
    known_mask = ~unknown_mask
    unknown_count = int(unknown_mask.sum().item())
    known_count = int(known_mask.sum().item())

    if known_count == 0:
        return {
            "known_label_count": 0,
            "unknown_label_excluded_count": unknown_count,
            "loss_sum": 0.0,
            "top1_correct": 0,
            "top5_correct": 0,
        }

    known_logits = logits[known_mask]
    known_targets = targets[known_mask]
    loss = F.cross_entropy(known_logits, known_targets)

    top1_predictions = known_logits.argmax(dim=1)
    top1_correct = int((top1_predictions == known_targets).sum().item())

    top_k = min(5, known_logits.size(1))
    topk_predictions = known_logits.topk(top_k, dim=1).indices
    top5_correct = int(
        (topk_predictions == known_targets.unsqueeze(1)).any(dim=1).sum().item()
    )

    return {
        "known_label_count": known_count,
        "unknown_label_excluded_count": unknown_count,
        "loss_sum": float(loss.item()) * known_count,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _is_valid_device(device_name: str) -> bool:
    """Accept ``auto``, ``cpu``, ``cuda``, and ``cuda:<index>``."""
    device_name = device_name.lower().strip()
    if device_name in {"auto", "cpu", "cuda"}:
        return True
    return device_name.startswith("cuda:") and device_name[5:].isdigit()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Training config requested CUDA, but no CUDA device is available.")
    return torch.device(device_name)


def primary_device(device: torch.device) -> torch.device:
    """Return the device that input batches should be placed on.

    When training spans multiple GPUs via :class:`torch.nn.DataParallel`, the
    batch must live on the module's primary device (``cuda:0``) and
    ``DataParallel`` scatters replicas across the remaining devices.
    """
    return device


def maybe_wrap_data_parallel(model: nn.Module, device: torch.device) -> tuple[nn.Module, torch.device, list[int]]:
    """Wrap ``model`` in :class:`torch.nn.DataParallel` when multiple GPUs exist.

    Returns the (possibly wrapped) model, the primary device batches must be
    moved to, and the list of CUDA device ids in use.  Single-GPU and CPU paths
    are returned unchanged so the rest of the training loop is unaffected.

    An explicit device index (e.g. ``cuda:0``) disables DataParallel even when
    several GPUs are present, which is the reliable choice in containers where
    NCCL or ``/dev/shm`` are restricted.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return model, device, [device.index or 0] if device.type == "cuda" else []
    if device.index is not None:
        # Explicit single-GPU request: skip DataParallel entirely.
        return model, device, [device.index]
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return model, device, [device.index or 0]
    device_ids = list(range(gpu_count))
    primary = torch.device(f"cuda:{device_ids[0]}")
    # PyG DataParallel splits a Batch along the graph dimension, preserving
    # edge_index connectivity (torch.nn.DataParallel would break it). It
    # expects a `List[Data]` (one graph per element), not a single `Batch`,
    # so wrap it to convert the loader's `Batch` into a data list first.
    model = PyGBatchDataParallel(model, device_ids=device_ids)
    return model, primary, device_ids


class PyGBatchDataParallel(nn.Module):
    """Thin wrapper exposing PyG ``DataParallel`` to a standard ``Batch`` input.

    :class:`torch_geometric.nn.DataParallel` iterates its input expecting a
    ``List[Data]`` (one graph per element). PyG's ``DataLoader`` instead yields
    a single ``Batch``; iterating a ``Batch`` yields ``(attr, value)`` tuples,
    which crashes inside ``DataParallel``. This wrapper converts a ``Batch`` into
    the expected data list, then lets ``DataParallel`` re-batch per device.

    If multi-GPU communication fails at runtime (e.g. NCCL "unhandled system
    error" in restricted containers), it transparently falls back to single-GPU
    execution so training is never hard-blocked by the environment.
    """

    def __init__(self, module: nn.Module, device_ids: list[int] | None = None) -> None:
        super().__init__()
        self.module = module
        self.device_ids = device_ids or [0]
        self.inner: nn.Module | None = None
        self._fallback = False

    def _ensure_inner(self) -> nn.Module:
        if self.inner is None and not self._fallback:
            if not torch.cuda.is_available():
                self._fallback = True
                self.inner = self.module
                return self.inner
            try:
                self.inner = PyGDataParallel(self.module, device_ids=self.device_ids)
            except Exception:
                self._fallback = True
                self.inner = self.module
        return self.inner

    def forward(self, batch) -> torch.Tensor:
        data_list = batch.to_data_list() if hasattr(batch, "to_data_list") else batch
        inner = self._ensure_inner()
        if self._fallback:
            # Permanently on single GPU: the bare module expects a Batch, not a
            # data list, so re-batch before calling it.
            from torch_geometric.data import Batch

            target = (
                torch.device(f"cuda:{self.device_ids[0]}")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            return inner(Batch.from_data_list(data_list).to(target))
        try:
            return inner(data_list)
        except RuntimeError as exc:
            if "NCCL" in str(exc) and not self._fallback:
                self._fallback = True
                self.inner = self.module
                primary = f"cuda:{self.device_ids[0]}"
                console_print(
                    "  [warn] multi-GPU communication (NCCL) failed; "
                    "falling back to single GPU. Set NCCL_SOCKET_IFNAME / "
                    "NCCL_P2P_DISABLE to enable both GPUs."
                )
                from torch_geometric.data import Batch

                return self.module(Batch.from_data_list(data_list).to(primary))
            raise


def build_baseline_model(metadata: PreparedMetadata, config: BaselineConfig) -> GraphSAGEStateClassifier | GATv2StateClassifier:
    common = dict(
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
        num_node_types=len(NODE_TYPE_TO_ID),
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        use_node_type=config.use_node_type,
    )
    if config.gnn_type == "gat":
        return GATv2StateClassifier(
            heads=config.model.heads,
            readout=config.model.readout,
            **common,
        )
    return GraphSAGEStateClassifier(**common)


def build_pointer_model(metadata: PreparedMetadata, config: PointerConfig) -> TacticWithArgsClassifier:
    return TacticWithArgsClassifier(
        num_node_labels=len(metadata.node_vocab),
        num_tactics=len(metadata.tactic_vocab),
        num_node_types=len(NODE_TYPE_TO_ID),
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        use_node_type=config.use_node_type,
        max_args=config.max_args,
        gnn_type=config.gnn_type,
        heads=config.model.heads,
        readout=config.model.readout,
        teacher_forcing=config.model.teacher_forcing,
    )


def detect_state_dict_model_type(state_dict: Mapping[str, Any]) -> str:
    """Return the architecture identity encoded by a checkpoint's weights.

    Neither ``BaselineConfig.to_dict`` nor ``PointerConfig.to_dict`` records the
    model type, so the weights are the only reliable witness.  A
    ``TacticWithArgsClassifier`` holds its baseline in ``self.backbone``, so that
    prefix is present in a pointer state dict and absent from a baseline one.
    """
    if any(str(key).startswith("argument_selector.gru.") for key in state_dict):
        return "pointer_gru"
    if any(str(key).startswith("backbone.") for key in state_dict):
        return "pointer"
    return "baseline"


def _stable_vocab_sha256(vocab: dict[str, int]) -> str:
    payload = json.dumps(
        vocab,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_pointer_from_baseline_checkpoint(
    model: TacticWithArgsClassifier,
    *,
    config: PointerConfig,
    metadata: PreparedMetadata,
    checkpoint_path: str | Path,
) -> dict[str, object]:
    """Initialize a pointer backbone and tactic embedding from a baseline.

    The transfer is deliberately strict.  A pointer may only inherit a
    baseline whose encoder, readout, and vocabularies are identical; otherwise
    a partial load could silently attach learned weights to different labels.
    The argument selector is not touched.
    """
    source_path = Path(checkpoint_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Pointer initialization checkpoint '{source_path}' does not exist."
        )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Pointer initialization checkpoint '{source_path}' is not a file."
        )

    checkpoint = _load_checkpoint(source_path, device=torch.device("cpu"))
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        sibling_config = source_path.parent / "config.json"
        if not sibling_config.exists():
            raise ValueError(
                "Baseline initialization checkpoint has no embedded config and "
                f"its run directory is missing '{sibling_config.name}'."
            )
        checkpoint_config = _read_json(sibling_config)

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Baseline initialization checkpoint '{source_path}' is missing "
            "a model_state_dict."
        )
    if detect_state_dict_model_type(state_dict) == "pointer":
        raise ValueError(
            f"Pointer initialization checkpoint '{source_path}' contains a pointer "
            "model, not a baseline model. Use --resume-run-dir to resume a pointer run."
        )

    try:
        baseline_config = BaselineConfig.from_dict(checkpoint_config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not load baseline config from initialization checkpoint "
            f"'{source_path}': {exc}"
        ) from exc

    incompatibilities: list[str] = []
    comparisons = {
        "gnn_type": (baseline_config.gnn_type, config.gnn_type),
        "edge_mode": (baseline_config.edge_mode, config.edge_mode),
        "use_node_type": (baseline_config.use_node_type, config.use_node_type),
        "model.hidden_dim": (
            baseline_config.model.hidden_dim,
            config.model.hidden_dim,
        ),
        "model.num_layers": (
            baseline_config.model.num_layers,
            config.model.num_layers,
        ),
        "model.dropout": (baseline_config.model.dropout, config.model.dropout),
    }
    if config.gnn_type == "gat":
        comparisons.update(
            {
                "model.heads": (
                    getattr(baseline_config.model, "heads", None),
                    config.model.heads,
                ),
                "model.readout": (
                    getattr(baseline_config.model, "readout", "state"),
                    config.model.readout,
                ),
            }
        )
    for field_name, (baseline_value, pointer_value) in comparisons.items():
        if baseline_value != pointer_value:
            incompatibilities.append(
                f"{field_name}: baseline={baseline_value!r}, pointer={pointer_value!r}"
            )
    if incompatibilities:
        raise ValueError(
            "Baseline checkpoint is architecture-incompatible with the pointer model: "
            + "; ".join(incompatibilities)
        )

    # Where the baseline's vocabularies come from decides how trustworthy this
    # check is.  A checkpoint that carries its own vocabularies states what the
    # weights were actually trained against; falling back to the prepared root
    # only reads what happens to sit at that path today, which may have been
    # rebuilt since.  Prefer the embedded copy, accept the path for checkpoints
    # written before they were embedded, and name which was used in any error so
    # a mismatch is attributable.
    embedded_node_vocab = checkpoint.get("node_vocab")
    embedded_tactic_vocab = checkpoint.get("tactic_vocab")
    if isinstance(embedded_node_vocab, dict) and isinstance(embedded_tactic_vocab, dict):
        baseline_node_vocab = {
            str(key): int(value) for key, value in embedded_node_vocab.items()
        }
        baseline_tactic_vocab = {
            str(key): int(value) for key, value in embedded_tactic_vocab.items()
        }
        vocab_source = f"the vocabularies embedded in '{source_path}'"
        baseline_metadata = PreparedMetadata.from_vocabs(
            node_vocab=baseline_node_vocab,
            tactic_vocab=baseline_tactic_vocab,
            root=baseline_config.prepared_root,
        )
    else:
        try:
            baseline_metadata = load_prepared_metadata(baseline_config.prepared_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                "Baseline checkpoint does not carry its own vocabularies, so they must "
                "be read from its prepared dataset, which is unavailable or invalid at "
                f"'{baseline_config.prepared_root}': {exc}"
            ) from exc
        vocab_source = (
            f"the prepared dataset at '{baseline_config.prepared_root}' (the checkpoint "
            "predates embedded vocabularies)"
        )
    if baseline_metadata.node_vocab != metadata.node_vocab:
        raise ValueError(
            f"Baseline and pointer node vocabularies differ, according to {vocab_source}; "
            "refusing to transfer label embeddings with incompatible token IDs."
        )
    if baseline_metadata.tactic_vocab != metadata.tactic_vocab:
        raise ValueError(
            f"Baseline and pointer tactic vocabularies differ, according to {vocab_source}; "
            "refusing to transfer the tactic classifier with incompatible class IDs."
        )

    baseline_model = build_baseline_model(baseline_metadata, baseline_config)
    try:
        baseline_model.load_state_dict(state_dict, strict=True)
        model.backbone.load_state_dict(baseline_model.state_dict(), strict=True)
    except RuntimeError as exc:
        raise ValueError(
            f"Baseline checkpoint weights do not match its declared architecture: {exc}"
        ) from exc

    with torch.no_grad():
        if model.tactic_embedding.weight.shape != baseline_model.classifier.weight.shape:
            raise ValueError(
                "Baseline classifier weights cannot initialize tactic embeddings: "
                f"classifier shape={tuple(baseline_model.classifier.weight.shape)}, "
                f"embedding shape={tuple(model.tactic_embedding.weight.shape)}."
            )
        model.tactic_embedding.weight.copy_(baseline_model.classifier.weight)

    return {
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": _file_sha256(source_path),
        "source_epoch": int(checkpoint.get("epoch", 0)),
        "source_prepared_root": str(baseline_config.prepared_root),
        "source_vocab_origin": (
            "checkpoint"
            if isinstance(checkpoint.get("node_vocab"), dict)
            else "prepared_root"
        ),
        "source_node_vocab_sha256": _stable_vocab_sha256(
            baseline_metadata.node_vocab
        ),
        "source_tactic_vocab_sha256": _stable_vocab_sha256(
            baseline_metadata.tactic_vocab
        ),
        "transferred_components": [
            "backbone.encoder",
            "backbone.readout",
            "backbone.tactic_classifier",
            "tactic_embedding_from_classifier_weights",
        ],
        "randomly_initialized_components": ["argument_selector"],
        "applied_this_invocation": True,
    }


def _amp_dtype(
    device: torch.device,
    config: BaselineConfig | PointerConfig,
) -> torch.dtype | None:
    if not config.training.use_amp or device.type != "cuda":
        return None
    # FP16 attention scores can overflow. Ampere GPUs support BF16, whose
    # exponent range matches FP32 and avoids that failure while retaining
    # Tensor Core acceleration.
    if getattr(config, "gnn_type", "sage") == "gat":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    return torch.float16


def _use_cuda_amp(device: torch.device, config: BaselineConfig | PointerConfig) -> bool:
    return _amp_dtype(device, config) is not None


def _should_log_batch(batch_index: int, total_batches: int, *, log_every_batches: int) -> bool:
    return (
        batch_index == 1
        or batch_index == total_batches
        or batch_index % log_every_batches == 0
    )


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _move_baseline_batch(
    model: nn.Module,
    batch,
    *,
    device: torch.device,
    pin_memory: bool,
):
    # PyG DataParallel expects a CPU List[Data] and performs its own balanced
    # scatter. Moving the Batch to cuda:0 first adds a GPU round-trip and makes
    # cuda:0 a transfer bottleneck.
    if isinstance(model, PyGBatchDataParallel) and not model._fallback:
        return batch
    return batch.to(
        device,
        non_blocking=(device.type == "cuda" and pin_memory),
    )


def train_one_epoch(
    model: GraphSAGEStateClassifier,
    loader: DataLoader,
    *,
    optimizer: AdamW,
    grad_scaler,
    device: torch.device,
    grad_clip: float,
    unknown_tactic_id: int,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
    amp_dtype: torch.dtype | None,
    pin_memory: bool,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_batches = len(loader)
    start_time = time.perf_counter()
    previous_step_end = start_time
    data_wait_seconds = 0.0

    console_print(
        f"  Starting epoch {epoch:02d}/{total_epochs:02d} "
        f"with {total_batches} train batches..."
    )

    for batch_index, batch in enumerate(loader, start=1):
        data_wait_seconds += time.perf_counter() - previous_step_end
        batch = _move_baseline_batch(
            model,
            batch,
            device=device,
            pin_memory=pin_memory,
        )
        targets = batch.y.view(-1).to(device, non_blocking=(device.type == "cuda"))
        if bool((targets == unknown_tactic_id).any()):
            raise ValueError("The train split contains '<UNK_TACTIC>' targets, which should never happen.")

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            logits = model(batch)
            loss = F.cross_entropy(logits, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss ({float(loss):.4g}) at batch {batch_index}. "
                "Training is unstable; reduce hidden_dim/heads or learning rate."
            )

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        batch_size = int(targets.numel())
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        if _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    train batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"avg_loss={total_loss / max(total_examples, 1):.4f} | "
                f"elapsed={elapsed}"
            )
        previous_step_end = time.perf_counter()

    return {
        "loss": total_loss / max(total_examples, 1),
        "example_count": total_examples,
        "data_wait_seconds": data_wait_seconds,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def evaluate_model(
    model: GraphSAGEStateClassifier,
    loader: DataLoader,
    *,
    device: torch.device,
    unknown_tactic_id: int,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    pin_memory: bool = False,
) -> dict[str, float | int]:
    model.eval()
    loss_sum = 0.0
    known_label_count = 0
    unknown_label_excluded_count = 0
    top1_correct = 0
    top5_correct = 0
    total_batches = len(loader)
    start_time = time.perf_counter()
    previous_step_end = start_time
    data_wait_seconds = 0.0

    if split_name is not None:
        console_print(f"  Evaluating {split_name} split ({total_batches} batches)...")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            data_wait_seconds += time.perf_counter() - previous_step_end
            batch = _move_baseline_batch(
                model,
                batch,
                device=device,
                pin_memory=pin_memory,
            )
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(batch)
            targets = batch.y.view(-1).to(device, non_blocking=(device.type == "cuda"))
            batch_metrics = compute_eval_metrics_from_logits(
                logits,
                targets,
                unknown_tactic_id=unknown_tactic_id,
            )
            loss_sum += float(batch_metrics["loss_sum"])
            known_label_count += int(batch_metrics["known_label_count"])
            unknown_label_excluded_count += int(batch_metrics["unknown_label_excluded_count"])
            top1_correct += int(batch_metrics["top1_correct"])
            top5_correct += int(batch_metrics["top5_correct"])

            if (
                split_name is not None
                and log_every_batches is not None
                and _should_log_batch(batch_index, total_batches, log_every_batches=log_every_batches)
            ):
                elapsed = _format_elapsed(time.perf_counter() - start_time)
                console_print(
                    f"    {split_name} batch {batch_index:>5}/{total_batches} | "
                    f"known={known_label_count} | "
                    f"excluded={unknown_label_excluded_count} | "
                    f"elapsed={elapsed}"
                )
            previous_step_end = time.perf_counter()

    top1 = top1_correct / known_label_count if known_label_count else 0.0
    top5 = top5_correct / known_label_count if known_label_count else 0.0
    loss = loss_sum / known_label_count if known_label_count else 0.0

    return {
        "loss": loss,
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "known_label_count": known_label_count,
        "unknown_label_excluded_count": unknown_label_excluded_count,
        "evaluated_count": known_label_count + unknown_label_excluded_count,
        "data_wait_seconds": data_wait_seconds,
        "elapsed_seconds": time.perf_counter() - start_time,
    }


def _create_run_dir(run_root: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = run_root / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping DataParallel wrappers."""
    if isinstance(model, PyGBatchDataParallel):
        model = model.module
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model


def _save_checkpoint(
    path: Path,
    *,
    model: GraphSAGEStateClassifier | TacticWithArgsClassifier,
    optimizer: AdamW,
    config: BaselineConfig | PointerConfig,
    metadata: PreparedMetadata,
    epoch: int,
    val_metrics: dict[str, float | int],
) -> Path:
    """Write a resumable checkpoint that also carries its own vocabularies.

    Weights alone are not a model.  Node IDs index the label embedding and
    tactic IDs index the classifier, and both are assigned by position in a
    sorted set, so weights are meaningless without the exact mapping they were
    trained against.  A config records the prepared root only as a path, which
    is worthless once the checkpoint moves to another machine and dangerous if
    that path now holds a rebuilt corpus: a vocabulary of the same length but a
    different order loads cleanly and mislabels every prediction.  Embedding the
    vocabularies and their hashes makes the checkpoint self-describing and makes
    that mismatch detectable instead of silent.
    """
    torch.save(
        {
            "epoch": epoch,
            "config": config.to_dict(),
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "node_vocab": metadata.node_vocab,
            "tactic_vocab": metadata.tactic_vocab,
            "node_vocab_sha256": _stable_vocab_sha256(metadata.node_vocab),
            "tactic_vocab_sha256": _stable_vocab_sha256(metadata.tactic_vocab),
        },
        path,
    )
    return path


def _load_checkpoint(path: Path, *, device: torch.device) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    return torch.load(path, map_location=device, weights_only=False)


def _write_eval_file(run_dir: Path, *, split: str, metrics: dict[str, object]) -> Path:
    return _write_json(run_dir / f"eval_{split}.json", metrics)


def train_baseline(
    config: BaselineConfig,
    *,
    resume_run_dir: str | Path | None = None,
) -> dict[str, object]:
    metadata = load_prepared_metadata(config.prepared_root)
    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_dtype = _amp_dtype(device, config)
    use_amp = amp_dtype is not None
    if config.gnn_type == "gat" and config.training.use_amp and amp_dtype is None and device.type == "cuda":
        console_print("  AMP disabled for GATv2 because BF16 is unavailable; FP16 attention is unsafe.")
    datasets, loaders = build_dataloaders(metadata, config, required_fields=REQUIRED_DATA_FIELDS)
    if resume_run_dir is None:
        run_dir = _create_run_dir(config.run_root)
        config_path = _write_json(run_dir / "config.json", config.to_dict())
        start_epoch = 1
        best_epoch = 0
        best_val_top1 = -1.0
    else:
        run_dir = Path(resume_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' does not exist.")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run path '{run_dir}' is not a directory.")
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' is missing 'config.json'.")
        start_epoch = 1
        best_epoch = 0
        best_val_top1 = -1.0

    oversize_report = {"policy": config.training.oversize_graph_policy, "splits": {}}
    effective_dataset_sizes: dict[str, int] = {}
    for split, dataset in datasets.items():
        sampler = getattr(loaders[split], "batch_sampler", None)
        indices = list(getattr(sampler, "oversize_indices", []))
        records = [
            {
                "dataset_index": index,
                "num_nodes": sampler.graph_sizes[index][0],
                "num_edges": sampler.graph_sizes[index][1],
            }
            for index in indices
        ]
        skipped_count = len(indices) if config.training.oversize_graph_policy == "skip" else 0
        effective_dataset_sizes[split] = len(dataset) - skipped_count
        oversize_report["splits"][split] = {
            "oversize_count": len(indices),
            "skipped_count": skipped_count,
            "records": records,
        }
    oversize_report_path = _write_json(run_dir / "oversize_graphs.json", oversize_report)

    metrics_path = run_dir / "metrics.jsonl"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    model = build_baseline_model(metadata, config)
    model, device, gpu_ids = maybe_wrap_data_parallel(model, device)
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(
        device.type,
        enabled=(amp_dtype == torch.float16),
    )

    if resume_run_dir is not None:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume run directory '{run_dir}' is missing 'last.pt', so training cannot resume."
            )
        last_checkpoint = _load_checkpoint(last_checkpoint_path, device=device)
        _unwrap_model(model).load_state_dict(last_checkpoint["model_state_dict"])
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        if best_checkpoint_path.exists():
            best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
            best_epoch = int(best_checkpoint["epoch"])
            best_val_top1 = float(
                dict(best_checkpoint.get("val_metrics", {})).get("top1_accuracy", -1.0)
            )
        if config.training.epochs < start_epoch:
            raise ValueError(
                f"Resume target is {config.training.epochs} total epochs, but checkpoint "
                f"'{last_checkpoint_path}' already completed epoch {start_epoch - 1}. "
                f"Choose --epochs {start_epoch} or greater to extend the run."
            )
        # Persist the extended total so later resumes and analyses see the
        # effective configuration rather than the original epoch limit.
        _write_json(config_path, config.to_dict())

    epochs_without_improvement = (
        max(0, start_epoch - 1 - best_epoch)
        if resume_run_dir is not None and best_epoch > 0
        else 0
    )
    last_completed_epoch = start_epoch - 1

    console_print(f"\n  Training baseline run in: {run_dir}")
    console_print(f"  Prepared cache           : {config.prepared_root}")
    console_print(f"  Device                   : {device}" + (f" (DataParallel over GPUs {gpu_ids})" if len(gpu_ids) > 1 else ""))
    precision = str(amp_dtype).removeprefix("torch.") if amp_dtype is not None else "float32"
    console_print(f"  Compute precision        : {precision}")
    console_print(f"  Readout                  : {getattr(config.model, 'readout', 'state')}")
    console_print(
        f"  Split sizes              : train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    console_print(
        f"  DataLoader settings      : max_graphs={config.training.batch_size}, "
        f"max_nodes={config.training.max_batch_nodes or 'unlimited'}, "
        f"max_edges={config.training.max_batch_edges or 'unlimited'}, "
        f"oversize_policy={config.training.oversize_graph_policy}, "
        f"process_workers={loaders['train'].num_workers}, "
        f"io_threads={datasets['train'].io_threads}, "
        f"pin_memory={config.training.pin_memory}, "
        f"persistent_workers={getattr(loaders['train'], 'persistent_workers', False)}, "
        f"cache_in_memory={config.training.cache_in_memory}, "
        f"packed_cache={datasets['train'].packed_cache_loaded}"
    )
    if resume_run_dir is not None:
        console_print(
            f"  Resuming from checkpoint : {last_checkpoint_path} "
            f"(next epoch {start_epoch})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        last_completed_epoch = epoch
        batch_sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(
            model,
            loaders["train"],
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            grad_clip=config.training.grad_clip,
            unknown_tactic_id=metadata.unknown_tactic_id,
            epoch=epoch,
            total_epochs=config.training.epochs,
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        )
        val_metrics = evaluate_model(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_example_count": int(train_metrics["example_count"]),
            "train_data_wait_seconds": float(train_metrics["data_wait_seconds"]),
            "train_elapsed_seconds": float(train_metrics["elapsed_seconds"]),
            "val_loss": float(val_metrics["loss"]),
            "val_top1": float(val_metrics["top1_accuracy"]),
            "val_top5": float(val_metrics["top5_accuracy"]),
            "known_label_eval_count": int(val_metrics["known_label_count"]),
            "unknown_label_excluded_count": int(val_metrics["unknown_label_excluded_count"]),
            "val_data_wait_seconds": float(val_metrics["data_wait_seconds"]),
            "val_elapsed_seconds": float(val_metrics["elapsed_seconds"]),
        }
        _append_jsonl(metrics_path, epoch_record)

        _save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            metadata=metadata,
            epoch=epoch,
            val_metrics=val_metrics,
        )
        improved = (
            float(val_metrics["top1_accuracy"])
            > best_val_top1 + config.training.early_stopping_min_delta
        )
        if improved:
            best_val_top1 = float(val_metrics["top1_accuracy"])
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=metadata,
                epoch=epoch,
                val_metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1

        console_print(
            f"  Epoch {epoch:02d}/{config.training.epochs:02d} | "
            f"train_loss={epoch_record['train_loss']:.4f} | "
            f"val_loss={epoch_record['val_loss']:.4f} | "
            f"val_top1={epoch_record['val_top1']:.4f} | "
            f"val_top5={epoch_record['val_top5']:.4f} | "
            f"data_wait={_format_elapsed(epoch_record['train_data_wait_seconds'])} | "
            f"known={epoch_record['known_label_eval_count']} | "
            f"excluded={epoch_record['unknown_label_excluded_count']}"
        )

        if (
            config.training.early_stopping_patience > 0
            and epochs_without_improvement >= config.training.early_stopping_patience
        ):
            console_print(
                "  Early stopping: validation top-1 did not improve for "
                f"{epochs_without_improvement} epochs (best epoch {best_epoch})."
            )
            break

    best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
    _unwrap_model(model).load_state_dict(best_checkpoint["model_state_dict"])

    eval_val = {
        "split": "val",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        ),
    }
    eval_test = {
        "split": "test",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model(
            model,
            loaders["test"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name="test",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        ),
    }
    _write_eval_file(run_dir, split="val", metrics=eval_val)
    _write_eval_file(run_dir, split="test", metrics=eval_test)

    summary = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "prepared_root": str(config.prepared_root),
        "device": str(device),
        "amp_enabled": use_amp,
        "amp_dtype": precision,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "effective_dataset_sizes": effective_dataset_sizes,
        "oversize_graph_report": str(oversize_report_path),
        "start_epoch": start_epoch,
        "last_completed_epoch": last_completed_epoch,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "resumed_from_checkpoint": resume_run_dir is not None,
        "best_validation": eval_val,
        "test_evaluation": eval_test,
    }
    _write_json(run_dir / "summary.json", summary)

    console_print(f"\n  Best checkpoint          : {best_checkpoint_path}")
    console_print(f"  Validation eval summary  : {run_dir / 'eval_val.json'}")
    console_print(f"  Test eval summary        : {run_dir / 'eval_test.json'}")
    console_print(f"  Training summary         : {run_dir / 'summary.json'}")

    return summary


def train_pointer(
    config: PointerConfig,
    *,
    resume_run_dir: str | Path | None = None,
) -> dict[str, object]:
    """Train pointer-based argument selection model."""
    metadata = load_prepared_metadata(config.prepared_root)
    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_dtype = _amp_dtype(device, config)
    use_amp = amp_dtype is not None
    datasets, loaders = build_dataloaders(metadata, config, required_fields=REQUIRED_POINTER_DATA_FIELDS)
    
    if resume_run_dir is None:
        run_dir = _create_run_dir(config.run_root)
        config_path = _write_json(run_dir / "config.json", config.to_dict())
        start_epoch = 1
        best_epoch = 0
        best_val_loss = float("inf")
    else:
        run_dir = Path(resume_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' does not exist.")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run path '{run_dir}' is not a directory.")
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Resume run directory '{run_dir}' is missing 'config.json'.")
        start_epoch = 1
        best_epoch = 0
        best_val_loss = float("inf")

    metrics_path = run_dir / "metrics.jsonl"
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"

    model = build_pointer_model(metadata, config)
    initialization_details: dict[str, object] | None = None
    if resume_run_dir is None and config.initialization_checkpoint is not None:
        initialization_details = initialize_pointer_from_baseline_checkpoint(
            model,
            config=config,
            metadata=metadata,
            checkpoint_path=config.initialization_checkpoint,
        )
        persisted_config = config.to_dict()
        persisted_config["pointer_initialization"] = initialization_details
        config_path = _write_json(config_path, persisted_config)
    elif resume_run_dir is not None:
        previous_summary_path = run_dir / "summary.json"
        if previous_summary_path.exists():
            previous_summary = _read_json(previous_summary_path)
            previous_initialization = previous_summary.get("pointer_initialization")
            if isinstance(previous_initialization, dict):
                initialization_details = dict(previous_initialization)
                initialization_details["applied_this_invocation"] = False
        else:
            run_config = _read_json(config_path)
            config_initialization = run_config.get("pointer_initialization")
            if isinstance(config_initialization, dict):
                initialization_details = dict(config_initialization)
                initialization_details["applied_this_invocation"] = False
            elif config.initialization_checkpoint is not None:
                initialization_details = {
                    "source_checkpoint": str(config.initialization_checkpoint),
                    "applied_this_invocation": False,
                    "provenance_recovered_from_config_only": True,
                }
    # Pointer outputs have replica-dependent padded widths, so PyG DataParallel
    # cannot gather them safely. Keep pointer training on one explicitly selected GPU.
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
        console_print(
            "  [warn] pointer training is single-GPU; use --device cuda:<index>."
        )
    gpu_ids = [device.index or 0] if device.type == "cuda" else []
    model = model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(
        device.type,
        enabled=(amp_dtype == torch.float16),
    )

    if resume_run_dir is not None:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume run directory '{run_dir}' is missing 'last.pt', so training cannot resume."
            )
        last_checkpoint = _load_checkpoint(last_checkpoint_path, device=device)
        _unwrap_model(model).load_state_dict(last_checkpoint["model_state_dict"])
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        start_epoch = int(last_checkpoint["epoch"]) + 1
        if best_checkpoint_path.exists():
            best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
            best_epoch = int(best_checkpoint["epoch"])
            best_val_loss = float(
                dict(best_checkpoint.get("val_metrics", {})).get("combined_loss", float("inf"))
            )

    epochs_without_improvement = 0
    last_completed_epoch = start_epoch - 1

    console_print(f"\n  Training pointer run in  : {run_dir}")
    console_print(f"  Prepared cache           : {config.prepared_root}")
    console_print(f"  Device                   : {device}" + (f" (DataParallel over GPUs {gpu_ids})" if len(gpu_ids) > 1 else ""))
    precision = str(amp_dtype).removeprefix("torch.") if amp_dtype is not None else "float32"
    console_print(f"  Compute precision        : {precision}")
    console_print(
        f"  Split sizes              : train={len(datasets['train'])}, "
        f"val={len(datasets['val'])}, test={len(datasets['test'])}"
    )
    console_print(
        f"  DataLoader settings      : batch_size={config.training.batch_size}, "
        f"process_workers={loaders['train'].num_workers}, "
        f"io_threads={datasets['train'].io_threads}, "
        f"pin_memory={config.training.pin_memory}, "
        f"persistent_workers={getattr(loaders['train'], 'persistent_workers', False)}, "
        f"cache_in_memory={config.training.cache_in_memory}, "
        f"packed_cache={datasets['train'].packed_cache_loaded}"
    )
    console_print(f"  Max args per step        : {config.max_args}")
    console_print(f"  Argument loss weight     : {config.arg_loss_weight}")
    if initialization_details is not None:
        console_print(
            "  Baseline initialization  : "
            f"{initialization_details.get('source_checkpoint')}"
        )
        if initialization_details.get("applied_this_invocation"):
            console_print(
                "  Transferred components   : encoder, readout, tactic classifier, "
                "tactic embeddings"
            )
    if resume_run_dir is not None:
        console_print(
            f"  Resuming from checkpoint : {last_checkpoint_path} "
            f"(next epoch {start_epoch})"
        )

    for epoch in range(start_epoch, config.training.epochs + 1):
        last_completed_epoch = epoch
        batch_sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch_with_args(
            model,
            loaders["train"],
            optimizer=optimizer,
            grad_scaler=grad_scaler,
            device=device,
            grad_clip=config.training.grad_clip,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            epoch=epoch,
            total_epochs=config.training.epochs,
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        )
        val_metrics = evaluate_model_with_args(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        )

        epoch_record = {
            "epoch": epoch,
            "train_tactic_loss": float(train_metrics["tactic_loss"]),
            "train_arg_loss": float(train_metrics["arg_loss"]),
            "train_combined_loss": float(train_metrics["combined_loss"]),
            "train_example_count": int(train_metrics["example_count"]),
            "val_tactic_loss": float(val_metrics["tactic_loss"]),
            "val_arg_loss": float(val_metrics["arg_loss"]),
            "val_combined_loss": float(val_metrics["combined_loss"]),
            "val_tactic_accuracy": float(val_metrics["tactic_top1_accuracy"]),
            "val_tactic_top5_accuracy": float(val_metrics["tactic_top5_accuracy"]),
            "val_arg_accuracy": float(val_metrics["arg_top1_accuracy"]),
            "val_arg_top5_accuracy": float(val_metrics["arg_top5_accuracy"]),
            "val_arg_valid_count": int(val_metrics["arg_valid_count"]),
            "val_arg_target_count": int(val_metrics["arg_target_count"]),
            "val_arg_target_coverage": float(val_metrics["arg_target_coverage"]),
            "known_label_eval_count": int(val_metrics["known_label_count"]),
        }
        _append_jsonl(metrics_path, epoch_record)

        _save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            metadata=metadata,
            epoch=epoch,
            val_metrics=val_metrics,
        )
        improved = (
            float(val_metrics["combined_loss"])
            < best_val_loss - config.training.early_stopping_min_delta
        )
        if improved:
            best_val_loss = float(val_metrics["combined_loss"])
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                metadata=metadata,
                epoch=epoch,
                val_metrics=val_metrics,
            )
        else:
            epochs_without_improvement += 1

        console_print(
            f"  Epoch {epoch:02d}/{config.training.epochs:02d} | "
            f"train_loss={epoch_record['train_combined_loss']:.4f} | "
            f"val_loss={epoch_record['val_combined_loss']:.4f} | "
            f"val_tactic_acc={epoch_record['val_tactic_accuracy']:.4f} | "
            f"val_arg_acc={epoch_record['val_arg_accuracy']:.4f} | "
            f"known={epoch_record['known_label_eval_count']}"
        )

        if (
            config.training.early_stopping_patience > 0
            and epochs_without_improvement >= config.training.early_stopping_patience
        ):
            console_print(
                "  Early stopping: validation loss did not improve for "
                f"{epochs_without_improvement} epochs (best epoch {best_epoch})."
            )
            break

    best_checkpoint = _load_checkpoint(best_checkpoint_path, device=device)
    _unwrap_model(model).load_state_dict(best_checkpoint["model_state_dict"])

    eval_val = {
        "split": "val",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model_with_args(
            model,
            loaders["val"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            split_name="val",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        ),
    }
    eval_test = {
        "split": "test",
        "checkpoint": str(best_checkpoint_path),
        "epoch": int(best_checkpoint["epoch"]),
        **evaluate_model_with_args(
            model,
            loaders["test"],
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            arg_loss_weight=config.arg_loss_weight,
            split_name="test",
            log_every_batches=config.training.log_every_batches,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        ),
    }
    _write_eval_file(run_dir, split="val", metrics=eval_val)
    _write_eval_file(run_dir, split="test", metrics=eval_test)

    summary = {
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "prepared_root": str(config.prepared_root),
        "device": str(device),
        "amp_enabled": use_amp,
        "amp_dtype": precision,
        "dataset_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "start_epoch": start_epoch,
        "last_completed_epoch": last_completed_epoch,
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "resumed_from_checkpoint": resume_run_dir is not None,
        "pointer_initialization": initialization_details,
        "best_validation": eval_val,
        "test_evaluation": eval_test,
    }
    _write_json(run_dir / "summary.json", summary)

    console_print(f"\n  Best checkpoint          : {best_checkpoint_path}")
    console_print(f"  Validation eval summary  : {run_dir / 'eval_val.json'}")
    console_print(f"  Test eval summary        : {run_dir / 'eval_test.json'}")
    console_print(f"  Training summary         : {run_dir / 'summary.json'}")

    return summary


def evaluate_baseline_run(run_dir: str | Path, *, split: str) -> dict[str, object]:
    run_directory = Path(run_dir)
    if not run_directory.exists():
        raise FileNotFoundError(f"Run directory '{run_directory}' does not exist.")
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run path '{run_directory}' is not a directory.")

    config_path = run_directory / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run directory '{run_directory}' is missing '{config_path.name}'.")

    config = load_baseline_config(config_path)
    metadata = load_prepared_metadata(config.prepared_root)
    device = resolve_device(config.device)
    model = build_baseline_model(metadata, config).to(device)
    amp_dtype = _amp_dtype(device, config)
    checkpoint_path = run_directory / "best.pt"
    checkpoint = _load_checkpoint(checkpoint_path, device=device)
    _unwrap_model(model).load_state_dict(checkpoint["model_state_dict"])

    canonical_split = canonicalize_split_name(split)
    if canonical_split not in {"val", "test"}:
        raise ValueError("Evaluation split must be either 'val' or 'test'.")

    dataset = PreparedGraphDataset(metadata, split=canonical_split, edge_mode=config.edge_mode)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)
    metrics = {
        "split": canonical_split,
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        **evaluate_model(
            model,
            loader,
            device=device,
            unknown_tactic_id=metadata.unknown_tactic_id,
            split_name=canonical_split,
            log_every_batches=config.training.log_every_batches,
            use_amp=amp_dtype is not None,
            amp_dtype=amp_dtype,
            pin_memory=config.training.pin_memory,
        ),
    }
    _write_eval_file(run_directory, split=canonical_split, metrics=metrics)
    console_print(f"  Wrote evaluation summary : {run_directory / f'eval_{canonical_split}.json'}")
    return metrics


def build_train_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a GNN model from a prepared artifact cache")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["baseline", "pointer"],
        default="baseline",
        help="Which model type to train (baseline GraphSAGE or pointer argument selector)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the training JSON config (defaults to baseline or pointer config)",
    )
    parser.add_argument(
        "--prepared-root",
        type=str,
        default=None,
        help="Optional override for the prepared artifact root",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default=None,
        help="Optional override for the run output root",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default=None,
        help="Resume an interrupted run from its existing run directory and last checkpoint",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for the number of training epochs",
    )
    return parser


def build_evaluate_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the saved baseline GNN checkpoint")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a completed training run directory")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["val", "test"],
        help="Which split to evaluate with the best checkpoint",
    )
    return parser


def train_main(argv: list[str] | None = None) -> int:
    parser = build_train_arg_parser()
    args = parser.parse_args(argv)

    try:
        model_type = args.model_type.lower()
        
        if model_type == "baseline":
            config_path = args.config or DEFAULT_BASELINE_CONFIG_PATH
            if args.resume_run_dir:
                resume_config_path = Path(args.resume_run_dir) / "config.json"
                config = load_baseline_config(resume_config_path, epochs_override=args.epochs)
                train_baseline(config, resume_run_dir=args.resume_run_dir)
            else:
                config = load_baseline_config(
                    config_path,
                    prepared_root_override=args.prepared_root,
                    run_root_override=args.run_root,
                    epochs_override=args.epochs,
                )
                train_baseline(config)
        elif model_type == "pointer":
            config_path = args.config or DEFAULT_POINTER_CONFIG_PATH
            if args.resume_run_dir:
                resume_config_path = Path(args.resume_run_dir) / "config.json"
                config = load_pointer_config(resume_config_path, epochs_override=args.epochs)
                train_pointer(config, resume_run_dir=args.resume_run_dir)
            else:
                config = load_pointer_config(
                    config_path,
                    prepared_root_override=args.prepared_root,
                    run_root_override=args.run_root,
                    epochs_override=args.epochs,
                )
                train_pointer(config)
        else:
            console_print(f"  ERROR: Unknown model type '{model_type}'")
            return 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0


def evaluate_main(argv: list[str] | None = None) -> int:
    parser = build_evaluate_arg_parser()
    args = parser.parse_args(argv)

    try:
        evaluate_baseline_run(args.run_dir, split=args.split)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console_print(f"  ERROR: {exc}")
        return 1

    return 0
