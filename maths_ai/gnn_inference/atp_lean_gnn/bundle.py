"""Self-contained model bundles: weights plus the vocabularies that give them meaning.

A training checkpoint is not a publishable model.  :func:`_save_checkpoint`
stores weights and a config, and that config names the prepared dataset root by
*filesystem path*; the vocabularies themselves live under that root.  Node IDs
index the label embedding and tactic IDs index the classifier, and both are
assigned by position in a sorted set -- see ``build_vocab_from_labels`` in
``pyg.py`` and ``build_tactic_vocab`` in ``labels.py`` -- so a vocabulary
rebuilt from a different set of rows renumbers everything after the first
difference.  When the renumbered vocabulary happens to have the same length the
weights still load, the service still starts, and the model answers confidently
with the wrong labels.

A bundle removes both failures at once.  It ships the vocabularies next to the
weights and records the SHA-256 of each, so the binding is verified against
content instead of trusted because a path happened to resolve.  And it can be
loaded with no prepared dataset present, which is what lets a published model
be served from the model artifacts alone.

Layout of one bundle directory::

    bundle.json          manifest: model_type, hashes, provenance
    config.json          architecture, with training-machine paths redacted
    model.safetensors    weights only; no optimizer state
    summary.json         final validation metrics (copied from the run, optional)
    metrics.jsonl        per-epoch history (copied from the run, optional)

The vocabularies are resolved from the bundle directory first and then from a
sibling ``vocab/`` directory, so a repository of several bundles can share one
canonical copy while a bundle exported with ``self_contained=True`` still works
when it is moved out on its own.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torch.nn as nn

from .premise_scoring import PremiseScorer
from .training import (
    BaselineConfig,
    PointerConfig,
    PreparedMetadata,
    build_baseline_model,
    build_pointer_model,
    detect_state_dict_model_type,
)
from .training import _stable_vocab_sha256 as stable_vocab_sha256

try:  # safetensors is preferred on disk but is not installed everywhere.
    from safetensors.torch import load_file as _safetensors_load
    from safetensors.torch import save_file as _safetensors_save

    HAS_SAFETENSORS = True
except ModuleNotFoundError:  # pragma: no cover - depends on the environment
    HAS_SAFETENSORS = False


BUNDLE_MANIFEST_NAME = "bundle.json"
BUNDLE_CONFIG_NAME = "config.json"
NODE_VOCAB_NAME = "node_vocab.json"
TACTIC_VOCAB_NAME = "tactic_vocab.json"
SHARED_VOCAB_DIRNAME = "vocab"
SAFETENSORS_MODEL_NAME = "model.safetensors"
TORCH_MODEL_NAME = "model.pt"
SAFETENSORS_SCORER_NAME = "scorer.safetensors"
TORCH_SCORER_NAME = "scorer.pt"
COPIED_RUN_FILES = ("summary.json", "metrics.jsonl")
VALID_MODEL_TYPES = ("baseline", "pointer", "pointer_gru")
VALID_WEIGHTS_FORMATS = ("auto", "safetensors", "torch")

# A pointer model built from baseline weights alone legitimately has these two
# submodules left at their random initialization: the baseline has no argument
# head, and its tactic embedding is seeded from the classifier afterwards.
POINTER_RANDOM_PREFIXES = (
    "tactic_embedding.",
    "argument_selector.",
    "stop_head.",
)

BUNDLE_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Hashing and small file helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_vocab(path: Path, vocab: Mapping[str, int]) -> Path:
    """Write a vocabulary in the canonical form its hash is computed over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(vocab), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _coerce_vocab(payload: Mapping[str, Any], *, source: str) -> dict[str, int]:
    try:
        return {str(key): int(value) for key, value in payload.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Vocabulary from {source} is not a str -> int mapping: {exc}") from exc


# ---------------------------------------------------------------------------
# Weight files
# ---------------------------------------------------------------------------


def _resolve_weights_format(requested: str) -> str:
    if requested not in VALID_WEIGHTS_FORMATS:
        raise ValueError(
            f"Unsupported weights format '{requested}'. "
            f"Use one of: {', '.join(VALID_WEIGHTS_FORMATS)}."
        )
    if requested == "safetensors":
        if not HAS_SAFETENSORS:
            raise RuntimeError(
                "weights_format='safetensors' was requested but the 'safetensors' "
                "package is not installed. Install it or use 'torch'."
            )
        return "safetensors"
    if requested == "torch":
        return "torch"
    return "safetensors" if HAS_SAFETENSORS else "torch"


def _write_tensors(
    state_dict: Mapping[str, Any],
    path: Path,
    *,
    weights_format: str,
) -> Path:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"State dict entry '{key}' is a {type(value).__name__}, not a tensor. "
                "A bundle stores weights only, so a checkpoint carrying anything else "
                "under its state dict cannot be exported."
            )
        # safetensors rejects shared storage and non-contiguous views, and a
        # detached CPU clone also makes the file byte-identical regardless of
        # which device the run trained on.
        tensors[str(key)] = value.detach().cpu().contiguous().clone()

    path.parent.mkdir(parents=True, exist_ok=True)
    if weights_format == "safetensors":
        _safetensors_save(tensors, str(path))
    else:
        # A dict of nothing but tensors is loadable with weights_only=True, so
        # this fallback keeps the property that actually matters: reading a
        # published weight file never executes pickle from the file.
        torch.save(tensors, path)
    return path


def _read_tensors(path: Path, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Bundle weights '{path}' do not exist.")
    if path.suffix == ".safetensors":
        if not HAS_SAFETENSORS:
            raise RuntimeError(
                f"Bundle weights '{path.name}' are in safetensors format but the "
                "'safetensors' package is not installed. Install it, or re-export the "
                "bundle with weights_format='torch'."
            )
        return _safetensors_load(str(path), device=str(device))
    # weights_only=True is the point of the exercise: a published weight file is
    # an untrusted file, and torch.load without it executes arbitrary pickle.
    return torch.load(path, map_location=device, weights_only=True)


def load_state_dict_checked(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    allow_missing_prefixes: Sequence[str] = (),
) -> list[str]:
    """Load ``state_dict`` into ``model``, tolerating only declared gaps.

    ``strict=False`` hides three unrelated failures behind one silent success:
    a checkpoint for a different architecture, a key-renaming bug, and a
    genuinely partial transfer.  Only the third is ever intended.  This accepts
    missing keys under ``allow_missing_prefixes`` and rejects everything else,
    which keeps the diagnostic strength of ``strict=True`` for the cases that
    are actually mistakes.

    Returns the missing keys that were allowed, so a caller can report which
    submodules are still randomly initialized.

    Raises ``ValueError`` for every kind of mismatch, including the size
    mismatches torch reports as ``RuntimeError``.  Callers treat a failed load
    as bad input rather than as a bug, and a single exception type is what lets
    them say so instead of printing a traceback.
    """
    try:
        incompatible = model.load_state_dict(dict(state_dict), strict=False)
    except RuntimeError as exc:
        # A size mismatch never reaches missing_keys/unexpected_keys: the keys
        # line up and only the tensors disagree. It means the config being built
        # from and the weights disagree -- a hand-edited hidden_dim, or a
        # vocabulary of a different size than the one that was trained on.
        raise ValueError(
            f"Weights do not fit the model they are being loaded into: {exc}"
        ) from exc
    unexpected = sorted(str(key) for key in incompatible.unexpected_keys)
    missing = sorted(str(key) for key in incompatible.missing_keys)
    unexplained = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allow_missing_prefixes)
    ]

    problems: list[str] = []
    if unexpected:
        problems.append(
            f"{len(unexpected)} unexpected key(s), e.g. {', '.join(unexpected[:5])}"
        )
    if unexplained:
        problems.append(
            f"{len(unexplained)} missing key(s), e.g. {', '.join(unexplained[:5])}"
        )
    if problems:
        raise ValueError(
            "Weights do not match the model they are being loaded into -- "
            + "; ".join(problems)
            + ". This is a genuine architecture or vocabulary mismatch, not a "
            "partial transfer."
        )
    return missing


def baseline_state_dict_as_pointer(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Re-key a baseline state dict so it loads into a pointer model's backbone.

    ``TacticWithArgsClassifier`` holds a baseline in ``self.backbone``, so every
    baseline key needs that prefix.  Doing this by declared model type rather
    than by guessing from key prefixes is what lets the load itself stay
    checked: the caller knows exactly which submodules are expected to be
    absent instead of accepting whatever does not match.
    """
    return {f"backbone.{key}": value for key, value in state_dict.items()}


def load_baseline_weights_into_pointer(
    model: nn.Module,
    baseline_state_dict: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    """Initialize a pointer model from baseline weights, the same way training does.

    The backbone takes the baseline verbatim, the tactic embedding is seeded
    from the baseline's classifier weights, and the argument selector is left
    random because a baseline has no argument head at all.  The returned tuple
    names the submodules that are still random, so a caller can say so rather
    than presenting untrained argument selection as a prediction.
    """
    missing = load_state_dict_checked(
        model,
        baseline_state_dict_as_pointer(baseline_state_dict),
        allow_missing_prefixes=POINTER_RANDOM_PREFIXES,
    )
    with torch.no_grad():
        classifier_weight = model.backbone.classifier.weight
        if model.tactic_embedding.weight.shape != classifier_weight.shape:
            raise ValueError(
                "Baseline classifier weights cannot seed the tactic embedding: "
                f"classifier shape={tuple(classifier_weight.shape)}, "
                f"embedding shape={tuple(model.tactic_embedding.weight.shape)}."
            )
        model.tactic_embedding.weight.copy_(classifier_weight)

    return tuple(
        sorted(
            {
                key.split(".", 1)[0]
                for key in missing
                if key.startswith("argument_selector.")
            }
        )
    )


# ---------------------------------------------------------------------------
# Vocabulary resolution
# ---------------------------------------------------------------------------


def resolve_vocab_paths(bundle_dir: Path) -> tuple[Path, Path]:
    """Locate a bundle's vocabularies.

    The bundle directory is searched first so a self-contained export keeps
    working when it is copied out on its own; a sibling ``vocab/`` directory is
    searched second so a repository of several bundles can share one canonical
    copy and cannot drift between them.
    """
    bundle_dir = Path(bundle_dir)
    candidates = (bundle_dir, bundle_dir.parent / SHARED_VOCAB_DIRNAME)
    for directory in candidates:
        node_path = directory / NODE_VOCAB_NAME
        tactic_path = directory / TACTIC_VOCAB_NAME
        if node_path.exists() and tactic_path.exists():
            return node_path, tactic_path

    searched = ", ".join(f"'{directory}'" for directory in candidates)
    raise FileNotFoundError(
        f"Bundle '{bundle_dir}' has no vocabularies. Looked for "
        f"{NODE_VOCAB_NAME} and {TACTIC_VOCAB_NAME} in: {searched}."
    )


def _vocabularies_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]] | None:
    node_vocab = checkpoint.get("node_vocab")
    tactic_vocab = checkpoint.get("tactic_vocab")
    if not isinstance(node_vocab, Mapping) or not isinstance(tactic_vocab, Mapping):
        return None
    return (
        _coerce_vocab(node_vocab, source="checkpoint"),
        _coerce_vocab(tactic_vocab, source="checkpoint"),
    )


def _vocabularies_from_prepared_root(
    prepared_root: Path,
) -> tuple[dict[str, int], dict[str, int]]:
    vocab_dir = Path(prepared_root) / SHARED_VOCAB_DIRNAME
    node_path = vocab_dir / NODE_VOCAB_NAME
    tactic_path = vocab_dir / TACTIC_VOCAB_NAME
    missing = [path for path in (node_path, tactic_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared root is missing required vocab files: "
            + ", ".join(str(path) for path in missing)
        )
    return (
        _coerce_vocab(_read_json(node_path), source=str(node_path)),
        _coerce_vocab(_read_json(tactic_path), source=str(tactic_path)),
    )


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

# Both fields name directories on the training machine. Inference needs no
# corpus, so they are useless to a consumer, and shipping them publishes the
# server's directory layout and username. They are replaced rather than dropped
# because ``BaselineConfig.from_dict`` requires the key to be present.
REDACTED_CONFIG_FIELDS = ("prepared_root", "run_root")
REDACTED_CONFIG_VALUE = "."


def redact_config_for_publication(config: Mapping[str, Any]) -> dict[str, Any]:
    published = dict(config)
    for field in REDACTED_CONFIG_FIELDS:
        if field in published:
            published[field] = REDACTED_CONFIG_VALUE
    return published


def redact_absolute_paths(value: Any) -> Any:
    """Replace absolute paths anywhere in a JSON value with their basenames.

    The run reports are worth publishing for their numbers, but they record the
    absolute location of everything they mention -- ``prepared_root``,
    ``run_dir``, every checkpoint and side report.  On a shared training host
    those strings publish the account name and the cluster's directory layout,
    which is of no use to a reader who has the bundle and cannot see that
    filesystem.  Keeping the basename keeps the report legible (``best.pt``
    still names which checkpoint the metrics belong to) without shipping the
    tree it sat in.
    """
    if isinstance(value, str):
        # Anchored on the separator rather than on Path.is_absolute so that a
        # path recorded on another platform is caught too.
        if value.startswith("/") or value.startswith("\\") or ":\\" in value:
            return PurePosixPath(value.replace("\\", "/")).name or REDACTED_CONFIG_VALUE
        return value
    if isinstance(value, Mapping):
        return {key: redact_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_absolute_paths(item) for item in value]
    return value


def _copy_run_file_redacted(source: Path, destination: Path) -> None:
    """Copy a run report, stripping the training machine's paths out of it."""
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        lines = []
        for line in text.splitlines():
            if not line.strip():
                continue
            lines.append(json.dumps(redact_absolute_paths(json.loads(line))))
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    payload = redact_absolute_paths(json.loads(text))
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_from_payload(
    payload: Mapping[str, Any],
    *,
    model_type: str,
) -> BaselineConfig | PointerConfig:
    declared_type = payload.get("model_type")
    if declared_type is not None and str(declared_type) != model_type:
        raise ValueError(
            f"Bundle config declares model_type '{declared_type}', but its manifest "
            f"declares '{model_type}'. Refusing to load."
        )
    if model_type in {"pointer", "pointer_gru"}:
        return PointerConfig.from_dict(dict(payload))
    return BaselineConfig.from_dict(dict(payload))


def pointer_config_from_baseline(
    config: BaselineConfig,
    *,
    max_args: int = 3,
) -> PointerConfig:
    """Derive the pointer config that wraps ``config``'s architecture.

    Used when only a baseline has been trained but the inference pipeline needs
    a pointer model, since ``InferencePipeline`` reaches through
    ``model.backbone``.  Architecture fields carry over verbatim so the wrapped
    backbone is the one the baseline weights were trained for.
    """
    payload = config.to_dict()
    payload["max_args"] = max_args
    model_payload = dict(payload.get("model") or {})
    model_payload["max_args"] = max_args
    payload["model"] = model_payload
    return PointerConfig.from_dict(payload)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def _load_source_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
    # weights_only=False is required here and is acceptable here: this reads a
    # checkpoint the operator produced on their own machine, at export time,
    # deliberately. It is exactly what a *consumer* must never have to do, which
    # is why the bundle this writes contains no pickle.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Checkpoint '{path}' is a {type(checkpoint).__name__}, not a dict of "
            "training state."
        )
    return checkpoint


def _scorer_shape_from_state_dict(state_dict: Mapping[str, Any]) -> tuple[int, str]:
    """Recover a ``PremiseScorer``'s constructor arguments from its weights.

    Neither the scorer's hidden width nor its scoring mode is recorded in the
    checkpoint ``train_scorer.py`` writes, but both are determined by the
    weights: ``key_proj`` is square in the hidden width, and only ``mode="mlp"``
    creates a ``scorer`` submodule.  Reading them here means a bundle never has
    to be told what its own scorer is.
    """
    key = "key_proj.weight"
    if key not in state_dict:
        raise ValueError(
            f"Scorer state dict is missing '{key}', so its hidden dimension cannot "
            "be recovered."
        )
    hidden_dim = int(state_dict[key].shape[0])
    mode = "mlp" if any(str(k).startswith("scorer.") for k in state_dict) else "dot"
    return hidden_dim, mode


def export_model_bundle(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    prepared_root: Path | None = None,
    vocab_dir: Path | None = None,
    self_contained: bool = False,
    weights_format: str = "auto",
    dataset: str | None = None,
    dataset_revision: str | None = None,
    run_dir: Path | None = None,
    include_scorer: bool = True,
) -> dict[str, Any]:
    """Write a self-contained bundle for ``checkpoint_path`` into ``output_dir``.

    The vocabularies come from the checkpoint when it carries them and from
    ``prepared_root`` otherwise, which is what makes checkpoints written before
    :func:`_save_checkpoint` embedded them publishable without retraining.

    Returns the manifest that was written.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    checkpoint = _load_source_checkpoint(checkpoint_path)

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has no 'model_state_dict'; it is not a "
            "model checkpoint this can export."
        )
    model_type = detect_state_dict_model_type(state_dict)

    vocabularies = _vocabularies_from_checkpoint(checkpoint)
    vocab_source: str
    if vocabularies is not None:
        vocab_source = "checkpoint"
    elif prepared_root is not None:
        vocabularies = _vocabularies_from_prepared_root(Path(prepared_root))
        vocab_source = str(prepared_root)
    else:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not carry its vocabularies (it "
            "predates them being saved) and no prepared_root was given to read them "
            "from. Pass the prepared root this checkpoint was trained against -- "
            "publishing weights without their vocabulary produces a model that "
            "cannot be loaded, or worse, loads and mislabels every prediction."
        )
    node_vocab, tactic_vocab = vocabularies

    # Config: prefer the copy embedded in the checkpoint, fall back to the
    # sibling config.json the run directory holds.
    config_payload = checkpoint.get("config")
    config_source: str
    if isinstance(config_payload, Mapping):
        config_payload = dict(config_payload)
        config_source = "checkpoint"
    else:
        sibling = checkpoint_path.parent / BUNDLE_CONFIG_NAME
        if not sibling.exists():
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' has no embedded config and its run "
                f"directory has no '{BUNDLE_CONFIG_NAME}'."
            )
        config_payload = _read_json(sibling)
        config_source = str(sibling)
    config_payload["model_type"] = model_type

    # Build the model once before writing anything. A config that disagrees with
    # its own weights must fail here, while the output directory is still empty,
    # rather than producing a bundle that only fails for whoever downloads it.
    metadata = PreparedMetadata.from_vocabs(
        node_vocab=node_vocab, tactic_vocab=tactic_vocab
    )
    config = _config_from_payload(config_payload, model_type=model_type)
    probe = (
        build_pointer_model(metadata, config)
        if model_type in {"pointer", "pointer_gru"}
        else build_baseline_model(metadata, config)
    )
    load_state_dict_checked(probe, state_dict)

    resolved_format = _resolve_weights_format(weights_format)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_name = (
        SAFETENSORS_MODEL_NAME if resolved_format == "safetensors" else TORCH_MODEL_NAME
    )
    weights_path = _write_tensors(
        state_dict, output_dir / weights_name, weights_format=resolved_format
    )

    # Vocabularies: shared by default so a repository of bundles cannot drift
    # between copies, inside the bundle when it must travel alone.
    if self_contained:
        vocab_target = output_dir
    else:
        vocab_target = Path(vocab_dir) if vocab_dir is not None else output_dir.parent / SHARED_VOCAB_DIRNAME
    node_vocab_path = _write_vocab(vocab_target / NODE_VOCAB_NAME, node_vocab)
    tactic_vocab_path = _write_vocab(vocab_target / TACTIC_VOCAB_NAME, tactic_vocab)

    _write_json(output_dir / BUNDLE_CONFIG_NAME, redact_config_for_publication(config_payload))

    scorer_block: dict[str, Any] | None = None
    scorer_state_dict = checkpoint.get("scorer_state_dict")
    if include_scorer and isinstance(scorer_state_dict, Mapping):
        # A premise scorer is trained jointly with a pointer and is meaningless
        # apart from it, so it rides in the same bundle rather than getting a
        # directory of its own that could be paired with the wrong weights.
        scorer_hidden_dim, scorer_mode = _scorer_shape_from_state_dict(scorer_state_dict)
        scorer_name = (
            SAFETENSORS_SCORER_NAME
            if resolved_format == "safetensors"
            else TORCH_SCORER_NAME
        )
        scorer_path = _write_tensors(
            scorer_state_dict, output_dir / scorer_name, weights_format=resolved_format
        )
        scorer_block = {
            "weights": scorer_path.name,
            "weights_sha256": file_sha256(scorer_path),
            "hidden_dim": scorer_hidden_dim,
            "scoring_mode": scorer_mode,
        }

    source_run_dir = Path(run_dir) if run_dir is not None else checkpoint_path.parent
    copied: list[str] = []
    for name in COPIED_RUN_FILES:
        candidate = source_run_dir / name
        if candidate.exists():
            # Redacted rather than copied: these reports name every path on the
            # training host, and config.json's redaction would be pointless if
            # summary.json republished the same directories next to it.
            try:
                _copy_run_file_redacted(candidate, output_dir / name)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Unparseable means unredactable, and an unredacted report is
                # not publishable, so leave it out and say so in the manifest.
                continue
            copied.append(name)

    val_metrics = checkpoint.get("val_metrics")
    manifest: dict[str, Any] = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "model_type": model_type,
        "weights": weights_path.name,
        "weights_format": resolved_format,
        "weights_sha256": file_sha256(weights_path),
        "config": BUNDLE_CONFIG_NAME,
        "node_vocab": NODE_VOCAB_NAME,
        "tactic_vocab": TACTIC_VOCAB_NAME,
        # Hashes are over the canonical JSON of the mapping, not over the file,
        # so they survive reformatting and identify the vocabulary itself.
        "node_vocab_sha256": stable_vocab_sha256(node_vocab),
        "tactic_vocab_sha256": stable_vocab_sha256(tactic_vocab),
        # Recorded redundantly so a wrong-sized vocabulary is rejected by name
        # before it becomes an opaque tensor shape error.
        "num_node_labels": len(node_vocab),
        "num_tactics": len(tactic_vocab),
        "vocab_location": "bundle" if self_contained else "shared",
        "source_checkpoint_name": checkpoint_path.name,
        "source_checkpoint_sha256": file_sha256(checkpoint_path),
        "source_epoch": int(checkpoint.get("epoch", 0)),
        "source_run": source_run_dir.name,
        "vocab_source": vocab_source,
        "config_source": config_source,
        "copied_run_files": copied,
        "optimizer_state_included": False,
    }
    if isinstance(val_metrics, Mapping):
        manifest["val_metrics"] = {
            str(key): value for key, value in val_metrics.items()
        }
    if dataset is not None:
        manifest["dataset"] = dataset
    if dataset_revision is not None:
        manifest["dataset_revision"] = dataset_revision
    if scorer_block is not None:
        manifest["scorer"] = scorer_block

    _write_json(output_dir / BUNDLE_MANIFEST_NAME, manifest)
    manifest["_paths"] = {
        "bundle_dir": str(output_dir),
        "weights": str(weights_path),
        "node_vocab": str(node_vocab_path),
        "tactic_vocab": str(tactic_vocab_path),
    }
    return manifest


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedBundle:
    """A model rebuilt from a bundle, with the vocabularies it was trained on."""

    directory: Path
    model_type: str
    model: nn.Module
    metadata: PreparedMetadata
    config: BaselineConfig | PointerConfig
    manifest: dict[str, Any]
    scorer: PremiseScorer | None = None
    randomly_initialized: tuple[str, ...] = ()

    @property
    def node_vocab(self) -> dict[str, int]:
        return self.metadata.node_vocab

    @property
    def tactic_vocab(self) -> dict[str, int]:
        return self.metadata.tactic_vocab


def read_bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / BUNDLE_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"'{bundle_dir}' is not a model bundle: no {BUNDLE_MANIFEST_NAME}."
        )
    manifest = _read_json(manifest_path)
    model_type = str(manifest.get("model_type", ""))
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Bundle '{bundle_dir}' declares model_type '{model_type}'; expected one "
            f"of: {', '.join(VALID_MODEL_TYPES)}."
        )
    return manifest


def load_bundle_vocabularies(
    bundle_dir: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    """Read and verify a bundle's vocabularies.

    Verification is the reason a bundle exists.  A vocabulary that hashes
    differently from the one the weights were trained against renumbers the
    label embedding and the classifier, and when its length is unchanged nothing
    downstream can detect that -- the weights load, the model runs, and every
    prediction is decoded against the wrong table.  So a mismatch is fatal here
    rather than a warning.
    """
    node_path, tactic_path = resolve_vocab_paths(bundle_dir)
    node_vocab = _coerce_vocab(_read_json(node_path), source=str(node_path))
    tactic_vocab = _coerce_vocab(_read_json(tactic_path), source=str(tactic_path))

    for name, vocab, path, expected_hash_key, expected_size_key in (
        ("node", node_vocab, node_path, "node_vocab_sha256", "num_node_labels"),
        ("tactic", tactic_vocab, tactic_path, "tactic_vocab_sha256", "num_tactics"),
    ):
        expected_size = manifest.get(expected_size_key)
        if expected_size is not None and int(expected_size) != len(vocab):
            raise ValueError(
                f"Bundle {name} vocabulary at '{path}' holds {len(vocab):,} entries but "
                f"the manifest declares {int(expected_size):,}. The weights index this "
                "vocabulary by position, so loading them against it would mislabel "
                "predictions."
            )
        expected_hash = manifest.get(expected_hash_key)
        if expected_hash is None:
            continue
        actual_hash = stable_vocab_sha256(vocab)
        if actual_hash != str(expected_hash):
            raise ValueError(
                f"Bundle {name} vocabulary at '{path}' does not match the one these "
                f"weights were trained against (sha256 {actual_hash[:12]} vs declared "
                f"{str(expected_hash)[:12]}). IDs are assigned by sorted position, so a "
                "different vocabulary silently repoints every row of the embedding and "
                "every class of the classifier. Refusing to load."
            )

    return node_vocab, tactic_vocab


def _verify_weights_hash(path: Path, manifest: Mapping[str, Any], key: str) -> None:
    expected = manifest.get(key)
    if expected is None:
        return
    actual = file_sha256(path)
    if actual != str(expected):
        raise ValueError(
            f"Bundle weights '{path.name}' hash {actual[:12]} but the manifest declares "
            f"{str(expected)[:12]}. The file has been modified or truncated."
        )


def load_model_bundle(
    bundle_dir: Path,
    *,
    device: torch.device | str = "cpu",
    eval_mode: bool = True,
) -> LoadedBundle:
    """Rebuild the model a bundle describes, with no prepared dataset required.

    This is the load path a serving process should use.  It verifies the
    vocabulary binding, builds the architecture the bundle declares, and loads
    the weights strictly, so the three ways a published model can be wrong --
    tampered file, wrong vocabulary, mismatched architecture -- all surface as
    errors here instead of as bad predictions later.
    """
    bundle_dir = Path(bundle_dir)
    device = torch.device(device)
    manifest = read_bundle_manifest(bundle_dir)
    model_type = str(manifest["model_type"])

    node_vocab, tactic_vocab = load_bundle_vocabularies(bundle_dir, manifest)
    metadata = PreparedMetadata.from_vocabs(
        node_vocab=node_vocab, tactic_vocab=tactic_vocab, root=bundle_dir
    )

    config_path = bundle_dir / str(manifest.get("config", BUNDLE_CONFIG_NAME))
    if not config_path.exists():
        raise FileNotFoundError(f"Bundle '{bundle_dir}' is missing '{config_path.name}'.")
    config = _config_from_payload(_read_json(config_path), model_type=model_type)

    weights_path = bundle_dir / str(manifest["weights"])
    _verify_weights_hash(weights_path, manifest, "weights_sha256")
    state_dict = _read_tensors(weights_path, device="cpu")

    model: nn.Module = (
        build_pointer_model(metadata, config)
        if model_type in {"pointer", "pointer_gru"}
        else build_baseline_model(metadata, config)
    )
    load_state_dict_checked(model, state_dict)
    model = model.to(device)

    scorer: PremiseScorer | None = None
    scorer_block = manifest.get("scorer")
    if isinstance(scorer_block, Mapping):
        scorer_path = bundle_dir / str(scorer_block["weights"])
        _verify_weights_hash(scorer_path, scorer_block, "weights_sha256")
        scorer = PremiseScorer(
            hidden_dim=int(scorer_block["hidden_dim"]),
            mode=str(scorer_block.get("scoring_mode", "dot")),
        )
        load_state_dict_checked(scorer, _read_tensors(scorer_path, device="cpu"))
        scorer = scorer.to(device)

    if eval_mode:
        model.eval()
        if scorer is not None:
            scorer.eval()

    return LoadedBundle(
        directory=bundle_dir,
        model_type=model_type,
        model=model,
        metadata=metadata,
        config=config,
        manifest=manifest,
        scorer=scorer,
    )


def load_pointer_bundle(
    bundle_dir: Path,
    *,
    device: torch.device | str = "cpu",
    max_args: int = 3,
    eval_mode: bool = True,
) -> LoadedBundle:
    """Load a bundle as a ``TacticWithArgsClassifier``, wrapping a baseline if needed.

    ``InferencePipeline`` reaches through ``model.backbone``, so it requires a
    pointer model even when only a baseline has been trained.  A baseline bundle
    is therefore wrapped: its weights go into the backbone, the tactic embedding
    is seeded from the classifier the way
    ``initialize_pointer_from_baseline_checkpoint`` does it, and the argument
    head stays random.  The wrapped submodules are reported in
    ``randomly_initialized`` so a caller can say so out loud rather than
    presenting untrained argument selection as a prediction.
    """
    bundle_dir = Path(bundle_dir)
    device = torch.device(device)
    manifest = read_bundle_manifest(bundle_dir)
    model_type = str(manifest["model_type"])

    if model_type in {"pointer", "pointer_gru"}:
        return load_model_bundle(bundle_dir, device=device, eval_mode=eval_mode)

    node_vocab, tactic_vocab = load_bundle_vocabularies(bundle_dir, manifest)
    metadata = PreparedMetadata.from_vocabs(
        node_vocab=node_vocab, tactic_vocab=tactic_vocab, root=bundle_dir
    )

    config_path = bundle_dir / str(manifest.get("config", BUNDLE_CONFIG_NAME))
    baseline_config = BaselineConfig.from_dict(_read_json(config_path))
    pointer_config = pointer_config_from_baseline(baseline_config, max_args=max_args)

    weights_path = bundle_dir / str(manifest["weights"])
    _verify_weights_hash(weights_path, manifest, "weights_sha256")
    baseline_state_dict = _read_tensors(weights_path, device="cpu")

    model = build_pointer_model(metadata, pointer_config)
    randomly_initialized = load_baseline_weights_into_pointer(model, baseline_state_dict)

    model = model.to(device)
    if eval_mode:
        model.eval()

    return LoadedBundle(
        directory=bundle_dir,
        model_type="pointer",
        model=model,
        metadata=metadata,
        config=pointer_config,
        manifest=manifest,
        scorer=None,
        randomly_initialized=randomly_initialized,
    )
