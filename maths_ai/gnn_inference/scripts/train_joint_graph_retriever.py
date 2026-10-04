"""Train both graph retriever towers; rebuild the entire lemma index each epoch.

Experimental run only. Its index uses the retriever's lemma tower and must not
be substituted for the pointer-bound serving index without a matching loader.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256, load_state_dict_checked
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import (
    LemmaIndex, load_index_for_encoder, state_dict_sha256,
)
from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import DualEncoderRetriever
from maths_ai.gnn_inference.atp_lean_gnn.premise_retriever_training import (
    LemmaGraphStore, extract_external_positive_ids,
    load_or_build_hnsw_index, train_retriever_epoch,
)
from maths_ai.gnn_inference.atp_lean_gnn.retriever_experiments import (
    encode_corpus_index, evaluate_index,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    REQUIRED_POINTER_DATA_FIELDS, build_dataloaders, build_pointer_model,
    load_pointer_config, load_prepared_metadata, resolve_device,
)


def _binding(index, model, corpus_sha: str) -> dict[str, object]:
    return {
        **index.manifest,
        "model_type": "joint_graph_retriever_index_v1",
        "corpus_sha256": corpus_sha,
        "encoder_state_sha256": state_dict_sha256(model.lemma_encoder.state_dict()),
        "lemma_projection_sha256": state_dict_sha256(model.lemma_projection.state_dict()),
        "state_graph_policy": "prepared_model_sexpr_v2",
        "lemma_graph_policy": "pretty_printed_type_graph_v1",
        "normalize": True,
    }


def _verify_index(index: LemmaIndex, model: DualEncoderRetriever, corpus_sha: str) -> None:
    manifest = index.manifest
    for key, expected in (
        ("model_type", "joint_graph_retriever_index_v1"),
        ("corpus_sha256", corpus_sha),
        ("encoder_state_sha256", state_dict_sha256(model.lemma_encoder.state_dict())),
        ("lemma_projection_sha256", state_dict_sha256(model.lemma_projection.state_dict())),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"joint graph index differs from trained model at {key}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "prepared-root", "checkpoint", "index-path", "corpus-path", "run-dir", "baseline-json"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--index-batch-size", type=int, default=64)
    parser.add_argument("--max-train-batches", type=int, default=1000,
                        help="0 processes entire train split")
    parser.add_argument("--random-negatives", type=int, default=4)
    parser.add_argument("--hard-negatives", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.index_batch_size) < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if min(args.max_train_batches, args.random_negatives, args.hard_negatives) < 0:
        raise ValueError("batch limit and negative counts must be nonnegative")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    config = load_pointer_config(args.config, device_override=args.device)
    config = replace(
        config, prepared_root=Path(args.prepared_root),
        training=replace(config.training, batch_size=args.batch_size, num_workers=0),
    ).normalized()
    metadata = load_prepared_metadata(config.prepared_root)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    pointer_state = checkpoint.get("model_state_dict", checkpoint)
    if any(str(key).startswith("module.") for key in pointer_state):
        pointer_state = {
            str(key).removeprefix("module."): value
            for key, value in pointer_state.items()
        }
    pointer = build_pointer_model(metadata, config).to(device)
    load_state_dict_checked(pointer, pointer_state)
    prior_index = load_index_for_encoder(
        args.index_path, encoder_state_dict=pointer_state,
        node_vocab=metadata.node_vocab, tactic_vocab=metadata.tactic_vocab,
        corpus_path=args.corpus_path, expected_edge_mode=config.edge_mode,
    )
    if not prior_index.normalize_queries:
        raise ValueError("prior index must be normalized")
    model = DualEncoderRetriever(
        copy.deepcopy(pointer.backbone), copy.deepcopy(pointer.backbone),
        hidden_dim=config.model.hidden_dim,
    ).to(device)
    del pointer, checkpoint
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    records = load_lemma_corpus(args.corpus_path)
    if len(records) != len(prior_index.lemma_ids):
        raise ValueError("lemma corpus and pointer index disagree on population")
    if [record.lemma_id for record in records] != prior_index.lemma_ids:
        raise ValueError("lemma corpus and pointer index order differ")
    corpus_sha = file_sha256(Path(args.corpus_path))
    store = LemmaGraphStore(records, node_vocab=metadata.node_vocab, edge_mode=config.edge_mode)
    _, loaders = build_dataloaders(
        metadata, config,
        required_fields=REQUIRED_POINTER_DATA_FIELDS + ("arg_lemma_ids", "arg_count"),
    )
    baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    baseline_metrics = baseline.get("text_bm25")
    if not isinstance(baseline_metrics, dict) or baseline_metrics.get("labeled_query_count") != 2501:
        raise ValueError("baseline must contain the matched 2,501-step text-BM25 result")
    if Path(baseline["config"]["corpus_path"]).resolve() != Path(args.corpus_path).resolve():
        raise ValueError("baseline must use the same lemma corpus")
    run_binding = {
        "pointer_checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "pointer_index_manifest": prior_index.manifest,
        "prepared_root": str(Path(args.prepared_root).resolve()),
        "corpus_sha256": corpus_sha,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_train_batches": args.max_train_batches,
        "random_negatives": args.random_negatives,
        "hard_negatives": args.hard_negatives,
    }
    run_dir = Path(args.run_dir)
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run already exists: {run_dir}; pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"
    epoch = 1
    phase = "ready"
    resume_batch = 0
    resume_rng = None
    resume_torch_rng = None
    resume_cuda_rng = None
    best_score = (-1.0, -1.0)
    if args.resume and last_path.is_file():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if (saved["corpus_sha256"] != corpus_sha
                or saved["node_vocab"] != metadata.node_vocab
                or saved["tactic_vocab"] != metadata.tactic_vocab
                or saved["pointer_index_manifest"] != prior_index.manifest
                or saved["run_binding"] != run_binding):
            raise ValueError("resume corpus, pointer index, or vocabulary differs")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        epoch, phase = int(saved["epoch"]), str(saved["phase"])
        if phase == "training":
            resume_batch = int(saved["next_batch"])
            resume_rng = saved["python_rng_state"]
            resume_torch_rng = saved["torch_rng_state"].cpu()
            resume_cuda_rng = saved.get("cuda_rng_states")
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        best_path = run_dir / "best.pt"
        if best_path.exists():
            best_metrics = torch.load(best_path, map_location="cpu", weights_only=False)["val_metrics"]
            best_score = (float(best_metrics["recall_at_200"]), float(best_metrics["mrr"]))
        if phase == "complete":
            epoch += 1
    prior = prior_index
    if epoch > 1 or (args.resume and phase == "train_done"):
        previous_epoch = epoch - 1
        if previous_epoch > 0:
            prior = LemmaIndex.load(run_dir / f"index_epoch_{previous_epoch}")
            # A train_done checkpoint has updated lemma weights; previous index
            # belongs to the pre-training model and is used only for mining.
            if phase == "complete":
                _verify_index(prior, model, corpus_sha)
    def save_checkpoint(current_epoch: int, current_phase: str, metrics=None, *, next_batch=None, rng=None) -> dict:
        payload = {
            "epoch": current_epoch, "phase": current_phase,
            "model_type": "joint_graph_retriever",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "corpus_sha256": corpus_sha,
            "node_vocab": metadata.node_vocab,
            "tactic_vocab": metadata.tactic_vocab,
            "pointer_index_manifest": prior_index.manifest,
            "run_binding": run_binding,
            "learning_rate": args.learning_rate,
            "val_metrics": metrics,
        }
        if next_batch is not None:
            payload.update({
                "next_batch": next_batch,
                "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            })
        torch.save(payload, last_path.with_suffix(".tmp"))
        last_path.with_suffix(".tmp").replace(last_path)
        return payload

    while epoch <= args.epochs:
        torch.manual_seed(args.seed + epoch)
        if resume_torch_rng is not None:
            torch.set_rng_state(resume_torch_rng)
        if resume_cuda_rng is not None:
            torch.cuda.set_rng_state_all(resume_cuda_rng)
        sampler = getattr(loaders["train"], "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        if phase != "train_done":
            mining = load_or_build_hnsw_index(
                prior, run_dir / f"mining_epoch_{epoch}.faiss"
            ) if args.hard_negatives else None
            rng = random.Random(args.seed + epoch)
            if resume_rng is not None:
                rng.setstate(resume_rng)
            train_metrics = train_retriever_epoch(
                model, loaders["train"], store, prior,
                optimizer=optimizer, device=device,
                accessible_negative_count=args.random_negatives,
                hard_negative_count=args.hard_negatives, grad_clip=1.0,
                rng=rng,
                hard_negative_index=mining,
                trainable_lemma_tower=True,
                max_batches=args.max_train_batches or None,
                start_batch=resume_batch,
                progress=lambda next_batch, current_rng: save_checkpoint(
                    epoch, "training", next_batch=next_batch, rng=current_rng
                ),
            )
            print(f"Epoch {epoch}: trained; {train_metrics}", flush=True)
            save_checkpoint(epoch, "train_done")
        model.eval()
        manifest = _binding(prior_index, model, corpus_sha)
        index_dir = run_dir / f"index_epoch_{epoch}"
        if (index_dir / "manifest.json").exists():
            rebuilt = LemmaIndex.load(index_dir)
            _verify_index(rebuilt, model, corpus_sha)
        else:
            print(f"Epoch {epoch}: re-encoding {len(records)} lemma graphs...", flush=True)
            rebuilt = encode_corpus_index(
                records,
                lambda ids: model.encode_lemmas(store.batch(ids, device=device)),
                batch_size=args.index_batch_size,
                manifest=manifest,
                output_dir=index_dir,
            )
        metrics = evaluate_index(
            rebuilt, loaders["val"],
            encode_queries=lambda batch: model.encode_states(batch.to(device)),
            extract_positives=lambda batch: extract_external_positive_ids(
                batch, name_to_id=rebuilt.name_to_id,
            ),
        )
        if (metrics["labeled_query_count"] != baseline_metrics["labeled_query_count"]
                or metrics["positive_count"] != baseline_metrics["positive_count"]
                or metrics["query_count"] != baseline_metrics["query_count"]):
            raise ValueError("graph retriever did not evaluate the same labeled validation steps")
        print(f"Epoch {epoch}: graph={metrics} baseline={baseline_metrics}", flush=True)
        (run_dir / f"metrics_epoch_{epoch}.json").write_text(
            json.dumps({"graph": metrics, "bm25": baseline_metrics}, indent=2),
            encoding="utf-8",
        )
        payload = save_checkpoint(epoch, "complete", metrics)
        score = (float(metrics["recall_at_200"]), float(metrics["mrr"]))
        if score > best_score:
            best_score = score
            torch.save(payload, run_dir / "best.pt")
            (run_dir / "best_index.json").write_text(
                json.dumps({"index_dir": index_dir.name, "epoch": epoch}), encoding="utf-8"
            )
        prior = rebuilt
        phase = "ready"
        epoch += 1
        resume_batch = 0
        resume_rng = None
        resume_torch_rng = None
        resume_cuda_rng = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
