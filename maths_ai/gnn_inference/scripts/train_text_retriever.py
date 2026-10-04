"""Fine-tune pretrained proof-state and lemma-text towers for retrieval.

Uses the same prepared trace labels, full Mathlib declaration corpus, exact
validation FAISS search, and 2,501 labeled validation rows as BM25. The run is
experimental: a trained text index is not compatible with pointer serving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.dataset import DATASET_NAME, iter_dataset_rows
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import state_dict_sha256
from maths_ai.gnn_inference.atp_lean_gnn.premise_retrieval import (
    build_positive_mask, multi_positive_info_nce,
)
from maths_ai.gnn_inference.atp_lean_gnn.retrieval_diagnostics import PremiseBM25, lexical_tokens
from maths_ai.gnn_inference.atp_lean_gnn.retriever_experiments import (
    encode_corpus_index, evaluate_index,
)
from maths_ai.gnn_inference.atp_lean_gnn.text_retriever import (
    TextDualEncoder, matched_text_rows, tokenize_texts,
)
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PreparedGraphDataset, REQUIRED_POINTER_DATA_FIELDS,
    load_prepared_metadata, resolve_device,
)


def _pairs(prepared_root, dataset_name, name_to_id, run_dir, *, resume, corpus_sha):
    train_path = run_dir / "train_text_pairs.pt"
    val_path = run_dir / "val_text_pairs.pt"
    manifest_path = run_dir / "pairs_manifest.json"
    packed_manifest = prepared_root / "packed" / "bidirectional" / "manifest.json"
    if not packed_manifest.is_file():
        raise FileNotFoundError(f"packed prepared dataset is required at '{packed_manifest}'")
    expected = {
        "dataset_name": dataset_name,
        "prepared_root": str(prepared_root.resolve()),
        "corpus_sha256": corpus_sha,
        "state_text_policy": "goal_first_v1",
        "packed_manifest_sha256": file_sha256(packed_manifest),
    }
    if resume and train_path.is_file() and val_path.is_file():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != expected:
            raise ValueError("cached text pairs do not match dataset or corpus")
        return (
            torch.load(train_path, weights_only=False),
            torch.load(val_path, weights_only=False),
        )
    metadata = load_prepared_metadata(prepared_root)
    result = []
    for split, path in (("train", train_path), ("val", val_path)):
        dataset = PreparedGraphDataset(
            metadata, split=split, cache_in_memory=True,
            required_fields=REQUIRED_POINTER_DATA_FIELDS + ("arg_lemma_ids", "arg_count"),
        )
        if not dataset.packed_cache_loaded:
            raise ValueError(f"prepared {split} packed cache is incomplete")
        pairs = matched_text_rows(
            dataset, iter_dataset_rows(dataset_name=dataset_name, split=split),
            name_to_id=name_to_id,
        )
        torch.save(pairs, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        print(f"Loaded {split}: {len(pairs)} aligned text states", flush=True)
        result.append(pairs)
        del dataset
    manifest_path.write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")
    return tuple(result)


def train_text_epoch(
    model, tokenizer, pairs, records_by_id, bm25, *, optimizer, device,
    batch_size, max_length, random_negatives, hard_negatives, max_batches,
    temperature, seed,
    start_batch=0, progress=None, progress_every=100, rng_state=None,
) -> dict[str, int | float]:
    model.train()
    labeled = [pair for pair in pairs if pair[1]]
    if not labeled:
        raise ValueError("text training split has no resolved external positives")
    rng = random.Random(seed)
    rng.shuffle(labeled)
    if rng_state is not None:
        rng.setstate(rng_state)
    all_ids = list(records_by_id)
    loss_sum = 0.0
    step_count = 0
    if start_batch < 0 or progress_every < 1:
        raise ValueError("start_batch must be nonnegative and progress_every positive")
    for batch_index, start in enumerate(range(0, len(labeled), batch_size)):
        if max_batches and batch_index >= max_batches:
            break
        if batch_index < start_batch:
            continue
        rows = labeled[start : start + batch_size]
        texts = [text for text, _ in rows]
        gold = [ids for _, ids in rows]
        candidate_ids = dict.fromkeys(lemma_id for ids in gold for lemma_id in ids)
        for text, positives in rows:
            excluded = set(positives)
            if hard_negatives:
                for lemma_id in bm25.search(lexical_tokens(text), k=hard_negatives + len(excluded) + 8):
                    if lemma_id not in excluded:
                        candidate_ids.setdefault(lemma_id, None)
                        # Keep at most the requested number from this row.
                        excluded.add(lemma_id)
                        if len(excluded) >= len(positives) + hard_negatives:
                            break
            for _ in range(random_negatives):
                lemma_id = rng.choice(all_ids)
                if lemma_id not in positives:
                    candidate_ids.setdefault(lemma_id, None)
        ids = list(candidate_ids)
        state_tokens = tokenize_texts(tokenizer, texts, device=device, max_length=max_length)
        lemma_tokens = tokenize_texts(
            tokenizer,
            [f"{records_by_id[lemma_id].name} : {records_by_id[lemma_id].statement}" for lemma_id in ids],
            device=device, max_length=max_length,
        )
        optimizer.zero_grad(set_to_none=True)
        scores = model.encode_states(state_tokens) @ model.encode_lemmas(lemma_tokens).T
        mask = build_positive_mask(ids, gold, device=device)
        loss, _ = multi_positive_info_nce(scores / temperature, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.detach().item())
        step_count += 1
        if progress is not None and (batch_index + 1) % progress_every == 0:
            progress(batch_index + 1, rng)
        if step_count % 100 == 0:
            print(f"  text batch {step_count}: loss={loss_sum / step_count:.4f}", flush=True)
    return {"loss": loss_sum / max(step_count, 1), "batches": step_count,
            "labeled_examples": min((start_batch + step_count) * batch_size, len(labeled))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared-root", "corpus-path", "run-dir", "baseline-json"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--index-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-train-batches", type=int, default=1000,
                        help="0 processes every labeled train state")
    parser.add_argument("--random-negatives", type=int, default=2)
    parser.add_argument("--hard-negatives", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.index_batch_size, args.max_length) < 1:
        raise ValueError("epochs and batch sizes/max length must be positive")
    if min(args.random_negatives, args.hard_negatives, args.max_train_batches) < 0 or args.temperature <= 0:
        raise ValueError("negative counts and batch limit must be nonnegative; temperature positive")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TextDualEncoder.from_pretrained(args.model_name).to(device)
    tokenizer_sha = hashlib.sha256(json.dumps(
        tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    base_encoder_sha = state_dict_sha256(model.state_encoder.state_dict())
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    records = load_lemma_corpus(args.corpus_path)
    corpus_sha = file_sha256(Path(args.corpus_path))
    by_id = {record.lemma_id: record for record in records}
    name_to_id = {record.name: record.lemma_id for record in records}
    run_dir = Path(args.run_dir)
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"run exists: {run_dir}; use --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    bm25_metrics = baseline.get("text_bm25")
    if not isinstance(bm25_metrics, dict) or bm25_metrics.get("labeled_query_count") != 2501:
        raise ValueError("matched 2,501-step text-BM25 baseline is required")
    if Path(baseline["config"]["corpus_path"]).resolve() != Path(args.corpus_path).resolve():
        raise ValueError("baseline must use the same lemma corpus")
    run_binding = {
        "base_encoder_sha256": base_encoder_sha,
        "tokenizer_sha256": tokenizer_sha,
        "prepared_root": str(Path(args.prepared_root).resolve()),
        "corpus_sha256": corpus_sha,
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "state_text_policy": "goal_first_v1",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_train_batches": args.max_train_batches,
        "random_negatives": args.random_negatives,
        "hard_negatives": args.hard_negatives,
        "temperature": args.temperature,
    }
    train_pairs, val_pairs = _pairs(
        Path(args.prepared_root), args.dataset_name, name_to_id, run_dir,
        resume=args.resume, corpus_sha=corpus_sha,
    )
    if len(val_pairs) != int(bm25_metrics["query_count"]):
        raise ValueError("text validation population differs from BM25")
    bm25 = PremiseBM25(records) if args.hard_negatives else None
    last_path = run_dir / "last.pt"
    epoch = 1
    phase = "ready"
    resume_batch = 0
    resume_rng = None
    resume_torch_rng = None
    resume_cuda_rng = None
    best_score = (-1.0, -1.0)
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if (saved["corpus_sha256"] != corpus_sha or saved["model_name"] != args.model_name
                or saved["tokenizer_sha256"] != tokenizer_sha
                or saved["max_length"] != args.max_length
                or saved["dataset_name"] != args.dataset_name
                or saved["state_text_policy"] != "goal_first_v1"
                or saved["run_binding"] != run_binding):
            raise ValueError("text resume model/tokenizer/corpus/data binding differs")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        epoch, phase = int(saved["epoch"]), str(saved["phase"])
        if phase == "training":
            resume_batch = int(saved["next_batch"])
            resume_rng = saved["python_rng_state"]
            resume_torch_rng = saved["torch_rng_state"].cpu()
            resume_cuda_rng = saved.get("cuda_rng_states")
        if (run_dir / "best.pt").exists():
            best_metrics = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["val_metrics"]
            best_score = (float(best_metrics["recall_at_200"]), float(best_metrics["mrr"]))
        if phase == "complete":
            epoch += 1
    def save(current_epoch: int, current_phase: str, metrics=None, *, next_batch=None, rng=None):
        payload = {
            "epoch": current_epoch, "phase": current_phase,
            "model_type": "pretrained_text_dual_encoder",
            "model_name": args.model_name, "max_length": args.max_length,
            "tokenizer_sha256": tokenizer_sha,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "corpus_sha256": corpus_sha,
            "dataset_name": args.dataset_name,
            "state_text_policy": "goal_first_v1",
            "run_binding": run_binding,
            "learning_rate": args.learning_rate,
            "val_metrics": metrics,
        }
        if next_batch is not None:
            payload.update({
                "next_batch": next_batch, "python_rng_state": rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            })
        torch.save(payload, last_path.with_suffix(".tmp"))
        last_path.with_suffix(".tmp").replace(last_path)
        return payload

    while epoch <= args.epochs:
        if phase != "train_done":
            torch.manual_seed(args.seed + epoch)
            if resume_torch_rng is not None:
                torch.set_rng_state(resume_torch_rng)
            if resume_cuda_rng is not None:
                torch.cuda.set_rng_state_all(resume_cuda_rng)
            train_metrics = train_text_epoch(
                model, tokenizer, train_pairs, by_id, bm25,
                optimizer=optimizer, device=device,
                batch_size=args.batch_size, max_length=args.max_length,
                random_negatives=args.random_negatives,
                hard_negatives=args.hard_negatives,
                max_batches=args.max_train_batches, temperature=args.temperature,
                seed=args.seed + epoch,
                start_batch=resume_batch, rng_state=resume_rng,
                progress=lambda next_batch, rng: save(
                    epoch, "training", next_batch=next_batch, rng=rng
                ),
            )
            print(f"Epoch {epoch}: trained {train_metrics}", flush=True)
            save(epoch, "train_done")
        model.eval()
        manifest = {
            "model_type": "pretrained_text_dual_encoder_index_v1",
            "model_name": args.model_name,
            "tokenizer_sha256": tokenizer_sha,
            "state_text_policy": "goal_first_v1",
            "corpus_sha256": corpus_sha,
            "lemma_encoder_sha256": state_dict_sha256(model.lemma_encoder.state_dict()),
            "max_length": args.max_length,
            "normalize": True,
        }
        index_dir = run_dir / f"index_epoch_{epoch}"
        if (index_dir / "manifest.json").exists():
            from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex
            index = LemmaIndex.load(index_dir)
            if index.manifest != manifest:
                raise ValueError("cached text index does not match checkpoint")
        else:
            print(f"Epoch {epoch}: encoding {len(records)} lemma signatures...", flush=True)
            index = encode_corpus_index(
                records,
                lambda ids: model.encode_lemmas(tokenize_texts(
                    tokenizer,
                    [f"{by_id[lemma_id].name} : {by_id[lemma_id].statement}" for lemma_id in ids],
                    device=device, max_length=args.max_length,
                )),
                batch_size=args.index_batch_size, manifest=manifest, output_dir=index_dir,
            )
        batches = (
            val_pairs[start : start + args.index_batch_size]
            for start in range(0, len(val_pairs), args.index_batch_size)
        )
        metrics = evaluate_index(
            index, batches,
            encode_queries=lambda rows: model.encode_states(tokenize_texts(
                tokenizer, [text for text, _ in rows], device=device,
                max_length=args.max_length,
            )),
            extract_positives=lambda rows: [gold for _, gold in rows],
        )
        if (metrics["labeled_query_count"] != bm25_metrics["labeled_query_count"]
                or metrics["positive_count"] != bm25_metrics["positive_count"]):
            raise ValueError("text model was evaluated against different labeled steps")
        print(f"Epoch {epoch}: text={metrics}; BM25={bm25_metrics}", flush=True)
        (run_dir / f"metrics_epoch_{epoch}.json").write_text(
            json.dumps({"text": metrics, "bm25": bm25_metrics}, indent=2), encoding="utf-8",
        )
        payload = save(epoch, "complete", metrics)
        score = (float(metrics["recall_at_200"]), float(metrics["mrr"]))
        if score > best_score:
            best_score = score
            torch.save(payload, run_dir / "best.pt")
            (run_dir / "best_index.json").write_text(
                json.dumps({"index_dir": index_dir.name, "epoch": epoch}), encoding="utf-8",
            )
        epoch += 1
        phase = "ready"
        resume_batch = 0
        resume_rng = None
        resume_torch_rng = None
        resume_cuda_rng = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
