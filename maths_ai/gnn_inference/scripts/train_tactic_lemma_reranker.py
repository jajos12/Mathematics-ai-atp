"""Fine-tune a tactic-aware text cross-encoder over frozen retrieved lemmas.

Corrected citations supervise library relevance even inside complex tactics.
This does not select local hypotheses, synthesize syntax, or execute Lean.
Training/checkpoint selection uses train/validation only; test stays untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from maths_ai.gnn_inference.atp_lean_gnn.bundle import file_sha256
from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import load_lemma_corpus
from maths_ai.gnn_inference.atp_lean_gnn.lemma_index import LemmaIndex, state_dict_sha256
from maths_ai.gnn_inference.atp_lean_gnn.tactic_lemma_reranker import (
    PROMPT_POLICY, RerankRow, TacticLemmaReranker, multi_positive_loss,
    pair_prompt, rank_metrics, sample_training_candidates,
)
from maths_ai.gnn_inference.atp_lean_gnn.text_candidates import CachedTextCandidates
from maths_ai.gnn_inference.atp_lean_gnn.training import (
    PreparedGraphDataset, load_prepared_metadata, resolve_device,
)


def _write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _tokenize_pairs(tokenizer, rows, *, device, max_length):
    goals, lemmas = zip(*rows)
    tokens = tokenizer(
        list(goals), list(lemmas), padding=True, truncation="longest_first",
        max_length=max_length, return_tensors="pt",
    )
    return {key: value.to(device) for key, value in tokens.items()}


def _score_group(model, tokenizer, row, ids, records, *, device, max_length,
                 pair_batch_size):
    scores = []
    for start in range(0, len(ids), pair_batch_size):
        prompts = [pair_prompt(row.state_text, row.tactic_name,
                               records[lemma_id].name, records[lemma_id].statement)
                   for lemma_id in ids[start:start + pair_batch_size]]
        tokens = _tokenize_pairs(tokenizer, prompts, device=device, max_length=max_length)
        scores.append(model(tokens).float())
    return torch.cat(scores)


def _build_rows(dataset, table, states, *, split, unknown_tactic_id):
    if len(states) != len(table["rows"]) or len(dataset) != len(states):
        raise ValueError(f"{split} source text and prepared/candidate populations differ")
    if not np.array_equal(table["rows"], np.arange(len(states))):
        raise ValueError(f"{split} source text cache requires original contiguous row indices")
    result = []
    counts = {"states": 0, "with_citation": 0, "usable": 0,
              "positive_pairs": 0, "absent": 0, "unknown_tactic": 0}
    for data in dataset:
        counts["states"] += 1
        row_index = int(data.row_index)
        if not 0 <= row_index < len(states):
            raise ValueError("prepared row outside source text population")
        citations = tuple(dict.fromkeys(int(x) for x in data.citation_lemma_ids.tolist() if int(x) >= 0))
        if not citations:
            continue
        counts["with_citation"] += 1
        if int(data.y.view(-1)[0]) == unknown_tactic_id:
            counts["unknown_tactic"] += 1
            continue
        ids = set(int(x) for x in table["ids"][row_index])
        hits = set(citations).intersection(ids)
        if not hits:
            counts["absent"] += 1
            continue
        counts["usable"] += 1
        counts["positive_pairs"] += len(hits)
        result.append(RerankRow(row_index, states[row_index][0], str(data.tactic_name),
                                citations, row_index))
    return result, counts


def _prepare_data(args, provider, corpus_sha, run_dir):
    metadata = load_prepared_metadata(args.prepared_root)
    source = json.loads((args.prepared_root / "argument_labels.json").read_text())
    manifest = json.loads((args.text_pairs_run / "pairs_manifest.json").read_text())
    if (Path(manifest["prepared_root"]).resolve() != Path(source["source_root"]).resolve()
            or manifest["corpus_sha256"] != corpus_sha
            or manifest["state_text_policy"] != "goal_first_v1"):
        raise ValueError("source text cache is not bound to corrected prepared-data origin")
    binding = {
        "prepared_root": str(args.prepared_root.resolve()),
        "candidate_manifest_sha256": file_sha256(args.candidate_dir / "manifest.json"),
        "argument_label_binding_sha256": file_sha256(args.prepared_root / "argument_labels.json"),
        "text_pairs_manifest_sha256": file_sha256(args.text_pairs_run / "pairs_manifest.json"),
        "corpus_sha256": corpus_sha,
    }
    binding_path = run_dir / "pair_binding.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("cached reranker pairs have different source/candidate provenance")
    cached = [run_dir / f"{split}_rerank_rows.pt" for split in ("train", "val")]
    if args.resume and binding_path.exists() and all(path.exists() for path in cached):
        return (
            torch.load(cached[0], weights_only=False),
            torch.load(cached[1], weights_only=False), binding,
        )
    loaded = []
    for split, path in zip(("train", "val"), cached):
        dataset = PreparedGraphDataset(metadata, split=split, cache_in_memory=True)
        provider.validate_dataset(dataset, split=split)
        states = torch.load(args.text_pairs_run / f"{split}_text_pairs.pt", weights_only=False)
        rows, counts = _build_rows(dataset, provider.tables[split], states, split=split,
                                   unknown_tactic_id=metadata.unknown_tactic_id)
        torch.save(rows, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        print(f"{split} cross-encoder pairs: {json.dumps(counts)}", flush=True)
        loaded.append(rows)
        del dataset, states
    _write_json(binding_path, binding)
    return *loaded, binding


def _evaluate(model, tokenizer, rows, table, records, *, device, max_length,
              pair_batch_size, output_dir, model_sha, candidate_sha):
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "eval_progress.json"
    score_path = output_dir / "eval_scores.npy"
    binding = {"model_sha256": model_sha, "candidate_manifest_sha256": candidate_sha,
               "row_count": len(rows), "next_row": 0}
    if progress_path.exists():
        current = json.loads(progress_path.read_text())
        if any(current[key] != binding[key] for key in ("model_sha256", "candidate_manifest_sha256", "row_count")):
            raise ValueError("unfinished validation scores belong to another model/candidate policy")
        start_row = int(current["next_row"])
        scores = np.load(score_path, mmap_mode="r+")
        if scores.shape != (len(rows), table["ids"].shape[1]):
            raise ValueError("validation score cache shape changed")
    else:
        start_row = 0
        scores = np.lib.format.open_memmap(
            score_path, mode="w+", dtype=np.float32,
            shape=(len(rows), table["ids"].shape[1]),
        )
        _write_json(progress_path, binding)
    model.eval()
    with torch.no_grad():
        for position in range(start_row, len(rows)):
            row = rows[position]
            ids = [int(x) for x in table["ids"][row.candidate_position]]
            scores[position] = _score_group(
                model, tokenizer, row, ids, records,
                device=device, max_length=max_length, pair_batch_size=pair_batch_size,
            ).cpu().numpy()
            if (position + 1) % 25 == 0 or position + 1 == len(rows):
                scores.flush()
                _write_json(progress_path, {**binding, "next_row": position + 1})
                print(f"validated {position + 1}/{len(rows)} gold-present states", flush=True)
    reranked = []
    baseline = []
    for row, values in zip(rows, scores):
        ids = [int(x) for x in table["ids"][row.candidate_position]]
        reranked.append([ids[int(i)] for i in np.argsort(-values, kind="stable")])
        baseline.append(ids)
    return {"cross_encoder": rank_metrics(rows, reranked),
            "text_retriever": rank_metrics(rows, baseline)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("prepared-root", "candidate-dir", "pointer-index", "corpus-path",
                 "text-pairs-run", "initial-retriever-checkpoint", "run-dir"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-states", type=int, default=20000,
                        help="0 uses all available gold-present training states")
    parser.add_argument("--max-positives", type=int, default=4)
    parser.add_argument("--negatives", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--pair-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.epochs, args.max_positives, args.negatives, args.grad_accum,
           args.pair_batch_size, args.max_length) < 1 or args.max_train_states < 0:
        raise ValueError("epochs and batch sizes must be positive; max train states nonnegative")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    corpus_sha = file_sha256(args.corpus_path)
    index = LemmaIndex.load(args.pointer_index)
    provider = CachedTextCandidates(args.candidate_dir, index)
    if provider.manifest["corpus_sha256"] != corpus_sha or provider.manifest["k"] != 200:
        raise ValueError("pilot needs bound full-corpus top-200 candidates")
    frozen = torch.load(args.initial_retriever_checkpoint, map_location="cpu", weights_only=False)
    if (frozen.get("model_type") != "pretrained_text_dual_encoder"
            or frozen.get("corpus_sha256") != corpus_sha
            or provider.manifest["text_checkpoint_sha256"] != file_sha256(args.initial_retriever_checkpoint)):
        raise ValueError("initial text checkpoint and frozen candidate list differ")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(frozen["model_name"])
    tokenizer_sha = hashlib.sha256(json.dumps(
        tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if tokenizer_sha != frozen["tokenizer_sha256"]:
        raise ValueError("initial text checkpoint tokenizer changed")
    model = TacticLemmaReranker.from_pretrained(frozen["model_name"]).to(device)
    state_weights = {name.removeprefix("state_encoder."): value for name, value
                     in frozen["model_state_dict"].items() if name.startswith("state_encoder.")}
    model.encoder.load_state_dict(state_weights, strict=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    run_dir = args.run_dir
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"reranker run already exists: {run_dir}; pass --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    records = {record.lemma_id: record for record in load_lemma_corpus(args.corpus_path)}
    train, val, pair_binding = _prepare_data(args, provider, corpus_sha, run_dir)
    if not train or not val:
        raise ValueError("no available gold-positive rows in train or validation")
    binding = {**pair_binding, "prompt_policy": PROMPT_POLICY,
               "initial_retriever_checkpoint_sha256": file_sha256(args.initial_retriever_checkpoint),
               "model_name": frozen["model_name"], "tokenizer_sha256": tokenizer_sha,
               "max_positives": args.max_positives, "negatives": args.negatives,
               "grad_accum": args.grad_accum, "max_train_states": args.max_train_states,
               "max_length": args.max_length, "seed": args.seed}
    last_path = run_dir / "last.pt"
    epoch, phase, offset = 1, "ready", 0
    saved_rng = None
    best = -1.0
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=False)
        if saved["binding"] != binding:
            raise ValueError("reranker resume uses another model/data/training policy")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        epoch, phase, offset = saved["epoch"], saved["phase"], saved.get("next_state", 0)
        saved_rng = saved.get("rng_state")
        if phase == "complete":
            epoch += 1
        if (run_dir / "best.pt").exists():
            best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["val_metrics"]["cross_encoder"]["any_positive_at_1"]

    def save(stage, *, next_state=0, rng=None, metrics=None):
        payload = {"model_type": "tactic_lemma_cross_encoder", "epoch": epoch,
                   "phase": stage, "next_state": next_state,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                   "binding": binding, "rng_state": rng.getstate() if rng else None,
                   "torch_rng_state": torch.get_rng_state(),
                   "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                   "val_metrics": metrics}
        torch.save(payload, last_path.with_suffix(".tmp"))
        last_path.with_suffix(".tmp").replace(last_path)
        return payload

    while epoch <= args.epochs:
        if phase != "train_done":
            rng = random.Random(args.seed + epoch)
            order = list(range(len(train)))
            rng.shuffle(order)
            if saved_rng is not None:
                rng.setstate(saved_rng)
                torch.set_rng_state(saved["torch_rng_state"].cpu())
                if device.type == "cuda" and saved.get("cuda_rng_states"):
                    torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng_states"]])
            limit = min(len(order), args.max_train_states) if args.max_train_states else len(order)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for position in range(offset, limit):
                row = train[order[position]]
                ranked = [int(x) for x in provider.tables["train"]["ids"][row.candidate_position]]
                ids, positive_positions = sample_training_candidates(
                    ranked, row.positive_ids, max_positives=args.max_positives,
                    negatives=args.negatives, rng=rng,
                )
                score = _score_group(
                    model, tokenizer, row, ids, records,
                    device=device, max_length=args.max_length, pair_batch_size=args.pair_batch_size,
                )
                loss = multi_positive_loss(score, positive_positions)
                (loss / args.grad_accum).backward()
                loss_sum += float(loss.detach().item())
                boundary = (position + 1) % args.grad_accum == 0 or position + 1 == limit
                if boundary:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if (position + 1) % (100 * args.grad_accum) == 0:
                        save("training", next_state=position + 1, rng=rng)
                        print(f"epoch {epoch}: trained {position + 1}/{limit} states", flush=True)
            print(f"epoch {epoch}: mean training loss {loss_sum / max(limit - offset, 1):.4f}", flush=True)
            save("train_done")
        model_sha = state_dict_sha256(model.state_dict())
        metrics = _evaluate(
            model, tokenizer, val, provider.tables["val"], records,
            device=device, max_length=args.max_length, pair_batch_size=args.pair_batch_size,
            output_dir=run_dir / f"eval_epoch_{epoch}", model_sha=model_sha,
            candidate_sha=binding["candidate_manifest_sha256"],
        )
        (run_dir / f"metrics_epoch_{epoch}.json").write_text(json.dumps(metrics, indent=2))
        print(f"epoch {epoch} validation: {json.dumps(metrics)}", flush=True)
        payload = save("complete", metrics=metrics)
        score = metrics["cross_encoder"]["any_positive_at_1"]
        if score > best:
            best = score
            torch.save(payload, run_dir / "best.pt")
        epoch, phase, offset, saved_rng = epoch + 1, "ready", 0, None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
