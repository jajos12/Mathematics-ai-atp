"""Tactic-conditioned reranking over typed local and library candidates."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .argument_selector import TacticWithArgsClassifier
from .premise_pool import CandidatePool, CandidateRef, CandidateSource


def load_unified_reranker_weights(
    checkpoint_path: str | Path,
    *,
    model: TacticWithArgsClassifier,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    expected_edge_mode: str,
    device: torch.device,
) -> dict[str, object] | None:
    """Load pointer weights when a scorer path is a unified reranker checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "unified_action_reranker":
        return None
    if checkpoint.get("node_vocab") != node_vocab or checkpoint.get("tactic_vocab") != tactic_vocab:
        raise ValueError("unified reranker vocabularies do not match the tactic model")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("unified reranker checkpoint is missing its embedded config")
    if config.get("edge_mode", "bidirectional") != expected_edge_mode:
        raise ValueError("unified reranker edge mode does not match the tactic model")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("unified reranker checkpoint is missing model_state_dict")
    from .bundle import load_state_dict_checked

    load_state_dict_checked(model, state_dict)
    return checkpoint


def verify_unified_reranker_dependencies(
    checkpoint: dict[str, object] | None,
    *,
    index_manifest: dict[str, object],
    retriever,
) -> None:
    """Refuse index or retriever substitutions that change the trained pool."""
    if checkpoint is None:
        return
    saved_manifest = checkpoint.get("lemma_index_manifest")
    if not isinstance(saved_manifest, dict):
        raise ValueError("unified reranker checkpoint is missing lemma index provenance")
    binding_fields = (
        "encoder_state_sha256",
        "node_vocab_sha256",
        "tactic_vocab_sha256",
        "corpus_sha256",
        "edge_mode",
        "normalize",
    )
    if any(saved_manifest.get(key) != index_manifest.get(key) for key in binding_fields):
        raise ValueError("unified reranker was trained against a different lemma index")

    expected_retriever_sha = checkpoint.get("retriever_state_sha256")
    if expected_retriever_sha is None:
        if retriever is not None:
            raise ValueError("unified reranker was trained without a dual-encoder retriever")
        return
    if retriever is None:
        raise ValueError("unified reranker requires its trained dual-encoder retriever")
    from .lemma_index import state_dict_sha256

    if expected_retriever_sha != state_dict_sha256(retriever.state_dict()):
        raise ValueError("unified reranker was trained with a different retriever checkpoint")


@dataclass(frozen=True)
class RankedAction:
    tactic_id: int
    tactic_name: str
    arguments: tuple[CandidateRef, ...]
    log_probability: float

    @property
    def probability(self) -> float:
        return float(torch.exp(torch.tensor(self.log_probability)).item())

    @property
    def stable_key(self) -> tuple[int, tuple[tuple[str, int, int | None], ...]]:
        return self.tactic_id, tuple(argument.key for argument in self.arguments)


@dataclass(frozen=True)
class ActionTarget:
    tactic_id: int
    arguments: tuple[CandidateRef, ...]
    resolved: bool = True
    argument_count: int | None = None

    @property
    def stable_key(self) -> tuple[int, tuple[tuple[str, int, int | None], ...]]:
        return self.tactic_id, tuple(argument.key for argument in self.arguments)


@dataclass
class _Beam:
    decoder_state: Tensor
    positions: tuple[int, ...]
    log_probability: float


def resolve_ordered_pool_targets(
    pools: Sequence[CandidatePool],
    arg_node_indices: Tensor,
    arg_lemma_ids: Tensor,
    arg_counts: Sequence[int],
    *,
    max_args: int,
) -> tuple[list[list[int]], dict[str, int]]:
    """Resolve each ordered trace argument to an exact mixed-pool position."""
    batch_size = len(pools)
    if arg_node_indices.size(0) != batch_size or arg_lemma_ids.size(0) != batch_size:
        raise ValueError("argument target tensors must have one row per pool")
    if len(arg_counts) != batch_size:
        raise ValueError("arg_counts must have one value per pool")
    rows: list[list[int]] = []
    local_count = 0
    library_count = 0
    unresolved_count = 0
    for row, pool in enumerate(pools):
        positions_by_key = {
            (candidate.source, candidate.stable_id): position
            for position, candidate in enumerate(pool.candidates)
        }
        positions: list[int] = []
        count = min(int(arg_counts[row]), max_args)
        if count > arg_node_indices.size(1) or count > arg_lemma_ids.size(1):
            raise ValueError(
                f"argument target row {row} stores fewer positions than arg_count"
            )
        for step in range(count):
            local_id = int(arg_node_indices[row, step].item())
            lemma_id = int(arg_lemma_ids[row, step].item())
            if local_id >= 0 and lemma_id >= 0:
                raise ValueError(
                    f"argument {step} in graph {row} has both local and library IDs"
                )
            if local_id >= 0:
                if local_id < 0:
                    raise ValueError("local node IDs must be non-negative")
                key = (CandidateSource.LOCAL, local_id)
                local_count += 1
            elif lemma_id >= 0:
                key = (CandidateSource.LIBRARY, lemma_id)
                library_count += 1
            else:
                positions.append(-1)
                unresolved_count += 1
                continue
            position = positions_by_key.get(key, -1)
            positions.append(position)
            if position < 0:
                unresolved_count += 1
        rows.append(positions)
    return rows, {
        "local_target_count": local_count,
        "library_target_count": library_count,
        "unresolved_target_count": unresolved_count,
    }


def compute_unified_reranking_loss(
    model: TacticWithArgsClassifier,
    state_embeddings: Tensor,
    tactic_ids: Tensor,
    pools: Sequence[CandidatePool],
    target_positions: Sequence[Sequence[int]],
) -> tuple[Tensor, dict[str, int | float]]:
    """Teacher-force ordered mixed candidates through the serving GRU pointer."""
    batch_size = int(state_embeddings.size(0))
    if len(pools) != batch_size or len(target_positions) != batch_size:
        raise ValueError("states, pools, and target positions must have equal rows")
    tactic_embeddings = model.tactic_embedding(tactic_ids)
    losses: list[Tensor] = []
    target_count = 0
    scored_count = 0
    top1_correct = 0
    source_correct = 0
    unresolved_count = 0

    for row, (pool, raw_targets) in enumerate(zip(pools, target_positions)):
        targets = list(raw_targets[: model.max_args])
        decoder_state = model.argument_selector.initial_state(
            state_embeddings[row : row + 1], tactic_embeddings[row : row + 1]
        )
        selected: set[int] = set()
        sequence_resolved = True
        target_count += len(targets)
        for step, target_position in enumerate(targets):
            continue_logit = model.stop_head(decoder_state).squeeze(-1)
            losses.append(
                F.binary_cross_entropy_with_logits(
                    continue_logit, torch.zeros_like(continue_logit)
                )
            )
            if target_position < 0:
                unresolved_count += len(targets) - step
                sequence_resolved = False
                break
            if target_position >= len(pool.candidates):
                raise ValueError(
                    f"target pool position {target_position} is outside graph {row}'s "
                    f"{len(pool.candidates)} candidates"
                )
            if target_position in selected:
                raise ValueError(
                    f"argument position {step} repeats pool candidate {target_position}"
                )
            scores = model.argument_selector.score_candidates(
                decoder_state, pool.candidate_vectors
            ).squeeze(0)
            if selected:
                mask = torch.zeros_like(scores, dtype=torch.bool)
                mask[list(selected)] = True
                scores = scores.masked_fill(mask, float("-inf"))
            target = torch.tensor([target_position], device=scores.device)
            losses.append(F.cross_entropy(scores.unsqueeze(0), target))
            prediction = int(scores.argmax().item())
            top1_correct += int(prediction == target_position)
            source_correct += int(
                pool.candidates[prediction].source
                is pool.candidates[target_position].source
            )
            scored_count += 1
            selected.add(target_position)
            decoder_state = model.argument_selector.gru(
                pool.candidate_vectors[target_position].unsqueeze(0), decoder_state
            )

        # Stop supervision remains part of the same decoder. Missing retrieval
        # targets cannot be teacher-forced, so do not pretend their shortened
        # prefix is a valid complete action.
        if sequence_resolved:
            stop_logit = model.stop_head(decoder_state).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(
                stop_logit, torch.ones_like(stop_logit)
            ))

    loss = torch.stack(losses).mean() if losses else state_embeddings.sum() * 0.0
    return loss, {
        "reranking_loss": float(loss.detach().item()),
        "target_count": target_count,
        "scored_target_count": scored_count,
        "target_coverage": scored_count / max(target_count, 1),
        "unresolved_target_count": unresolved_count,
        "candidate_top1_accuracy": top1_correct / max(scored_count, 1),
        "candidate_source_accuracy": source_correct / max(scored_count, 1),
    }


@torch.no_grad()
def rank_fresh_name_action(
    model: TacticWithArgsClassifier,
    state_embedding: Tensor,
    *,
    tactic_id: int,
    tactic_name: str,
    tactic_log_probability: float,
    candidates: Sequence[CandidateRef],
    candidate_vectors: Tensor,
) -> RankedAction:
    """Score deterministic outer-binder selection with the shared stop decoder."""
    if candidate_vectors.shape != (len(candidates), model.hidden_dim):
        raise ValueError("fresh-name candidates and vectors do not align")
    tactic_embedding = model.tactic_embedding(
        torch.tensor([tactic_id], device=state_embedding.device)
    )
    decoder_state = model.argument_selector.initial_state(
        state_embedding, tactic_embedding
    )
    log_probability = float(tactic_log_probability)
    selected: list[CandidateRef] = []
    for position, candidate in enumerate(candidates[: model.max_args]):
        stop_logit = model.stop_head(decoder_state).view(())
        if float(stop_logit.item()) >= 0:
            log_probability += float(F.logsigmoid(stop_logit).item())
            break
        log_probability += float(F.logsigmoid(-stop_logit).item())
        selected.append(candidate)
        decoder_state = model.argument_selector.gru(
            candidate_vectors[position].unsqueeze(0), decoder_state
        )
    else:
        stop_logit = model.stop_head(decoder_state).view(())
        log_probability += float(F.logsigmoid(stop_logit).item())
    return RankedAction(
        tactic_id=tactic_id,
        tactic_name=tactic_name,
        arguments=tuple(selected),
        log_probability=log_probability,
    )


@torch.no_grad()
def rank_complete_actions(
    model: TacticWithArgsClassifier,
    state_embedding: Tensor,
    tactic_logits: Tensor,
    pool: CandidatePool,
    id_to_tactic: dict[int, str],
    *,
    top_k: int,
    tactic_k: int | None = None,
    beam_size: int = 8,
    candidate_filter: Callable[[int, CandidateRef], bool] | None = None,
    tactic_filter: Callable[[int], bool] | None = None,
) -> list[RankedAction]:
    """Globally rank complete ``(tactic, ordered arguments...)`` actions."""
    if top_k < 1 or beam_size < 1:
        raise ValueError("top_k and beam_size must be positive")
    if state_embedding.shape != (1, model.hidden_dim):
        raise ValueError("state_embedding must have shape [1, hidden_dim]")
    logits = tactic_logits.view(-1)
    tactic_limit = logits.numel() if tactic_k is None else min(
        logits.numel(), max(int(tactic_k), 1)
    )
    tactic_log_probs = F.log_softmax(logits, dim=0)
    tactic_ids = tactic_log_probs.topk(tactic_limit).indices.tolist()
    complete: list[RankedAction] = []

    for tactic_id in tactic_ids:
        if tactic_filter is not None and not tactic_filter(tactic_id):
            continue
        allowed_positions = [
            position
            for position, candidate in enumerate(pool.candidates)
            if candidate_filter is None or candidate_filter(tactic_id, candidate)
        ]
        tactic_embedding = model.tactic_embedding(
            torch.tensor([tactic_id], device=state_embedding.device)
        )
        initial_state = model.argument_selector.initial_state(
            state_embedding, tactic_embedding
        )
        beams = [
            _Beam(
                decoder_state=initial_state,
                positions=(),
                log_probability=float(tactic_log_probs[tactic_id].item()),
            )
        ]
        for step in range(model.max_args + 1):
            next_beams: list[_Beam] = []
            for beam in beams:
                stop_logit = model.stop_head(beam.decoder_state).view(())
                complete.append(
                    RankedAction(
                        tactic_id=tactic_id,
                        tactic_name=id_to_tactic.get(tactic_id, "<UNK>"),
                        arguments=tuple(
                            pool.candidates[position] for position in beam.positions
                        ),
                        log_probability=(
                            beam.log_probability
                            + float(F.logsigmoid(stop_logit).item())
                        ),
                    )
                )
                if step == model.max_args or len(beam.positions) == len(allowed_positions):
                    continue
                scores = model.argument_selector.score_candidates(
                    beam.decoder_state, pool.candidate_vectors
                ).squeeze(0)
                if beam.positions:
                    mask = torch.zeros_like(scores, dtype=torch.bool)
                    mask[list(beam.positions)] = True
                    scores = scores.masked_fill(mask, float("-inf"))
                if len(allowed_positions) != len(pool.candidates):
                    source_mask = torch.ones_like(scores, dtype=torch.bool)
                    source_mask[allowed_positions] = False
                    scores = scores.masked_fill(source_mask, float("-inf"))
                available = int(torch.isfinite(scores).sum().item())
                if available == 0:
                    continue
                candidate_log_probs = F.log_softmax(scores, dim=0)
                for position in candidate_log_probs.topk(
                    min(beam_size, available)
                ).indices.tolist():
                    decoder_state = model.argument_selector.gru(
                        pool.candidate_vectors[position].unsqueeze(0),
                        beam.decoder_state,
                    )
                    next_beams.append(
                        _Beam(
                            decoder_state=decoder_state,
                            positions=(*beam.positions, position),
                            log_probability=(
                                beam.log_probability
                                + float(F.logsigmoid(-stop_logit).item())
                                + float(candidate_log_probs[position].item())
                            ),
                        )
                    )
            next_beams.sort(key=lambda beam: beam.log_probability, reverse=True)
            beams = next_beams[:beam_size]
            if not beams:
                break

    # Different beam paths cannot produce the same typed sequence, but tactic
    # loops can when a pool is empty. Deduplicate by stable action identity.
    best_by_key: dict[tuple, RankedAction] = {}
    for action in complete:
        previous = best_by_key.get(action.stable_key)
        if previous is None or action.log_probability > previous.log_probability:
            best_by_key[action.stable_key] = action
    return sorted(
        best_by_key.values(), key=lambda action: action.log_probability, reverse=True
    )[:top_k]


def complete_action_metrics(
    ranked_actions: Sequence[Sequence[RankedAction]],
    targets: Sequence[ActionTarget],
    *,
    ks: Sequence[int] = (1, 5),
) -> dict[str, object]:
    """Report exact complete actions, source accuracy, recall, and per-tactic."""
    if len(ranked_actions) != len(targets):
        raise ValueError("ranked actions and targets must have equal rows")
    clean_ks = tuple(sorted(set(int(k) for k in ks)))
    if not clean_ks or clean_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")
    hits = {k: 0 for k in clean_ks}
    exact = 0
    source_correct = 0
    source_count = 0
    per_tactic: dict[int, dict[str, int]] = {}
    for actions, target in zip(ranked_actions, targets):
        keys = [action.stable_key for action in actions]
        rank = (
            keys.index(target.stable_key) + 1
            if target.resolved and target.stable_key in keys
            else None
        )
        is_exact = int(rank == 1)
        exact += is_exact
        for k in clean_ks:
            hits[k] += int(rank is not None and rank <= k)
        tactic_metrics = per_tactic.setdefault(
            target.tactic_id, {"count": 0, "exact": 0, "top_k": 0}
        )
        tactic_metrics["count"] += 1
        tactic_metrics["exact"] += is_exact
        tactic_metrics["top_k"] += int(rank is not None and rank <= clean_ks[-1])
        predicted_args = actions[0].arguments if actions else ()
        expected_count = (
            len(target.arguments)
            if target.argument_count is None
            else target.argument_count
        )
        for position in range(expected_count):
            source_correct += int(
                target.resolved
                and bool(actions)
                and actions[0].tactic_id == target.tactic_id
                and position < len(predicted_args)
                and position < len(target.arguments)
                and predicted_args[position].source is target.arguments[position].source
            )
            source_count += 1
    denominator = max(len(targets), 1)
    result: dict[str, object] = {
        "action_count": len(targets),
        "complete_action_exact_match": exact / denominator,
        "candidate_source_accuracy": source_correct / max(source_count, 1),
        "candidate_source_count": source_count,
        "per_tactic": per_tactic,
    }
    for k, count in hits.items():
        result[f"action_recall_at_{k}"] = count / denominator
    return result
