"""Premise-aware training and evaluation loops.

These extend the argument-aware loops in ``argument_training.py`` by adding
premise ranking loss from the unified candidate pool.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.loader import DataLoader

from .argument_selector import TacticWithArgsClassifier, compute_combined_loss
from .labels import parse_tactic_arguments
from .graph import BINDER_KIND_FORALL
from .lemma_index import LemmaIndex
from .premise_pool import (
    CandidateRef,
    CandidateSource,
    build_unified_pools,
    ensure_library_targets,
)
from .premise_scoring import PremiseScorer
from .reporting import console_print
from .unified_reranking import (
    ActionTarget,
    complete_action_metrics,
    compute_unified_reranking_loss,
    rank_complete_actions,
    rank_fresh_name_action,
    resolve_ordered_pool_targets,
)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining_seconds:.0f}s"


def _should_log_batch(
    batch_index: int, total_batches: int, *, log_every_batches: int
) -> bool:
    return (
        batch_index == 1
        or batch_index == total_batches
        or batch_index % log_every_batches == 0
    )


def _extract_tactic_names(batch) -> list[str]:
    """Extract per-sample tactic family names from a PyG Batch."""
    if hasattr(batch, "tactic_name"):
        names = batch.tactic_name
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        return [str(names)]
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    return [""] * batch_size


def _extract_arg_targets(
    batch, max_args: int, device: torch.device
) -> torch.Tensor:
    """Extract ground-truth argument node indices [B, max_args], padded with -1."""
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    targets = torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)
    
    if hasattr(batch, "arg_node_indices") and hasattr(batch, "arg_count"):
        flat_targets = batch.arg_node_indices.to(device=device, dtype=torch.long)
        counts = batch.arg_count.tolist()
        offset = 0
        for i, count in enumerate(counts):
            n_copy = min(count, max_args)
            if n_copy > 0:
                targets[i, :n_copy] = flat_targets[offset : offset + n_copy]
            offset += count
            
    return targets


def _local_to_global_arg_targets(
    local_targets: torch.Tensor, batch, device: torch.device
) -> torch.Tensor:
    """Convert per-graph node ids to indices in the concatenated PyG batch."""
    global_targets = local_targets.clone()
    valid = global_targets >= 0
    graph_sizes = (batch.ptr[1:] - batch.ptr[:-1]).to(
        device=device, dtype=torch.long
    ).unsqueeze(1)
    invalid = valid & (global_targets >= graph_sizes)
    if invalid.any():
        row, column = invalid.nonzero(as_tuple=False)[0].tolist()
        raise ValueError(
            f"argument node index {int(global_targets[row, column].item())} "
            f"is outside graph {row} ({int(graph_sizes[row, 0].item())} nodes)"
        )
    offsets = batch.ptr[:-1].to(device=device, dtype=torch.long).unsqueeze(1)
    global_targets[valid] += offsets.expand_as(global_targets)[valid]
    return global_targets


def _extract_arg_lemma_ids(
    batch, max_args: int, device: torch.device
) -> torch.Tensor:
    """Extract ground-truth lemma IDs [B, max_args], padded with -1."""
    batch_size = int(batch.y.size(0)) if hasattr(batch, "y") else 1
    targets = torch.full((batch_size, max_args), -1, dtype=torch.long, device=device)
    
    if hasattr(batch, "arg_lemma_ids") and hasattr(batch, "arg_count"):
        flat_targets = batch.arg_lemma_ids.to(device=device, dtype=torch.long)
        counts = batch.arg_count.tolist()
        offset = 0
        for i, count in enumerate(counts):
            n_copy = min(count, max_args)
            if n_copy > 0:
                targets[i, :n_copy] = flat_targets[offset : offset + n_copy]
            offset += count
            
    return targets


def _recover_lemma_targets(
    batch, local_targets: torch.Tensor, lemma_targets: torch.Tensor,
    lemma_index: LemmaIndex | None,
) -> torch.Tensor:
    """Resolve external tactic arguments by name when cached lemma IDs are absent."""
    name_to_id = getattr(lemma_index, "name_to_id", {}) if lemma_index is not None else {}
    if not name_to_id or not hasattr(batch, "tactic_raw"):
        return lemma_targets
    raw_tactics = batch.tactic_raw
    if not isinstance(raw_tactics, (list, tuple)):
        raw_tactics = [raw_tactics]
    recovered = lemma_targets.clone()
    for row, raw_tactic in enumerate(raw_tactics):
        _, arguments = parse_tactic_arguments(str(raw_tactic))
        for column, argument in enumerate(arguments[: recovered.size(1)]):
            if local_targets[row, column] < 0 and recovered[row, column] < 0:
                recovered[row, column] = int(name_to_id.get(argument, -1))
    return recovered


def train_one_epoch_with_premises(
    model: TacticWithArgsClassifier,
    scorer: PremiseScorer,
    loader: DataLoader,
    lemma_index: LemmaIndex,
    *,
    retriever=None,
    optimizer: AdamW,
    grad_scaler,
    device: torch.device,
    grad_clip: float,
    unknown_tactic_id: int,
    arg_loss_weight: float,
    premise_loss_weight: float,
    k: int = 500,
    epoch: int,
    total_epochs: int,
    log_every_batches: int,
    use_amp: bool,
    amp_dtype: torch.dtype | None = None,
    pin_memory: bool = False,
) -> dict[str, float | int]:
    """Train one epoch with combined tactic + argument + premise ranking loss."""
    model.train()
    model.backbone.eval()  # frozen backbone stays in eval mode
    scorer.train()
    if retriever is not None:
        retriever.eval()

    total_tactic_loss = 0.0
    total_arg_loss = 0.0
    total_premise_loss = 0.0
    total_combined_loss = 0.0
    total_examples = 0
    total_reranking_targets = 0
    total_reranking_scored = 0
    total_unresolved_targets = 0
    total_source_correct_weighted = 0.0
    total_batches = len(loader)
    start_time = time.perf_counter()

    console_print(
        f"  Starting epoch {epoch:02d}/{total_epochs:02d} "
        f"with {total_batches} train batches (premise-aware)..."
    )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(
            device, non_blocking=(device.type == "cuda" and pin_memory)
        )
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)
        arg_targets = _extract_arg_targets(batch, model.max_args, device)
        pointer_arg_targets = _local_to_global_arg_targets(arg_targets, batch, device)
        arg_lemma_targets = _extract_arg_lemma_ids(batch, model.max_args, device)
        arg_lemma_targets = _recover_lemma_targets(
            batch, arg_targets, arg_lemma_targets, lemma_index
        )
        arg_counts = [int(value) for value in batch.arg_count.tolist()]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            # Forward pass through model
            tactic_logits, arg_logits_list, stop_logits_list = model(
                batch,
                teacher_tactic_ids=targets,
                arg_targets=pointer_arg_targets,
            )

            # Combined tactic + argument loss
            ta_loss, ta_metrics = compute_combined_loss(
                tactic_logits,
                arg_logits_list,
                targets,
                pointer_arg_targets,
                batch.batch,
                arg_count_per_sample=arg_counts,
                stop_logits_list=stop_logits_list,
                arg_loss_weight=arg_loss_weight,
                unknown_tactic_id=unknown_tactic_id,
            )

            # Recompute embeddings (detached) for the premise scoring branch
            with torch.no_grad():
                node_embeddings = model.backbone.encode_nodes(batch)
                state_emb = model.backbone.readout(node_embeddings, batch)
                retrieval_state_emb = (
                    state_emb
                    if retriever is None
                    else retriever.encode_states(batch)
                )
            node_embeddings = node_embeddings.detach()
            state_emb = state_emb.detach()

            # Build premise mask
            premise_mask = batch.premise_mask.to(
                dtype=torch.bool, device=device
            )

            # Build unified candidate pools
            pools = build_unified_pools(
                state_emb,
                node_embeddings,
                premise_mask,
                batch.batch,
                lemma_index=lemma_index,
                k=k,
                retrieval_state_vecs=retrieval_state_emb,
            )
            pools = ensure_library_targets(pools, lemma_index, arg_lemma_targets)

            target_positions, target_metrics = resolve_ordered_pool_targets(
                pools,
                arg_targets,
                arg_lemma_targets,
                arg_counts,
                max_args=model.max_args,
            )
            # One recurrent pointer now scores both local and library
            # candidates in argument order. The standalone scorer parameter is
            # retained in this function's signature only for old callers and
            # checkpoints; it no longer defines the action path.
            p_loss, p_metrics = compute_unified_reranking_loss(
                model,
                state_emb,
                targets,
                pools,
                target_positions,
            )

            # Total loss
            total_loss = ta_loss + premise_loss_weight * p_loss

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Non-finite premise-aware loss ({float(total_loss):.4g}) "
                f"at batch {batch_index}."
            )

        grad_scaler.scale(total_loss).backward()
        grad_scaler.unscale_(optimizer)
        trainable_params = [p for p in list(model.parameters()) + list(scorer.parameters()) if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        grad_scaler.step(optimizer)
        grad_scaler.update()

        batch_size = int(targets.numel())
        total_tactic_loss += ta_metrics["tactic_loss"] * batch_size
        total_arg_loss += ta_metrics["arg_loss"] * batch_size
        total_premise_loss += p_metrics["reranking_loss"] * batch_size
        total_combined_loss += float(total_loss.item()) * batch_size
        total_examples += batch_size
        total_reranking_targets += int(p_metrics["target_count"])
        total_reranking_scored += int(p_metrics["scored_target_count"])
        total_unresolved_targets += int(target_metrics["unresolved_target_count"])
        total_source_correct_weighted += (
            float(p_metrics["candidate_source_accuracy"])
            * int(p_metrics["scored_target_count"])
        )

        if _should_log_batch(
            batch_index, total_batches, log_every_batches=log_every_batches
        ):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            n = max(total_examples, 1)
            console_print(
                f"    train batch {batch_index:>5}/{total_batches} | "
                f"seen={total_examples} | "
                f"tac={total_tactic_loss / n:.4f} "
                f"arg={total_arg_loss / n:.4f} "
                f"prem={total_premise_loss / n:.4f} | "
                f"elapsed={elapsed}"
            )

    n = max(total_examples, 1)
    return {
        "tactic_loss": total_tactic_loss / n,
        "arg_loss": total_arg_loss / n,
        "premise_loss": total_premise_loss / n,
        "combined_loss": total_combined_loss / n,
        "example_count": total_examples,
        "reranking_target_count": total_reranking_targets,
        "reranking_scored_target_count": total_reranking_scored,
        "reranking_target_coverage": total_reranking_scored
        / max(total_reranking_targets, 1),
        "reranking_unresolved_target_count": total_unresolved_targets,
        "candidate_source_accuracy": total_source_correct_weighted
        / max(total_reranking_scored, 1),
    }


@torch.no_grad()
def evaluate_model_with_premises(
    model: TacticWithArgsClassifier,
    scorer: PremiseScorer,
    loader: DataLoader,
    lemma_index: LemmaIndex,
    *,
    retriever=None,
    device: torch.device,
    unknown_tactic_id: int,
    arg_loss_weight: float,
    premise_loss_weight: float,
    tactic_vocab: dict[str, int] | None = None,
    k: int = 500,
    split_name: str | None = None,
    log_every_batches: int | None = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
    pin_memory: bool = False,
) -> dict[str, float | int]:
    """Evaluate model with combined tactic + argument + premise metrics."""
    model.eval()
    scorer.eval()
    if retriever is not None:
        retriever.eval()

    total_tactic_loss = 0.0
    total_arg_loss = 0.0
    total_premise_loss = 0.0
    total_combined_loss = 0.0
    top1_correct = 0
    known_count = 0

    reranking_target_count = 0
    reranking_scored_count = 0
    reranking_unresolved_count = 0
    reranking_top1_weighted = 0.0
    predicted_ranked_actions = []
    oracle_ranked_actions = []
    action_targets: list[ActionTarget] = []

    total_count = 0
    total_batches = len(loader)
    start_time = time.perf_counter()

    if split_name is not None:
        console_print(
            f"  Evaluating {split_name} split "
            f"({total_batches} batches, premise-aware)..."
        )

    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(
            device, non_blocking=(device.type == "cuda" and pin_memory)
        )
        targets = batch.y.view(-1)
        tactic_names = _extract_tactic_names(batch)
        arg_targets = _extract_arg_targets(batch, model.max_args, device)
        pointer_arg_targets = _local_to_global_arg_targets(arg_targets, batch, device)
        arg_lemma_targets = _extract_arg_lemma_ids(batch, model.max_args, device)
        arg_lemma_targets = _recover_lemma_targets(
            batch, arg_targets, arg_lemma_targets, lemma_index
        )
        arg_counts = [int(value) for value in batch.arg_count.tolist()]

        with torch.amp.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            tactic_logits, arg_logits_list, stop_logits_list = model(batch)
            ta_loss, ta_metrics = compute_combined_loss(
                tactic_logits,
                arg_logits_list,
                targets,
                pointer_arg_targets,
                batch.batch,
                arg_count_per_sample=arg_counts,
                stop_logits_list=stop_logits_list,
                arg_loss_weight=arg_loss_weight,
                unknown_tactic_id=unknown_tactic_id,
            )

            with torch.no_grad():
                node_embeddings = model.backbone.encode_nodes(batch)
                state_emb = model.backbone.readout(node_embeddings, batch)
                retrieval_state_emb = (
                    state_emb
                    if retriever is None
                    else retriever.encode_states(batch)
                )
            premise_mask = batch.premise_mask.to(
                dtype=torch.bool, device=device
            )

            pools = build_unified_pools(
                state_emb,
                node_embeddings,
                premise_mask,
                batch.batch,
                lemma_index=lemma_index,
                k=k,
                retrieval_state_vecs=retrieval_state_emb,
            )
            target_positions, target_metrics = resolve_ordered_pool_targets(
                pools,
                arg_targets,
                arg_lemma_targets,
                arg_counts,
                max_args=model.max_args,
            )
            p_loss, p_metrics = compute_unified_reranking_loss(
                model,
                state_emb,
                targets,
                pools,
                target_positions,
            )

        id_to_tactic = (
            {tactic_id: name for name, tactic_id in tactic_vocab.items()}
            if tactic_vocab is not None
            else {
                tactic_id: str(tactic_id)
                for tactic_id in range(tactic_logits.size(1))
            }
        )
        fresh_tactic_ids = {
            tactic_id
            for tactic_id, name in id_to_tactic.items()
            if name in {"intro", "rintro", "introV2"}
        }
        for row, (pool, positions) in enumerate(zip(pools, target_positions)):
            expected_count = min(arg_counts[row], model.max_args)
            resolved = not (
                arg_counts[row] > model.max_args
                or len(positions) != expected_count
                or any(position < 0 for position in positions)
            )
            expected_arguments = (
                tuple(pool.candidates[position] for position in positions)
                if resolved
                else ()
            )
            action_targets.append(
                ActionTarget(
                    tactic_id=int(targets[row].item()),
                    arguments=expected_arguments,
                    resolved=resolved,
                    argument_count=arg_counts[row],
                )
            )
            graph_nodes = (batch.batch == row).nonzero(as_tuple=False).view(-1)
            graph_offset = int(graph_nodes[0].item()) if graph_nodes.numel() else 0
            source_nodes = set(int(value) for value in batch.edge_index[0].tolist())
            fresh_global_ids = [
                int(node_id)
                for node_id in graph_nodes.tolist()
                if int(batch.is_bound[node_id].item()) == 1
                and int(batch.binder_kind[node_id].item()) == BINDER_KIND_FORALL
                and int(batch.binder_depth[node_id].item()) == 1
                and int(node_id) not in source_nodes
            ] if all(
                hasattr(batch, field)
                for field in ("is_bound", "binder_kind", "binder_depth")
            ) else []
            fresh_candidates = [
                CandidateRef(
                    source=CandidateSource.LOCAL,
                    stable_id=node_id - graph_offset,
                    graph_id=row,
                    local_node_index=node_id - graph_offset,
                    metadata={"fresh_name": True},
                )
                for node_id in fresh_global_ids
            ]
            fresh_vectors = (
                node_embeddings[fresh_global_ids]
                if fresh_global_ids
                else node_embeddings.new_empty((0, model.hidden_dim))
            )

            def ranked_for_logits(row_logits: Tensor, *, tactic_k: int | None = None):
                actions = rank_complete_actions(
                    model,
                    state_emb[row : row + 1],
                    row_logits,
                    pool,
                    id_to_tactic,
                    top_k=5,
                    tactic_k=tactic_k,
                    tactic_filter=lambda tactic_id: tactic_id not in fresh_tactic_ids,
                )
                tactic_log_probs = F.log_softmax(row_logits, dim=0)
                for fresh_tactic_id in fresh_tactic_ids:
                    if not torch.isfinite(tactic_log_probs[fresh_tactic_id]):
                        continue
                    actions.append(
                        rank_fresh_name_action(
                            model,
                            state_emb[row : row + 1],
                            tactic_id=fresh_tactic_id,
                            tactic_name=id_to_tactic[fresh_tactic_id],
                            tactic_log_probability=float(
                                tactic_log_probs[fresh_tactic_id].item()
                            ),
                            candidates=fresh_candidates,
                            candidate_vectors=fresh_vectors,
                        )
                    )
                actions.sort(key=lambda action: action.log_probability, reverse=True)
                return actions[:5]

            predicted_ranked_actions.append(ranked_for_logits(tactic_logits[row]))
            oracle_logits = torch.full_like(tactic_logits[row], float("-inf"))
            oracle_logits[targets[row]] = 0.0
            oracle_ranked_actions.append(ranked_for_logits(oracle_logits, tactic_k=1))

        bs = int(targets.numel())
        total_tactic_loss += ta_metrics["tactic_loss"] * bs
        total_arg_loss += ta_metrics["arg_loss"] * bs
        total_premise_loss += p_metrics["reranking_loss"] * bs
        total_combined_loss += (
            ta_metrics["total_loss"]
            + premise_loss_weight * p_metrics["reranking_loss"]
        ) * bs
        reranking_target_count += int(p_metrics["target_count"])
        reranking_scored_count += int(p_metrics["scored_target_count"])
        reranking_unresolved_count += int(target_metrics["unresolved_target_count"])
        reranking_top1_weighted += (
            float(p_metrics["candidate_top1_accuracy"])
            * int(p_metrics["scored_target_count"])
        )

        # Tactic top-1 accuracy (excluding UNK)
        known_mask = targets != unknown_tactic_id
        kc = int(known_mask.sum().item())
        if kc > 0:
            preds = tactic_logits[known_mask].argmax(dim=1)
            top1_correct += int((preds == targets[known_mask]).sum().item())
        known_count += kc
        total_count += bs

        if (
            split_name is not None
            and log_every_batches is not None
            and _should_log_batch(
                batch_index,
                total_batches,
                log_every_batches=log_every_batches,
            )
        ):
            elapsed = _format_elapsed(time.perf_counter() - start_time)
            console_print(
                f"    {split_name} batch {batch_index:>5}/{total_batches} | "
                f"known={known_count} | elapsed={elapsed}"
            )

    n = max(total_count, 1)
    predicted_action_metrics = complete_action_metrics(
        predicted_ranked_actions, action_targets, ks=(1, 5)
    )
    oracle_action_metrics = complete_action_metrics(
        oracle_ranked_actions, action_targets, ks=(1, 5)
    )
    return {
        "tactic_loss": total_tactic_loss / n,
        "arg_loss": total_arg_loss / n,
        "premise_loss": total_premise_loss / n,
        "combined_loss": total_combined_loss / n,
        "tactic_top1_accuracy": top1_correct / max(known_count, 1),
        "reranking_target_coverage": reranking_scored_count
        / max(reranking_target_count, 1),
        "reranking_target_count": reranking_target_count,
        "reranking_scored_target_count": reranking_scored_count,
        "reranking_unresolved_target_count": reranking_unresolved_count,
        "candidate_top1_accuracy": reranking_top1_weighted
        / max(reranking_scored_count, 1),
        "complete_action_exact_match_predicted_tactic": predicted_action_metrics[
            "complete_action_exact_match"
        ],
        "complete_action_exact_match_oracle_tactic": oracle_action_metrics[
            "complete_action_exact_match"
        ],
        "action_recall_at_1_predicted_tactic": predicted_action_metrics[
            "action_recall_at_1"
        ],
        "action_recall_at_5_predicted_tactic": predicted_action_metrics[
            "action_recall_at_5"
        ],
        "action_recall_at_1_oracle_tactic": oracle_action_metrics[
            "action_recall_at_1"
        ],
        "action_recall_at_5_oracle_tactic": oracle_action_metrics[
            "action_recall_at_5"
        ],
        "candidate_source_accuracy_predicted_tactic": predicted_action_metrics[
            "candidate_source_accuracy"
        ],
        "candidate_source_accuracy_oracle_tactic": oracle_action_metrics[
            "candidate_source_accuracy"
        ],
        "complete_action_per_tactic": predicted_action_metrics["per_tactic"],
        "complete_action_labeled_count": len(action_targets),
        # Compatibility aliases for old dashboards. These now describe the
        # complete-action decoder rather than the retired one-premise scorer.
        "premise_recall": reranking_scored_count / max(reranking_target_count, 1),
        "premise_mrr": predicted_action_metrics["complete_action_exact_match"],
        "premise_top1_accuracy": reranking_top1_weighted
        / max(reranking_scored_count, 1),
        "premise_top5_accuracy": predicted_action_metrics["action_recall_at_5"],
        "known_label_count": known_count,
        "premise_target_present_count": reranking_target_count,
        "premise_valid_count": reranking_scored_count,
        "evaluated_count": total_count,
    }
