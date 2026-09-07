"""Pointer-network argument selector for tactic argument prediction.

This module is additive — the existing ``GraphSAGEStateClassifier`` in
``model.py`` is used as a backbone and remains untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .model import GATv2StateClassifier, GraphSAGEStateClassifier
from .pyg import NODE_TYPE_TO_ID


# ---------------------------------------------------------------------------
# ArgumentSelector: Scaled dot-product pointer head
# ---------------------------------------------------------------------------


class ArgumentSelector(nn.Module):
    """Score every node in the DAG as a candidate tactic argument.

    **Query** = concat(state_embedding, tactic_embedding) projected to key-space.
    **Keys**  = node_embeddings from the GNN encoder.

    After selecting argument *k*, the selected node's embedding is fused into
    the query before selecting argument *k + 1* (autoregressive).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.init_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self._scale = 1.0 / math.sqrt(hidden_dim)

    def initial_state(self, state_emb: Tensor, tactic_emb: Tensor) -> Tensor:
        return torch.tanh(self.init_proj(torch.cat([state_emb, tactic_emb], dim=1)))

    def score_candidates(self, decoder_state: Tensor, candidate_embeddings: Tensor) -> Tensor:
        """Score one candidate pool for a batch of decoder states."""
        query = self.out_proj(decoder_state)
        return (candidate_embeddings @ query.unsqueeze(-1)).squeeze(-1) * self._scale

    def forward(
        self,
        state_emb: Tensor,        # [B, H]
        tactic_emb: Tensor,        # [B, H]
        node_embeddings: Tensor,   # [total_nodes, H]
        premise_mask: Tensor,      # [total_nodes]  bool
        batch_index: Tensor,       # [total_nodes]  → which graph each node belongs to
        decoder_state: Tensor | None = None,
        excluded_positions: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(arg_logits, selected_emb)``.

        ``arg_logits``   — shape ``[B, max_nodes_in_batch]``, padded with -inf.
        ``selected_emb`` — shape ``[B, H]``, the embedding of the argmax node
                           (used as context for the next autoregressive step).
        """
        batch_size = state_emb.size(0)
        hidden_dim = state_emb.size(1)
        device = state_emb.device

        # --- 1. Build the query vector --------------------------------
        if decoder_state is None:
            decoder_state = self.initial_state(state_emb, tactic_emb)
        query = self.out_proj(decoder_state)

        # --- 2. Scatter node embeddings into a padded [B, N_max, H] tensor -
        #     where N_max is the max number of nodes in any graph in the batch.
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        counts.scatter_add_(0, batch_index, torch.ones_like(batch_index))
        max_nodes = int(counts.max().item())

        # Compute per-node offset within its graph
        offsets = torch.zeros_like(batch_index)
        for b in range(batch_size):
            graph_mask = batch_index == b
            offsets[graph_mask] = torch.arange(
                int(graph_mask.sum().item()), device=device, dtype=torch.long
            )

        # Padded node embedding matrix
        padded_keys = torch.zeros(batch_size, max_nodes, hidden_dim, device=device, dtype=node_embeddings.dtype)
        padded_keys[batch_index, offsets] = node_embeddings

        # Padded premise mask (False = invalid ⇒ will be masked to -inf)
        padded_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=device)
        padded_mask[batch_index, offsets] = premise_mask

        # --- 3. Scaled dot-product attention scores -------------------
        # query: [B, H] → [B, 1, H];  keys: [B, N_max, H]
        scores = torch.bmm(query.unsqueeze(1), padded_keys.transpose(1, 2)).squeeze(1)  # [B, N_max]
        scores = scores * self._scale

        # Mask out non-premise positions
        scores = scores.masked_fill(~padded_mask, float("-inf"))
        if excluded_positions is not None:
            scores = scores.masked_fill(excluded_positions, float("-inf"))

        # --- 4. Selected node embedding for autoregressive context ----
        with torch.no_grad():
            selected_idx = scores.argmax(dim=1)  # [B]

        selected_emb = padded_keys[torch.arange(batch_size, device=device), selected_idx]  # [B, H]

        return scores, selected_emb


# ---------------------------------------------------------------------------
# TacticWithArgsClassifier: Full model wrapping backbone + pointer head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TacticWithArgsConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.2
    max_args: int = 3
    arg_loss_weight: float = 0.5
    heads: int = 8
    readout: str = "state"
    teacher_forcing: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "max_args": self.max_args,
            "arg_loss_weight": self.arg_loss_weight,
            "heads": self.heads,
            "readout": self.readout,
            "teacher_forcing": self.teacher_forcing,
        }


class TacticWithArgsClassifier(nn.Module):
    """Tactic family prediction + pointer-based argument selection.

    A GraphSAGE or GATv2 backbone is instantiated internally and its
    ``encode_nodes`` / ``readout`` methods are reused.  The tactic
    classification head is inherited from the backbone.  A new
    ``ArgumentSelector`` pointer head is added on top.
    """

    def __init__(
        self,
        *,
        num_node_labels: int,
        num_tactics: int,
        num_node_types: int = len(NODE_TYPE_TO_ID),
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        use_node_type: bool = True,
        max_args: int = 3,
        gnn_type: str = "sage",
        heads: int = 8,
        readout: str = "state",
        teacher_forcing: bool = True,
    ) -> None:
        super().__init__()

        # Backbone — shared encoder + tactic head
        backbone_kwargs = dict(
            num_node_labels=num_node_labels,
            num_tactics=num_tactics,
            num_node_types=num_node_types,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_node_type=use_node_type,
        )
        if gnn_type == "gat":
            self.backbone = GATv2StateClassifier(
                heads=heads,
                readout=readout,
                **backbone_kwargs,
            )
        else:
            self.backbone = GraphSAGEStateClassifier(**backbone_kwargs)

        # Tactic embedding (one learned vector per tactic family)
        self.tactic_embedding = nn.Embedding(num_tactics, hidden_dim)

        # Pointer head
        self.argument_selector = ArgumentSelector(hidden_dim)

        self.max_args = max_args
        self.hidden_dim = hidden_dim
        self.teacher_forcing = teacher_forcing
        self.stop_head = nn.Linear(hidden_dim, 1)

    # ---- convenience accessors for the backbone ----
    @property
    def label_embedding(self) -> nn.Embedding:
        return self.backbone.label_embedding

    @property
    def node_type_embedding(self) -> nn.Embedding | None:
        return self.backbone.node_type_embedding

    def forward(
        self,
        data,
        *,
        teacher_tactic_ids: Tensor | None = None,
        arg_targets: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Return tactic logits, argument logits, and stop logits.

        Parameters
        ----------
        data : PyG Batch
            Must include ``premise_mask`` (bool tensor, one per node) and
            ``batch`` (standard PyG batch vector).
        teacher_tactic_ids : Tensor, optional
            Ground-truth tactic ids for teacher-forcing during training.
            Shape ``[B]``.
        arg_targets : Tensor, optional
            Global node indices used for teacher forcing, shaped ``[B, max_args]``.

        Returns
        -------
        tactic_logits : Tensor, shape ``[B, num_tactics]``
        arg_logits_list : list[Tensor]
            One entry per argument step (up to ``max_args``).
            Each has shape ``[B, max_nodes_in_batch]``.
        """
        # 1. Encode all nodes
        node_embeddings = self.backbone.encode_nodes(data)  # [total_nodes, H]

        # 2. Readout the State node embedding per graph
        state_emb = self.backbone.readout(node_embeddings, data)  # [B, H]

        # 3. Tactic classification
        tactic_logits = self.backbone.classifier(
            self.backbone.dropout(state_emb)
        )  # [B, num_tactics]

        # 4. Determine which tactic embedding to use as query context
        if teacher_tactic_ids is not None:
            tactic_ids = teacher_tactic_ids
        else:
            tactic_ids = tactic_logits.argmax(dim=1)  # [B]

        tactic_emb = self.tactic_embedding(tactic_ids)  # [B, H]

        # 5. Always expose the bounded decoder positions. The stop head,
        # supervised by each sample's arg_count, controls the sequence length.
        if hasattr(data, "premise_mask") and data.premise_mask is not None:
            premise_mask = data.premise_mask.to(device=node_embeddings.device)
        else:
            # Fallback if no cached premise mask exists: only allow var (0), type (1), and predicate (2)
            node_types = data.node_type.to(device=node_embeddings.device)
            premise_mask = (node_types >= 0) & (node_types <= 2)
        batch_index = data.batch.to(device=node_embeddings.device)

        arg_logits_list: list[Tensor] = []
        stop_logits_list: list[Tensor] = []
        decoder_state = self.argument_selector.initial_state(state_emb, tactic_emb)
        counts = torch.bincount(batch_index, minlength=state_emb.size(0))
        max_nodes = int(counts.max().item())
        excluded_positions = torch.zeros(
            state_emb.size(0), max_nodes, dtype=torch.bool, device=node_embeddings.device
        )
        node_offsets = torch.zeros_like(batch_index)
        for graph_index in range(state_emb.size(0)):
            graph_mask = batch_index == graph_index
            node_offsets[graph_mask] = torch.arange(
                int(graph_mask.sum().item()), device=node_embeddings.device
            )

        for step in range(self.max_args + 1):
            stop_logits_list.append(self.stop_head(decoder_state).squeeze(-1))
            if step == self.max_args:
                break
            scores, selected_emb = self.argument_selector(
                state_emb, tactic_emb, node_embeddings, premise_mask, batch_index,
                decoder_state=decoder_state,
                excluded_positions=excluded_positions,
            )
            arg_logits_list.append(scores)

            if self.teacher_forcing and arg_targets is not None and step < arg_targets.size(1):
                previous_indices = arg_targets[:, step]
                valid = previous_indices >= 0
                next_input = torch.zeros_like(selected_emb)
                if valid.any():
                    next_input[valid] = node_embeddings[previous_indices[valid]]
                    graph_ids = torch.arange(
                        state_emb.size(0), device=node_embeddings.device
                    )[valid]
                    excluded_positions[graph_ids, node_offsets[previous_indices[valid]]] = True
            else:
                next_input = selected_emb
                with torch.no_grad():
                    selected_positions = scores.argmax(dim=1)
                    excluded_positions.scatter_(1, selected_positions.unsqueeze(1), True)
            decoder_state = self.argument_selector.gru(next_input, decoder_state)

        return tactic_logits, arg_logits_list, stop_logits_list


# ---------------------------------------------------------------------------
# Combined loss computation
# ---------------------------------------------------------------------------


def resolve_arg_targets_to_padded(
    arg_node_indices: Tensor,
    batch_index: Tensor,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Remap global node indices to padded per-graph positions.

    Returns [B, max_gt_args] of positions into the padded [B, N_max] logit
    matrix, with -1 for invalid arguments.
    """
    offsets = torch.zeros_like(batch_index)
    for b in range(batch_size):
        graph_mask = batch_index == b
        offsets[graph_mask] = torch.arange(
            int(graph_mask.sum().item()), device=device, dtype=torch.long
        )

    result = arg_node_indices.clone().to(device)
    valid = result >= 0
    result[~valid] = -1

    flat_valid_indices = result[valid]
    total_nodes = batch_index.size(0)
    oob = (flat_valid_indices >= total_nodes)
    if oob.any():
        temp = result.clone()
        temp[valid] = torch.where(oob, torch.tensor(-1, device=device), offsets[flat_valid_indices.clamp(max=total_nodes - 1)])
        return temp

    result[valid] = offsets[flat_valid_indices]
    return result


def compute_combined_loss(
    tactic_logits: Tensor,
    arg_logits_list: list[Tensor],
    tactic_targets: Tensor,
    arg_targets: Tensor,
    batch_index: Tensor,
    *,
    arg_count_per_sample: list[int],
    stop_logits_list: list[Tensor] | None = None,
    arg_loss_weight: float = 0.5,
    unknown_tactic_id: int = 0,
    node_labels: Tensor | None = None,
    node_types: Tensor | None = None,
) -> tuple[Tensor, dict[str, float | int]]:
    """Tactic classification, sequence argument, and stop-decision loss."""
    device = tactic_logits.device
    batch_size = tactic_logits.size(0)

    known_mask = tactic_targets != unknown_tactic_id
    if known_mask.any():
        tactic_loss = F.cross_entropy(tactic_logits[known_mask], tactic_targets[known_mask])
    else:
        tactic_loss = torch.tensor(0.0, device=device)

    if not arg_logits_list:
        return tactic_loss, {
            "tactic_loss": float(tactic_loss.item()),
            "arg_loss": 0.0,
            "total_loss": float(tactic_loss.item()),
            "arg_top1_correct": 0,
            "arg_top5_correct": 0,
            "arg_valid_count": 0,
            "arg_target_count": 0,
            "arg_top1_accuracy": 0.0,
            "arg_top5_accuracy": 0.0,
            "arg_target_coverage": 0.0,
            "arg_exact_sequence_correct": 0,
            "arg_sequence_count": 0,
            "arg_exact_sequence_flags": [],
            "arg_position_top1_accuracy": 0.0,
            "arg_position_top5_accuracy": 0.0,
            "stop_loss": 0.0,
            "stop_accuracy": 0.0,
        }

    padded_targets = resolve_arg_targets_to_padded(
        arg_targets, batch_index, batch_size, device
    )

    arg_losses: list[Tensor] = []
    arg_top1_correct = 0
    arg_top5_correct = 0
    arg_valid_count = 0
    arg_target_count = sum(arg_count_per_sample)
    exact_sequence_correct = 0
    sequence_count = batch_size
    exact_sequence_flags = [0] * batch_size
    for step_k, arg_logits_k in enumerate(arg_logits_list):
        if step_k >= padded_targets.size(1):
            break

        gt_k = padded_targets[:, step_k]  # [B]
        valid = gt_k >= 0
        for b_idx in range(batch_size):
            if arg_count_per_sample[b_idx] <= step_k:
                valid[b_idx] = False
            if valid[b_idx]:
                # Skip target if it was masked out by premise_mask
                if torch.isneginf(arg_logits_k[b_idx, gt_k[b_idx]]):
                    valid[b_idx] = False

        if not valid.any():
            continue

        step_loss = F.cross_entropy(arg_logits_k[valid].clamp(min=-1e4), gt_k[valid])
        predictions = arg_logits_k[valid].argmax(dim=1)
        arg_top1_correct += int((predictions == gt_k[valid]).sum().item())
        top_k = min(5, int(arg_logits_k.size(1)))
        top_indices = arg_logits_k[valid].topk(top_k, dim=1).indices
        arg_top5_correct += int(
            (top_indices == gt_k[valid].unsqueeze(1)).any(dim=1).sum().item()
        )
        arg_valid_count += int(valid.sum().item())
        arg_losses.append(step_loss)

    if arg_losses:
        arg_loss = torch.stack(arg_losses).mean()
    else:
        arg_loss = torch.tensor(0.0, device=device)

    stop_loss = torch.tensor(0.0, device=device)
    stop_correct = 0
    stop_count = 0
    if stop_logits_list:
        stop_targets = torch.tensor(
            [[step >= count for step in range(len(stop_logits_list))]
             for count in arg_count_per_sample],
            device=device,
            dtype=torch.float32,
        )
        stop_logits = torch.stack(stop_logits_list, dim=1)
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_targets)
        stop_correct = int(((stop_logits >= 0) == stop_targets.bool()).sum().item())
        stop_count = int(stop_targets.numel())

        for sample_index, target_count in enumerate(arg_count_per_sample):
            predicted_count = len(stop_logits_list)
            for step_index, stop_logit in enumerate(stop_logits_list):
                if stop_logit[sample_index] >= 0:
                    predicted_count = step_index
                    break
            target_sequence = [
                int(padded_targets[sample_index, step].item())
                for step in range(min(target_count, padded_targets.size(1)))
                if padded_targets[sample_index, step] >= 0
            ]
            predicted_sequence = [
                int(arg_logits_list[step][sample_index].argmax().item())
                for step in range(min(predicted_count, len(arg_logits_list)))
            ]
            if predicted_sequence == target_sequence and target_count <= len(arg_logits_list):
                exact_sequence_correct += 1
                exact_sequence_flags[sample_index] = 1

    total_loss = tactic_loss + arg_loss_weight * (arg_loss + stop_loss)
    return total_loss, {
        "tactic_loss": float(tactic_loss.item()),
        "arg_loss": float(arg_loss.item()),
        "total_loss": float(total_loss.item()),
        "arg_top1_correct": arg_top1_correct,
        "arg_top5_correct": arg_top5_correct,
        "arg_valid_count": arg_valid_count,
        "arg_target_count": arg_target_count,
        "arg_top1_accuracy": arg_top1_correct / max(arg_valid_count, 1),
        "arg_top5_accuracy": arg_top5_correct / max(arg_valid_count, 1),
        "arg_target_coverage": arg_valid_count / max(arg_target_count, 1),
        "arg_exact_sequence_correct": exact_sequence_correct,
        "arg_sequence_count": sequence_count,
        "arg_exact_sequence_flags": exact_sequence_flags,
        "arg_position_top1_accuracy": arg_top1_correct / max(arg_valid_count, 1),
        "arg_position_top5_accuracy": arg_top5_correct / max(arg_valid_count, 1),
        "arg_truncated_count": sum(
            max(count - len(arg_logits_list), 0) for count in arg_count_per_sample
        ),
        "arg_truncated_examples": sum(
            count > len(arg_logits_list) for count in arg_count_per_sample
        ),
        "stop_loss": float(stop_loss.item()),
        "stop_accuracy": stop_correct / max(stop_count, 1),
    }
