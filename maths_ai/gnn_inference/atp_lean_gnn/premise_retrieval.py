"""First-stage external-premise retrieval.

Unlike the downstream premise scorer, this module learns the vector space used
by the corpus index itself. Proof states and lemma statements use independent
encoders, then meet in a normalized projection space trained with a
multi-positive contrastive objective.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GraphEncoder(Protocol):
    """Structural interface shared by current GNN backbones."""

    hidden_dim: int

    def encode_nodes(self, graph_batch) -> Tensor: ...

    def readout(self, node_embeddings: Tensor, graph_batch) -> Tensor: ...


class DualEncoderRetriever(nn.Module):
    """Encode proof states and lemma statements into one retrieval space.

    Encoders are deliberately separate modules. Sharing their initial weights is
    useful, but tying them permanently prevents the two towers from specializing
    for structurally different proof-state and declaration graphs.
    """

    def __init__(
        self,
        state_encoder: nn.Module,
        lemma_encoder: nn.Module,
        *,
        hidden_dim: int,
        projection_dim: int | None = None,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        output_dim = hidden_dim if projection_dim is None else projection_dim
        if output_dim <= 0:
            raise ValueError("projection_dim must be positive")
        self.state_encoder = state_encoder
        self.lemma_encoder = lemma_encoder
        self.state_projection = nn.Linear(hidden_dim, output_dim)
        self.lemma_projection = nn.Linear(hidden_dim, output_dim)
        if output_dim == hidden_dim:
            # Start in the pointer-derived index space. This makes the existing
            # index an honest epoch-zero baseline and permits a frozen lemma
            # tower policy without rebuilding 319k declaration vectors after
            # every state-tower update.
            nn.init.eye_(self.state_projection.weight)
            nn.init.zeros_(self.state_projection.bias)
            nn.init.eye_(self.lemma_projection.weight)
            nn.init.zeros_(self.lemma_projection.bias)
        self.output_dim = output_dim
        self.temperature = temperature

    def freeze_lemma_tower(self) -> None:
        """Keep index keys fixed while the proof-state tower learns queries."""
        for parameter in self.lemma_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.lemma_projection.parameters():
            parameter.requires_grad = False
        self.lemma_encoder.eval()
        self.lemma_projection.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.lemma_encoder.parameters()):
            self.lemma_encoder.eval()
            self.lemma_projection.eval()
        return self

    @staticmethod
    def _readout(encoder: GraphEncoder, graph_batch) -> Tensor:
        node_embeddings = encoder.encode_nodes(graph_batch)
        return encoder.readout(node_embeddings, graph_batch)

    def encode_states(self, graph_batch) -> Tensor:
        states = self._readout(self.state_encoder, graph_batch)
        return F.normalize(self.state_projection(states), dim=-1)

    def encode_lemmas(self, graph_batch) -> Tensor:
        lemmas = self._readout(self.lemma_encoder, graph_batch)
        return F.normalize(self.lemma_projection(lemmas), dim=-1)

    def similarity(self, state_embeddings: Tensor, lemma_embeddings: Tensor) -> Tensor:
        """Return temperature-scaled all-pairs state/lemma scores."""
        if state_embeddings.ndim != 2 or lemma_embeddings.ndim != 2:
            raise ValueError("state and lemma embeddings must be rank-2")
        if state_embeddings.size(1) != lemma_embeddings.size(1):
            raise ValueError("state and lemma embedding dimensions must match")
        return state_embeddings @ lemma_embeddings.transpose(0, 1) / self.temperature


def load_retriever_checkpoint(
    checkpoint_path: str | Path,
    *,
    pointer_backbone: nn.Module,
    hidden_dim: int,
    node_vocab: dict[str, int],
    tactic_vocab: dict[str, int],
    device: torch.device,
) -> DualEncoderRetriever:
    """Load a trained query tower against the pointer-derived lemma tower."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"'{checkpoint_path}' is not a retriever checkpoint object")
    if checkpoint.get("model_type") != "dual_encoder_retriever":
        raise ValueError(f"'{checkpoint_path}' is not a dual-encoder retriever checkpoint")
    embedded_node_vocab = checkpoint.get("node_vocab")
    embedded_tactic_vocab = checkpoint.get("tactic_vocab")
    if embedded_node_vocab != node_vocab or embedded_tactic_vocab != tactic_vocab:
        raise ValueError("retriever checkpoint vocabularies do not match the tactic model")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("retriever checkpoint is missing model_state_dict")
    projection_weight = state_dict.get("state_projection.weight")
    if not torch.is_tensor(projection_weight):
        raise ValueError("retriever checkpoint is missing state_projection.weight")
    projection_dim = int(projection_weight.size(0))
    config = checkpoint.get("config", {})
    temperature = float(config.get("temperature", 0.07)) if isinstance(config, dict) else 0.07
    model = DualEncoderRetriever(
        copy.deepcopy(pointer_backbone),
        copy.deepcopy(pointer_backbone),
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        temperature=temperature,
    ).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"retriever checkpoint does not match the pointer model: {exc}") from exc
    from .lemma_index import state_dict_sha256

    if state_dict_sha256(model.lemma_encoder.state_dict()) != state_dict_sha256(
        pointer_backbone.state_dict()
    ):
        raise ValueError(
            "retriever lemma tower does not match the pointer/index encoder"
        )
    model.freeze_lemma_tower()
    model.eval()
    return model


def multi_positive_info_nce(
    logits: Tensor,
    positive_mask: Tensor,
) -> tuple[Tensor, dict[str, int | float]]:
    """Compute InfoNCE where each query may cite multiple positive lemmas.

    Every non-positive candidate in the row is a negative. Thus positives from
    other examples automatically become in-batch negatives, while callers may
    append accessible-unused and prior-index hard negatives as extra columns.
    Rows without a known positive are excluded and reported as label coverage,
    rather than silently becoming all-negative training examples.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [queries, candidates]")
    if positive_mask.shape != logits.shape:
        raise ValueError("positive_mask must have the same shape as logits")
    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    valid_rows = positive_mask.any(dim=1)
    valid_count = int(valid_rows.sum().item())
    query_count = int(logits.size(0))
    if valid_count == 0:
        # Keep the zero connected to logits so backward remains valid in a
        # batch whose citations all failed corpus resolution.
        loss = logits.sum() * 0.0
    else:
        valid_logits = logits[valid_rows]
        valid_positives = positive_mask[valid_rows]
        numerator = torch.logsumexp(
            valid_logits.masked_fill(~valid_positives, float("-inf")), dim=1
        )
        denominator = torch.logsumexp(valid_logits, dim=1)
        loss = (denominator - numerator).mean()
    return loss, {
        "query_count": query_count,
        "labeled_query_count": valid_count,
        "positive_count": int(positive_mask.sum().item()),
        "label_coverage": valid_count / max(query_count, 1),
    }


def build_positive_mask(
    candidate_ids: Sequence[int],
    positive_ids_per_query: Sequence[Iterable[int]],
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Map external lemma IDs to a multi-positive contrastive target mask."""
    candidate_positions: dict[int, list[int]] = {}
    for position, lemma_id in enumerate(candidate_ids):
        candidate_positions.setdefault(int(lemma_id), []).append(position)
    mask = torch.zeros(
        (len(positive_ids_per_query), len(candidate_ids)),
        dtype=torch.bool,
        device=device,
    )
    for row, positive_ids in enumerate(positive_ids_per_query):
        for lemma_id in set(int(value) for value in positive_ids if int(value) >= 0):
            for position in candidate_positions.get(lemma_id, ()):
                mask[row, position] = True
    return mask


def mine_hard_negative_ids(
    lemma_index,
    query_embeddings: Tensor,
    positive_ids_per_query: Sequence[Iterable[int]],
    *,
    k: int,
    oversample: int = 4,
    accessible_ids_per_query: Sequence[set[int]] | None = None,
) -> list[list[int]]:
    """Mine non-positive nearest neighbors from an earlier, frozen index.

    ``accessible_ids_per_query`` is optional but, when supplied, is enforced
    before a candidate can become a negative. Visibility is intentionally an
    input: current prepared examples do not contain Lean import information, so
    guessing accessibility here would introduce false negatives and leakage.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if oversample < 1:
        raise ValueError("oversample must be at least 1")
    query_count = int(query_embeddings.size(0))
    if len(positive_ids_per_query) != query_count:
        raise ValueError("positive IDs must have one row per query")
    if accessible_ids_per_query is not None and len(accessible_ids_per_query) != query_count:
        raise ValueError("accessible IDs must have one row per query")
    if k == 0:
        return [[] for _ in range(query_count)]

    ids_per_query, _, _ = lemma_index.search(
        query_embeddings.detach().float().cpu().numpy(),
        k=min(len(lemma_index.lemma_ids), max(k, k * oversample)),
    )
    result: list[list[int]] = []
    for row, retrieved_ids in enumerate(ids_per_query):
        positives = {int(value) for value in positive_ids_per_query[row]}
        accessible = (
            None
            if accessible_ids_per_query is None
            else accessible_ids_per_query[row]
        )
        selected: list[int] = []
        for raw_id in retrieved_ids:
            lemma_id = int(raw_id)
            if lemma_id in positives:
                continue
            if accessible is not None and lemma_id not in accessible:
                continue
            if lemma_id not in selected:
                selected.append(lemma_id)
            if len(selected) == k:
                break
        result.append(selected)
    return result


@dataclass(frozen=True)
class RetrievalMetrics:
    query_count: int
    labeled_query_count: int
    positive_count: int
    reciprocal_rank_sum: float
    hits: dict[int, int]

    def as_dict(self) -> dict[str, int | float]:
        denominator = max(self.labeled_query_count, 1)
        result: dict[str, int | float] = {
            "query_count": self.query_count,
            "labeled_query_count": self.labeled_query_count,
            "positive_count": self.positive_count,
            "label_coverage": self.labeled_query_count / max(self.query_count, 1),
            "mrr": self.reciprocal_rank_sum / denominator,
        }
        for k, count in sorted(self.hits.items()):
            result[f"recall_at_{k}"] = count / denominator
        return result


def retrieval_metrics(
    ranked_ids_per_query: Sequence[Sequence[int]],
    positive_ids_per_query: Sequence[Iterable[int]],
    *,
    ks: Sequence[int] = (1, 10, 50, 200),
) -> RetrievalMetrics:
    """Measure first-stage corpus retrieval, separate from label coverage.

    Recall@K is proof-step success: a labeled query is a hit when any cited
    external declaration appears in its top K. MRR uses the first cited result.
    Positive citation count is reported so multi-positive density remains
    visible beside proof-step metrics.
    """
    if len(ranked_ids_per_query) != len(positive_ids_per_query):
        raise ValueError("ranked and positive IDs must have the same row count")
    clean_ks = tuple(sorted(set(int(k) for k in ks)))
    if not clean_ks or clean_ks[0] <= 0:
        raise ValueError("ks must contain positive integers")

    labeled = 0
    positive_count = 0
    reciprocal_rank_sum = 0.0
    hits = {k: 0 for k in clean_ks}
    for ranked_ids, raw_positives in zip(ranked_ids_per_query, positive_ids_per_query):
        positives = {int(value) for value in raw_positives if int(value) >= 0}
        if not positives:
            continue
        labeled += 1
        positive_count += len(positives)
        first_rank: int | None = None
        for rank, raw_id in enumerate(ranked_ids, start=1):
            if int(raw_id) in positives:
                first_rank = rank
                break
        if first_rank is None:
            continue
        reciprocal_rank_sum += 1.0 / first_rank
        for k in clean_ks:
            if first_rank <= k:
                hits[k] += 1
    return RetrievalMetrics(
        query_count=len(ranked_ids_per_query),
        labeled_query_count=labeled,
        positive_count=positive_count,
        reciprocal_rank_sum=reciprocal_rank_sum,
        hits=hits,
    )
