"""Tactic-aware cross-encoder for ranking retrieved library declarations."""

from __future__ import annotations

import random
from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


PROMPT_POLICY = "goal_first_tactic_lemma_v1"


@dataclass(frozen=True)
class RerankRow:
    row_index: int
    state_text: str
    tactic_name: str
    positive_ids: tuple[int, ...]
    candidate_position: int


def pair_prompt(state_text: str, tactic_name: str, lemma_name: str, lemma_type: str) -> tuple[str, str]:
    """Separate text fields retain goal/tactic and lemma under token truncation."""
    if not tactic_name or not state_text:
        raise ValueError("state and tactic family are required")
    # Only tactic *family* enters the input. Raw tactic text includes gold
    # `<a>lemma_name</a>` spans and would leak the answer.
    return f"Tactic family: {tactic_name}\n{state_text}", f"{lemma_name} : {lemma_type}"


def sample_training_candidates(
    ranked_ids: Sequence[int], positive_ids: Sequence[int], *,
    max_positives: int, negatives: int, rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Group positives with high-ranked uncited candidates without false labels."""
    if max_positives < 1 or negatives < 1:
        raise ValueError("positive and negative limits must be positive")
    positives = set(positive_ids)
    available = list(dict.fromkeys(lemma_id for lemma_id in ranked_ids if lemma_id in positives))
    if not available:
        raise ValueError("no cited lemma appears in retrieved candidates")
    rng.shuffle(available)
    chosen = available[:max_positives]
    chosen += list(dict.fromkeys(
        lemma_id for lemma_id in ranked_ids if lemma_id not in positives
    ))[:negatives]
    if len(chosen) <= len(available[:max_positives]):
        raise ValueError("no uncited retrieved candidates available")
    rng.shuffle(chosen)
    return chosen, [position for position, lemma_id in enumerate(chosen) if lemma_id in positives]


def multi_positive_loss(logits: torch.Tensor, positive_positions: Sequence[int]) -> torch.Tensor:
    if logits.ndim != 1 or not positive_positions:
        raise ValueError("one candidate-score vector and positive positions are required")
    # Average positive log-likelihood rather than log-sum-exp: every observed
    # citation receives a learning signal, even when one positive is easy.
    return -F.log_softmax(logits.float(), dim=0)[list(positive_positions)].mean()


class TacticLemmaReranker(nn.Module):
    """Pretrained encoder sees state, tactic family and lemma jointly."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.relevance = nn.Linear(encoder.config.hidden_size, 1)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "TacticLemmaReranker":
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(model_name))

    def forward(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].to(outputs.dtype).unsqueeze(-1)
        pooled = (outputs * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.relevance(pooled).view(-1)


def rank_metrics(rows, rankings, *, ks=(1, 5, 10, 50, 200)):
    """Compare ranking policies on the same gold-present state population."""
    if len(rows) != len(rankings):
        raise ValueError("rows and rankings must have identical populations")
    result = {"gold_present_states": len(rows),
              "known_in_pool_citations": 0}
    hits = {k: 0 for k in ks}
    for row, ranked_ids in zip(rows, rankings):
        positives = set(row.positive_ids).intersection(ranked_ids)
        if not positives:
            raise ValueError("rank metrics require at least one positive per row")
        result["known_in_pool_citations"] += len(positives)
        for k in ks:
            hits[k] += int(bool(positives.intersection(ranked_ids[:k])))
    for k in ks:
        result[f"any_positive_at_{k}"] = hits[k] / max(len(rows), 1)
    return result
