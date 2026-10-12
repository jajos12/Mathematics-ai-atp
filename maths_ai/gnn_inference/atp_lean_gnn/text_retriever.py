"""Trainable proof-state/type-signature dual encoder for premise experiments."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TextDualEncoder(nn.Module):
    """Independent pretrained towers with masked-mean pooling and cosine keys."""

    def __init__(self, pretrained_encoder: nn.Module) -> None:
        super().__init__()
        self.state_encoder = pretrained_encoder
        self.lemma_encoder = copy.deepcopy(pretrained_encoder)

    @classmethod
    def from_pretrained(cls, model_name: str) -> "TextDualEncoder":
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(model_name))

    @staticmethod
    def _embed(encoder: nn.Module, tokens: dict[str, Tensor]) -> Tensor:
        output = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].to(dtype=output.dtype).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1)

    def encode_states(self, tokens: dict[str, Tensor]) -> Tensor:
        return self._embed(self.state_encoder, tokens)

    def encode_lemmas(self, tokens: dict[str, Tensor]) -> Tensor:
        return self._embed(self.lemma_encoder, tokens)


def tokenize_texts(tokenizer, texts: list[str], *, device: torch.device, max_length: int):
    if max_length < 1:
        raise ValueError("max_length must be positive")
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def goal_first_text(state: str) -> str:
    """Keep goal visible when long hypothesis contexts exceed token budget."""
    context, separator, goal = state.partition("⊢")
    if not separator:
        return state
    if not context.strip():
        return f"Goal: {goal.strip()}"
    return f"Goal: {goal.strip()}\nContext: {context.strip()}"


def matched_text_rows(
    prepared: Iterable,
    upstream: Iterable,
    *,
    name_to_id: dict[str, int],
) -> list[tuple[str, list[int]]]:
    """Align original state text with exactly the prepared trace-citation labels."""
    from .premise_retriever_training import extract_external_positive_ids

    prepared_rows = list(prepared)
    examples = {int(data.row_index): data for data in prepared_rows}
    if len(examples) != len(prepared_rows):
        raise ValueError("prepared source has duplicate row indices")
    result: dict[int, tuple[str, list[int]]] = {}
    for row in upstream:
        data = examples.get(row.row_index)
        if data is None:
            continue
        if (
            row.theorem != str(data.theorem)
            or row.tactic != str(data.tactic_raw)
            or row.dataset_name != str(data.dataset_name)
            or row.split != str(data.split)
        ):
            raise ValueError(f"upstream source differs from prepared row {row.row_index}")
        if not row.state.strip():
            raise ValueError(f"upstream proof state missing at row {row.row_index}")
        example = SimpleNamespace(
            y=data.y.view(-1),
            arg_count=torch.tensor([int(data.arg_count)]),
            arg_lemma_ids=data.arg_lemma_ids,
            arg_node_indices=data.arg_node_indices,
            tactic_raw=data.tactic_raw,
        )
        if hasattr(data, "arg_label_policy"):
            example.arg_label_policy = data.arg_label_policy
        if hasattr(data, "citation_lemma_ids"):
            example.citation_lemma_ids = data.citation_lemma_ids
            example.citation_count = torch.tensor([int(data.citation_count)])
        positives = extract_external_positive_ids(example, name_to_id=name_to_id)[0]
        result[row.row_index] = goal_first_text(row.state), positives
        if len(result) == len(examples):
            break
    if len(result) != len(examples):
        raise ValueError(f"upstream contains {len(result)}/{len(examples)} prepared rows")
    return [result[row_index] for row_index in sorted(examples)]
