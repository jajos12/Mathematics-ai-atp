"""Accepted text towers and masked-mean pooling, independent of graph keys."""

import copy

import torch
from torch import nn
import torch.nn.functional as F


class TextDualEncoder(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.state_encoder = pretrained_encoder
        self.lemma_encoder = copy.deepcopy(pretrained_encoder)

    @staticmethod
    def _embed(encoder, tokens):
        output = encoder(**tokens).last_hidden_state
        mask = tokens["attention_mask"].to(dtype=output.dtype).unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1)

    def encode_states(self, tokens):
        return self._embed(self.state_encoder, tokens)

    def encode_lemmas(self, tokens):
        return self._embed(self.lemma_encoder, tokens)


def tokenize_texts(tokenizer, texts, *, device, max_length):
    if max_length < 1:
        raise ValueError("max_length must be positive")
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def goal_first_text(state):
    context, separator, goal = state.partition("⊢")
    if not separator:
        return state
    if not context.strip():
        return f"Goal: {goal.strip()}"
    return f"Goal: {goal.strip()}\nContext: {context.strip()}"
