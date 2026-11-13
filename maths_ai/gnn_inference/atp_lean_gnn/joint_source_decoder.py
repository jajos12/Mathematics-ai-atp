"""Source-aware flat argument decoder with text lemmas and local graph nodes.

Library ranking starts at the frozen text retriever's order. A learned adapter
supplies compatible GRU inputs, so library arguments never become zero vectors.
This decoder does not synthesize compound Lean tactic syntax.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .argument_selector import PublishedPointerClassifier
from .parser import parse_sexp_string
from .premise_pool import CandidateSource


@dataclass(frozen=True)
class SourceCandidates:
    local_ids: tuple[int, ...]  # Per-graph node indices, not batch-global indices.
    local_vectors: Tensor
    lemma_ids: tuple[int, ...]
    lemma_vectors: Tensor  # Frozen text index vectors, before projection.
    lemma_scores: Tensor   # Frozen retrieval cosine scores.
    local_type_signal: Tensor | None = None  # Optional first-position soft evidence.
    local_type_tactic: str | None = None  # Family used to construct the optional signal.

    def __post_init__(self):
        if (self.local_vectors.ndim != 2 or self.lemma_vectors.ndim != 2
                or self.lemma_scores.ndim != 1
                or len(self.local_ids) != self.local_vectors.size(0)
                or len(self.lemma_ids) != self.lemma_vectors.size(0)
                or len(self.lemma_ids) != self.lemma_scores.numel()
                or len(set(self.local_ids)) != len(self.local_ids)
                or len(set(self.lemma_ids)) != len(self.lemma_ids)
                or (self.local_type_signal is not None and (
                    self.local_type_signal.ndim != 1
                    or self.local_type_signal.numel() != len(self.local_ids)
                    or not bool(torch.isfinite(self.local_type_signal).all())
                ))):
            raise ValueError("candidate IDs, vectors, and text scores must align uniquely")


@dataclass(frozen=True)
class DecodedArgument:
    source: CandidateSource
    stable_id: int


def _contains_subexpression(tree, expression) -> bool:
    pending = [tree]
    while pending:
        current = pending.pop()
        if current == expression:
            return True
        if isinstance(current, list):
            pending.extend(current)
    return False


def local_type_signals(candidates: SourceCandidates, node_labels: Tensor,
                       id_to_label: dict[int, str], *, tactic_name: str,
                       goal_sexp: str, hyp_sexps: list[dict]) -> Tensor:
    """Conservative first-position type cues, without filtering identities.

    `rw`, `simp` and `subst` share an Eq/Iff/HEq side-in-goal feature.
    It is soft evidence, not tactic applicability. Later arguments may
    operate on a goal changed by earlier arguments.
    """
    values = candidates.local_vectors.new_zeros(len(candidates.local_ids))
    if tactic_name not in {"exact", "rw", "simp", "subst"}:
        return values
    hypotheses = {f"FV{int(h['context_index'])}": h for h in hyp_sexps}
    goal = parse_sexp_string(goal_sexp) if tactic_name != "exact" else None
    for position, node_id in enumerate(candidates.local_ids):
        label = id_to_label.get(int(node_labels[node_id]), "")
        if not re.fullmatch(r"FV\d+", label):
            continue
        hypothesis = hypotheses.get(label)
        if hypothesis is None or not isinstance(hypothesis.get("sexp"), str):
            continue
        if tactic_name == "exact":
            values[position] = float(hypothesis["sexp"] == goal_sexp)
            continue
        relation = parse_sexp_string(hypothesis["sexp"])
        if (not isinstance(relation, list) or len(relation) < 4 or relation[0] != ":app"
                or relation[1] not in ([":c", "Eq"], [":c", "Iff"], [":c", "HEq"])):
            continue
        explicit = [item[3] for item in relation[2:]
                    if isinstance(item, list) and len(item) == 4
                    and item[0] == ":arg" and item[1] == ":explicit"]
        if len(explicit) >= 2 and any(_contains_subexpression(goal, side) for side in explicit[-2:]):
            values[position] = 1.0
    return values


def candidate_groups_from_batch(batch, node_embeddings: Tensor, library_ids,
                                library_vectors, library_scores) -> list[SourceCandidates]:
    """Keep local node identities relative to each graph, unlike PyG offsets."""
    count = int(batch.y.numel())
    if len(library_ids) != count or len(library_vectors) != count or len(library_scores) != count:
        raise ValueError("one retrieved library pool is required per graph")
    groups = []
    for row in range(count):
        start, end = (int(batch.ptr[row]), int(batch.ptr[row + 1]))
        local_ids = batch.premise_mask[start:end].nonzero(as_tuple=False).view(-1)
        groups.append(SourceCandidates(
            tuple(int(value) for value in local_ids.tolist()),
            node_embeddings[start:end].index_select(0, local_ids),
            tuple(int(value) for value in library_ids[row]),
            torch.as_tensor(library_vectors[row], dtype=node_embeddings.dtype, device=node_embeddings.device),
            torch.as_tensor(library_scores[row], dtype=node_embeddings.dtype, device=node_embeddings.device),
        ))
    return groups


class JointSourceDecoder(nn.Module):
    """Predict action length, then GRU source and identity for every slot."""

    def __init__(self, pointer: PublishedPointerClassifier, *, text_dim: int) -> None:
        super().__init__()
        self.pointer = pointer
        hidden = pointer.hidden_dim
        if text_dim < 1:
            raise ValueError("text dimension must be positive")
        self.lemma_adapter = nn.Sequential(nn.Linear(text_dim, hidden), nn.LayerNorm(hidden), nn.Tanh())
        # Evidence includes within-source best score and pool sizes. Source
        # probabilities, rather than an uncalibrated global text-score boost,
        # determine how local and library rankings meet.
        self.source_head = nn.Linear(hidden + 4, 2)
        # Learn length even on rows whose gold library lemma is absent from
        # retrieval. A binary stop head remains an auxiliary diagnostic only.
        self.arg_count_head = nn.Linear(hidden, pointer.max_args + 1)
        self.log_inverse_text_temperature = nn.Parameter(torch.tensor(3.0))
        # A zero-initialized contextual residual preserves text order at epoch
        # zero, while allowing the GRU to choose a different second lemma.
        self.library_residual_scale = nn.Parameter(torch.zeros(()))
        # Runtime-only experiment setting. No new checkpoint tensor: V2's
        # state_dict remains compatible, and the default is unchanged.
        self.local_type_bonus_strength = 0.0
        self.local_type_bonus_strengths: dict[str, float] = {}

    def initial_state(self, state: Tensor, tactic_id: int) -> Tensor:
        tactic = self.pointer.tactic_embedding(torch.tensor(
            [tactic_id], device=state.device, dtype=torch.long,
        ))
        return self.pointer.argument_selector.initial_state(state.view(1, -1), tactic)

    def count_logits(self, state: Tensor, tactic_id: int) -> Tensor:
        return self.arg_count_head(self.initial_state(state, tactic_id)).view(-1)

    def score_step(self, hidden: Tensor, candidates: SourceCandidates, *,
                   adapted: Tensor | None = None, step: int = 0):
        if hidden.shape != (1, self.pointer.hidden_dim):
            raise ValueError("one decoder state is required")
        if candidates.local_vectors.size(1) != self.pointer.hidden_dim:
            raise ValueError("local graph vectors have the wrong dimension")
        if candidates.lemma_vectors.size(1) != self.lemma_adapter[0].in_features:
            raise ValueError("lemma vectors have the wrong text dimension")
        local = (self.pointer.argument_selector.score_candidates(
            hidden, candidates.local_vectors,
        ).view(-1) if candidates.local_ids else hidden.new_empty(0))
        library = candidates.lemma_scores * self.log_inverse_text_temperature.clamp(-4, 5).exp()
        if candidates.lemma_ids:
            adapted = self.lemma_adapter(candidates.lemma_vectors) if adapted is None else adapted
            residual = self.pointer.argument_selector.score_candidates(hidden, adapted).view(-1)
            library = library + self.library_residual_scale.tanh() * residual
        evidence = hidden.new_tensor([
            float(local.detach().max()) if local.numel() else 0.0,
            float(library.detach().max()) if library.numel() else 0.0,
            float(len(candidates.local_ids)),
            float(len(candidates.lemma_ids)),
        ]).view(1, 4)
        source = self.source_head(torch.cat([hidden, evidence], dim=1)).view(-1)
        if not candidates.local_ids:
            source = source.masked_fill(torch.tensor([True, False], device=source.device), float("-inf"))
        if not candidates.lemma_ids:
            source = source.masked_fill(torch.tensor([False, True], device=source.device), float("-inf"))
        strength = self.local_type_bonus_strengths.get(
            candidates.local_type_tactic, self.local_type_bonus_strength,
        )
        if step == 0 and candidates.local_type_signal is not None and strength:
            if not math.isfinite(strength) or strength < 0:
                raise ValueError("local type bonus must be finite and nonnegative")
            local = local + strength * candidates.local_type_signal.to(
                device=local.device, dtype=local.dtype,
            )
        return source, local, library

    def selected_vector(self, candidates: SourceCandidates, source: CandidateSource, position: int,
                        *, adapted: Tensor | None = None) -> Tensor:
        if source is CandidateSource.LOCAL:
            return candidates.local_vectors[position].view(1, -1)
        if source is CandidateSource.LIBRARY:
            return (adapted[position].view(1, -1) if adapted is not None
                    else self.lemma_adapter(candidates.lemma_vectors[position].view(1, -1)))
        raise ValueError("unknown argument source")

    @torch.no_grad()
    def decode(self, state: Tensor, tactic_id: int, candidates: SourceCandidates):
        """Greedy flat action with explicitly predicted argument count."""
        hidden = self.initial_state(state, tactic_id)
        count = int(self.arg_count_head(hidden).argmax().item())
        adapted = None
        arguments = []
        for step in range(count):
            if adapted is None and candidates.lemma_ids:
                adapted = self.lemma_adapter(candidates.lemma_vectors)
            source_logits, local, library = self.score_step(
                hidden, candidates, adapted=adapted, step=step,
            )
            if not torch.isfinite(source_logits).any():
                return tuple(arguments), False
            source = CandidateSource.LOCAL if int(source_logits.argmax()) == 0 else CandidateSource.LIBRARY
            values = local if source is CandidateSource.LOCAL else library
            position = int(values.argmax().item())
            identity = candidates.local_ids[position] if source is CandidateSource.LOCAL else candidates.lemma_ids[position]
            arguments.append(DecodedArgument(source, identity))
            hidden = self.pointer.argument_selector.gru(
                self.selected_vector(candidates, source, position, adapted=adapted), hidden,
            )
        return tuple(arguments), True
