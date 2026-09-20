from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import numpy as np
import torch
from torch import Tensor


class LemmaIndexLike(Protocol):
    def search(
        self,
        state_vecs: Tensor,
        *,
        k: int,
    ) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        ...


class CandidateSource(str, Enum):
    LOCAL = "local"
    LIBRARY = "library"

    @classmethod
    def parse(cls, value: str | "CandidateSource") -> "CandidateSource":
        if value == "lemma":  # legacy serialized/test spelling
            return cls.LIBRARY
        return cls(value)


@dataclass(frozen=True)
class CandidateRef:
    """Stable source-aware identity for one argument candidate."""

    source: CandidateSource
    stable_id: int
    graph_id: int | None = None
    local_node_index: int | None = None
    expression: str | None = None
    accessible: bool = True
    retrieval_score: float | None = None
    argument_position: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int | None]:
        return self.source.value, self.stable_id, self.graph_id


@dataclass(frozen=True)
class CandidatePool:
    candidate_vectors: Tensor
    candidate_sources: list[str]
    candidate_ids: list[int]
    local_node_ids: list[int]
    lemma_ids: list[int]
    candidates: list[CandidateRef] = field(default_factory=list)
    graph_id: int = 0

    def __post_init__(self) -> None:
        if self.candidate_vectors.ndim != 2:
            raise ValueError("candidate_vectors must have shape [candidates, hidden_dim]")
        if len(self.candidate_sources) != len(self.candidate_ids):
            raise ValueError("candidate source and ID counts differ")
        if self.candidate_vectors.size(0) != len(self.candidate_ids):
            raise ValueError("candidate vector and identity counts differ")
        normalized_sources = [
            CandidateSource.parse(source).value for source in self.candidate_sources
        ]
        object.__setattr__(self, "candidate_sources", normalized_sources)
        if self.candidates:
            if len(self.candidates) != len(self.candidate_ids):
                raise ValueError("typed candidate and vector counts differ")
            candidates = self.candidates
        else:
            candidates = [
                CandidateRef(
                    source=CandidateSource.parse(source),
                    stable_id=int(candidate_id),
                    graph_id=(self.graph_id if source == CandidateSource.LOCAL.value else None),
                    local_node_index=(
                        int(candidate_id)
                        if source == CandidateSource.LOCAL.value
                        else None
                    ),
                )
                for source, candidate_id in zip(normalized_sources, self.candidate_ids)
            ]
            object.__setattr__(self, "candidates", candidates)
        for candidate in candidates:
            if candidate.source is CandidateSource.LOCAL:
                if candidate.graph_id != self.graph_id:
                    raise ValueError("local candidate belongs to a different graph")
                if candidate.local_node_index is None or candidate.local_node_index < 0:
                    raise ValueError("local candidate requires a non-negative node index")
            elif candidate.graph_id is not None or candidate.local_node_index is not None:
                raise ValueError("library candidate cannot carry a local graph/node index")


def ensure_library_targets(
    pools: list[CandidatePool],
    lemma_index,
    arg_lemma_ids: Tensor,
) -> list[CandidatePool]:
    """Inject cited declarations omitted by top-K retrieval into training pools."""
    if arg_lemma_ids.size(0) != len(pools):
        raise ValueError("lemma targets must have one row per candidate pool")
    result: list[CandidatePool] = []
    for row, pool in enumerate(pools):
        present = {
            candidate.stable_id
            for candidate in pool.candidates
            if candidate.source is CandidateSource.LIBRARY
        }
        missing = list(
            dict.fromkeys(
                int(value)
                for value in arg_lemma_ids[row].view(-1).tolist()
                if int(value) >= 0 and int(value) not in present
            )
        )
        if not missing:
            result.append(pool)
            continue
        missing_positions: list[int] = []
        for lemma_id in missing:
            position = lemma_index.id_to_position.get(lemma_id)
            if position is None:
                raise ValueError(
                    f"cited lemma ID {lemma_id} is absent from the bound lemma index"
                )
            missing_positions.append(position)
        extra_vectors = torch.from_numpy(
            lemma_index.lemma_vectors[missing_positions]
        ).to(device=pool.candidate_vectors.device, dtype=pool.candidate_vectors.dtype)
        extra_candidates = [
            CandidateRef(
                source=CandidateSource.LIBRARY,
                stable_id=lemma_id,
                metadata={"injected_positive": True},
            )
            for lemma_id in missing
        ]
        result.append(
            CandidatePool(
                candidate_vectors=torch.cat([pool.candidate_vectors, extra_vectors], dim=0),
                candidate_sources=[
                    *pool.candidate_sources,
                    *[CandidateSource.LIBRARY.value for _ in missing],
                ],
                candidate_ids=[*pool.candidate_ids, *missing],
                local_node_ids=pool.local_node_ids,
                lemma_ids=[*pool.lemma_ids, *missing],
                candidates=[*pool.candidates, *extra_candidates],
                graph_id=pool.graph_id,
            )
        )
    return result


def build_unified_pools(
    state_vecs: Tensor,
    node_embeddings: Tensor,
    premise_mask: Tensor,
    batch_index: Tensor,
    *,
    lemma_index: LemmaIndexLike | None = None,
    k: int = 500,
    retrieval_state_vecs: Tensor | None = None,
) -> list[CandidatePool]:
    """Return per-graph candidate pools combining local and library premises."""
    if state_vecs.dim() != 2:
        raise ValueError("state_vecs must be [batch, hidden_dim].")
    if node_embeddings.dim() != 2:
        raise ValueError("node_embeddings must be [total_nodes, hidden_dim].")

    device = node_embeddings.device
    batch_size = int(state_vecs.size(0))

    # Library search (optional)
    if lemma_index is not None:
        search_vecs = state_vecs if retrieval_state_vecs is None else retrieval_state_vecs
        if search_vecs.size(0) != batch_size:
            raise ValueError("retrieval_state_vecs returned a batch size mismatch.")
        lemma_ids_batch, lemma_vecs_batch, _scores = lemma_index.search(search_vecs, k=k)
        if len(lemma_ids_batch) != batch_size:
            raise ValueError("lemma_index returned a batch size mismatch.")
        lemma_vecs_batch = torch.from_numpy(lemma_vecs_batch).to(device=device, dtype=node_embeddings.dtype)
    else:
        lemma_ids_batch = [[] for _ in range(batch_size)]
        lemma_vecs_batch = None

    pools: list[CandidatePool] = []
    for b in range(batch_size):
        graph_mask = batch_index == b
        graph_node_ids = graph_mask.nonzero(as_tuple=False).view(-1)
        graph_offset = int(graph_node_ids[0].item()) if graph_node_ids.numel() > 0 else 0

        local_mask = graph_mask & premise_mask
        local_ids = local_mask.nonzero(as_tuple=False).view(-1)
        local_vecs = node_embeddings.index_select(0, local_ids)
        local_id_list = [int(i - graph_offset) for i in local_ids.tolist()]

        raw_lemma_ids = [int(x) for x in lemma_ids_batch[b]] if lemma_ids_batch else []
        valid_lemma_positions = [
            position for position, lemma_id in enumerate(raw_lemma_ids) if lemma_id >= 0
        ]
        lemma_ids = [raw_lemma_ids[position] for position in valid_lemma_positions]
        if lemma_vecs_batch is not None and valid_lemma_positions:
            lemma_vecs = lemma_vecs_batch[b][valid_lemma_positions]
        else:
            lemma_vecs = torch.empty(
                0, node_embeddings.size(1), device=device, dtype=node_embeddings.dtype
            )
        retrieval_scores = (
            [float(_scores[b, position]) for position in valid_lemma_positions]
            if lemma_index is not None
            else []
        )

        if local_vecs.numel() == 0:
            candidate_vectors = lemma_vecs
            candidate_sources = [CandidateSource.LIBRARY.value] * len(lemma_ids)
            candidate_ids = lemma_ids
        elif lemma_vecs.numel() == 0:
            candidate_vectors = local_vecs
            candidate_sources = ["local"] * len(local_id_list)
            candidate_ids = local_id_list
        else:
            candidate_vectors = torch.cat([local_vecs, lemma_vecs], dim=0)
            candidate_sources = [CandidateSource.LOCAL.value] * len(local_id_list) + [
                CandidateSource.LIBRARY.value
            ] * len(lemma_ids)
            candidate_ids = local_id_list + lemma_ids

        candidates = [
            CandidateRef(
                source=CandidateSource.LOCAL,
                stable_id=node_id,
                graph_id=b,
                local_node_index=node_id,
            )
            for node_id in local_id_list
        ] + [
            CandidateRef(
                source=CandidateSource.LIBRARY,
                stable_id=lemma_id,
                retrieval_score=score,
                metadata={"retrieval_rank": rank},
            )
            for rank, (lemma_id, score) in enumerate(
                zip(lemma_ids, retrieval_scores), start=1
            )
        ]

        pools.append(
            CandidatePool(
                candidate_vectors=candidate_vectors,
                candidate_sources=candidate_sources,
                candidate_ids=candidate_ids,
                local_node_ids=local_id_list,
                lemma_ids=lemma_ids,
                candidates=candidates,
                graph_id=b,
            )
        )

    return pools
