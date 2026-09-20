"""Training utilities for first-stage external-premise retrieval."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import Optimizer
from torch_geometric.data import Batch

from .graph import lemma_statement_to_dag
from .lemma_corpus import LemmaRecord, load_lemma_corpus
from .premise_retrieval import (
    DualEncoderRetriever,
    build_positive_mask,
    mine_hard_negative_ids,
    multi_positive_info_nce,
    retrieval_metrics,
)
from .pyg import dag_to_pyg
from .training import transform_edge_index


class LemmaGraphStore:
    """Lazily materialize corpus declarations as PyG graphs by lemma ID."""

    def __init__(
        self,
        records: Iterable[LemmaRecord],
        *,
        node_vocab: dict[str, int],
        edge_mode: str = "bidirectional",
        cache_size: int = 4096,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative")
        self.records = {record.lemma_id: record for record in records}
        if not self.records:
            raise ValueError("lemma corpus is empty")
        self.lemma_ids = tuple(self.records)
        self.node_vocab = node_vocab
        self.edge_mode = edge_mode
        self.cache_size = cache_size
        self._cache: OrderedDict[int, object] = OrderedDict()

    @classmethod
    def from_corpus(
        cls,
        corpus_path: str | Path,
        *,
        node_vocab: dict[str, int],
        edge_mode: str = "bidirectional",
        cache_size: int = 4096,
    ) -> "LemmaGraphStore":
        return cls(
            load_lemma_corpus(corpus_path),
            node_vocab=node_vocab,
            edge_mode=edge_mode,
            cache_size=cache_size,
        )

    def _build_graph(self, lemma_id: int):
        try:
            record = self.records[lemma_id]
        except KeyError as exc:
            raise KeyError(f"lemma ID {lemma_id} is absent from the corpus") from exc
        try:
            dag = lemma_statement_to_dag(record.statement)
            data = dag_to_pyg(dag, self.node_vocab)
            state_nodes = [node.id for node in dag.nodes if node.label == "State"]
            if len(state_nodes) != 1:
                raise ValueError(f"expected one State node, found {len(state_nodes)}")
            data.state_node_index = torch.tensor(state_nodes, dtype=torch.long)
            data.edge_index = transform_edge_index(
                data.edge_index, edge_mode=self.edge_mode
            )
            return data
        except Exception as exc:
            # A cited positive must never disappear from a batch because its
            # declaration graph failed. Make the target and reason explicit.
            raise ValueError(
                f"failed to encode lemma {lemma_id} ({record.name}): {exc}"
            ) from exc

    def graph(self, lemma_id: int):
        lemma_id = int(lemma_id)
        cached = self._cache.pop(lemma_id, None)
        if cached is not None:
            self._cache[lemma_id] = cached
            return cached
        data = self._build_graph(lemma_id)
        if self.cache_size:
            self._cache[lemma_id] = data
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return data

    def batch(self, lemma_ids: Sequence[int], *, device: torch.device) -> Batch:
        if not lemma_ids:
            raise ValueError("cannot create an empty lemma batch")
        return Batch.from_data_list([self.graph(int(value)) for value in lemma_ids]).to(
            device
        )


def extract_external_positive_ids(batch) -> list[list[int]]:
    """Recover every trace-cited external lemma ID for each proof step."""
    batch_size = int(batch.y.numel()) if hasattr(batch, "y") else int(batch.num_graphs)
    if not hasattr(batch, "arg_count") or not hasattr(batch, "arg_lemma_ids"):
        return [[] for _ in range(batch_size)]
    counts = [int(value) for value in batch.arg_count.view(-1).tolist()]
    flat_ids = [int(value) for value in batch.arg_lemma_ids.view(-1).tolist()]
    if len(counts) != batch_size:
        raise ValueError("arg_count must contain one value per proof state")
    if sum(counts) != len(flat_ids):
        raise ValueError("arg_count does not match flattened arg_lemma_ids")
    result: list[list[int]] = []
    offset = 0
    for count in counts:
        row = flat_ids[offset : offset + count]
        result.append(list(dict.fromkeys(value for value in row if value >= 0)))
        offset += count
    return result


def sample_accessible_unused_ids(
    accessible_ids_per_query: Sequence[Sequence[int]],
    positive_ids_per_query: Sequence[Iterable[int]],
    *,
    count: int,
    rng: random.Random,
) -> list[list[int]]:
    """Sample random visible declarations, excluding each query's citations."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if len(accessible_ids_per_query) != len(positive_ids_per_query):
        raise ValueError("accessible and positive IDs must have the same row count")
    rows: list[list[int]] = []
    for accessible_ids, raw_positives in zip(
        accessible_ids_per_query, positive_ids_per_query
    ):
        positives = {int(value) for value in raw_positives}
        target_count = min(count, max(len(accessible_ids) - len(positives), 0))
        selected: dict[int, None] = {}
        # The production corpus has 319k unique IDs. Building a filtered copy
        # once per proof state dominates GNN training, so sample by index and
        # reject citations. The bounded fallback handles tiny or duplicate-heavy
        # custom visibility lists without risking an endless loop.
        max_attempts = max(target_count * 8, 32)
        attempts = 0
        while len(selected) < target_count and attempts < max_attempts:
            attempts += 1
            value = int(accessible_ids[rng.randrange(len(accessible_ids))])
            if value not in positives:
                selected.setdefault(value, None)
        if len(selected) < target_count:
            for raw_value in accessible_ids:
                value = int(raw_value)
                if value not in positives:
                    selected.setdefault(value, None)
                if len(selected) == target_count:
                    break
        rows.append(list(selected))
    return rows


@dataclass(frozen=True)
class RetrievalCandidates:
    lemma_ids: list[int]
    in_batch_negative_count: int
    accessible_negative_count: int
    hard_negative_count: int


def combine_retrieval_candidates(
    positive_ids_per_query: Sequence[Sequence[int]],
    accessible_negative_ids: Sequence[Sequence[int]],
    hard_negative_ids: Sequence[Sequence[int]],
) -> RetrievalCandidates:
    """Build one deduplicated lemma batch and count proposed negative sources."""
    row_count = len(positive_ids_per_query)
    if len(accessible_negative_ids) != row_count or len(hard_negative_ids) != row_count:
        raise ValueError("candidate sources must have one row per query")
    all_positive_ids = set().union(*(set(row) for row in positive_ids_per_query))
    in_batch_count = sum(
        len(all_positive_ids - set(row)) for row in positive_ids_per_query
    )
    ordered: dict[int, None] = {}
    for rows in (
        positive_ids_per_query,
        accessible_negative_ids,
        hard_negative_ids,
    ):
        for row in rows:
            for lemma_id in row:
                ordered.setdefault(int(lemma_id), None)
    return RetrievalCandidates(
        lemma_ids=list(ordered),
        in_batch_negative_count=in_batch_count,
        accessible_negative_count=sum(len(row) for row in accessible_negative_ids),
        hard_negative_count=sum(len(row) for row in hard_negative_ids),
    )


def train_retriever_epoch(
    model: DualEncoderRetriever,
    loader,
    lemma_store: LemmaGraphStore,
    frozen_index,
    *,
    optimizer: Optimizer,
    device: torch.device,
    accessible_negative_count: int,
    hard_negative_count: int,
    grad_clip: float,
    rng: random.Random,
) -> dict[str, int | float]:
    """Train state queries against encoded positives and three negative sources."""
    if not frozen_index.normalize_queries:
        raise ValueError(
            "retriever training requires a normalized frozen index; rebuild it "
            "with build_lemma_index.py --normalize"
        )
    model.train()
    loss_sum = 0.0
    query_count = 0
    labeled_query_count = 0
    positive_count = 0
    in_batch_count = 0
    accessible_count = 0
    hard_count = 0
    # The served index is the accessible declaration universe for this run.
    # Sampling from the raw corpus could select declarations whose graph failed
    # during index construction and then abort a batch for an unusable negative.
    all_accessible = tuple(int(value) for value in frozen_index.lemma_ids)

    for batch in loader:
        batch = batch.to(device)
        positive_rows = extract_external_positive_ids(batch)
        state_embeddings = model.encode_states(batch)
        hard_rows = mine_hard_negative_ids(
            frozen_index,
            state_embeddings,
            positive_rows,
            k=hard_negative_count,
        )
        accessible_rows = sample_accessible_unused_ids(
            [all_accessible] * len(positive_rows),
            positive_rows,
            count=accessible_negative_count,
            rng=rng,
        )
        candidates = combine_retrieval_candidates(
            positive_rows, accessible_rows, hard_rows
        )
        if not candidates.lemma_ids:
            query_count += len(positive_rows)
            continue

        lemma_batch = lemma_store.batch(candidates.lemma_ids, device=device)
        lemma_embeddings = model.encode_lemmas(lemma_batch)
        logits = model.similarity(state_embeddings, lemma_embeddings)
        positive_mask = build_positive_mask(
            candidates.lemma_ids, positive_rows, device=device
        )
        loss, metrics = multi_positive_info_nce(logits, positive_mask)

        if metrics["labeled_query_count"]:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                grad_clip,
            )
            optimizer.step()

        labeled = int(metrics["labeled_query_count"])
        loss_sum += float(loss.detach().item()) * labeled
        query_count += int(metrics["query_count"])
        labeled_query_count += labeled
        positive_count += int(metrics["positive_count"])
        in_batch_count += candidates.in_batch_negative_count
        accessible_count += candidates.accessible_negative_count
        hard_count += candidates.hard_negative_count

    return {
        "loss": loss_sum / max(labeled_query_count, 1),
        "query_count": query_count,
        "labeled_query_count": labeled_query_count,
        "positive_count": positive_count,
        "label_coverage": labeled_query_count / max(query_count, 1),
        "in_batch_negative_count": in_batch_count,
        "accessible_negative_count": accessible_count,
        "hard_negative_count": hard_count,
    }


@torch.no_grad()
def evaluate_retriever(
    model: DualEncoderRetriever,
    loader,
    frozen_index,
    *,
    device: torch.device,
    ks: Sequence[int] = (1, 10, 50, 200),
) -> dict[str, int | float]:
    """Evaluate corpus retrieval directly, not downstream reranker accuracy."""
    if not frozen_index.normalize_queries:
        raise ValueError("retriever evaluation requires a normalized index")
    model.eval()
    ranked_rows: list[list[int]] = []
    positive_rows: list[list[int]] = []
    max_k = max(ks)
    for batch in loader:
        batch = batch.to(device)
        queries = model.encode_states(batch)
        ranked_ids, _, _ = frozen_index.search(queries, k=max_k)
        ranked_rows.extend(ranked_ids)
        positive_rows.extend(extract_external_positive_ids(batch))
    return retrieval_metrics(ranked_rows, positive_rows, ks=ks).as_dict()
