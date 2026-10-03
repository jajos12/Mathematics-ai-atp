"""Dependency-free lexical baseline over the same declaration corpus as FAISS."""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from .lemma_corpus import LemmaRecord


_TOKEN = re.compile(r"[a-z][a-z0-9]*|[0-9]+", re.IGNORECASE)
_STRUCTURAL = frozenset({"state", "goal", "hyp", "app", "forall", "lambda", "unk"})


def lexical_tokens(text: str) -> list[str]:
    """Split qualified names and snake case, retaining type identifiers."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STRUCTURAL]


def load_cached_state_text(prepared_root: str | Path, data, *, split: str = "val") -> str:
    """Recover real pre-tactic state only from matching extractor cache row."""
    row_index = int(data.row_index)
    path = Path(prepared_root) / split / "sexpr" / f"{row_index:09d}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Source text cache '{path}' is missing; use --text-source upstream "
            "to stream the original benchmark instead"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema_version") != 4:
        raise ValueError(f"Source text cache '{path}' has an invalid schema")
    expected = {
        "dataset": str(data.dataset_name),
        "split": split,
        "row_index": row_index,
        "theorem": str(data.theorem),
        "tactic": str(data.tactic_raw),
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Source text cache '{path}' differs at {key}")
    state = record.get("text_state")
    if not isinstance(state, str) or not state.strip():
        raise ValueError(f"Source text cache '{path}' has no pre-tactic state")
    return state


class PremiseBM25:
    """Inverted-index BM25 (k1=1.2, b=0.75); no new server dependencies."""

    def __init__(self, records: Sequence[LemmaRecord]) -> None:
        self.ids = [record.lemma_id for record in records]
        self.id_to_position = {
            lemma_id: position for position, lemma_id in enumerate(self.ids)
        }
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, record in enumerate(records):
            counts = Counter(lexical_tokens(f"{record.name} {record.statement}"))
            self.lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self.postings[token].append((position, frequency))
        self.average_length = sum(self.lengths) / max(len(records), 1)

    def _scores(self, query_tokens: Iterable[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        count = len(self.ids)
        # Stable token order prevents rounding differences for close BM25 scores
        # when Python starts with a different hash seed.
        for token in sorted(set(query_tokens)):
            hits = self.postings.get(token, ())
            if not hits:
                continue
            idf = math.log1p((count - len(hits) + 0.5) / (len(hits) + 0.5))
            for position, frequency in hits:
                norm = 1.2 * (0.25 + 0.75 * self.lengths[position] / max(self.average_length, 1))
                scores[position] += idf * frequency * 2.2 / (frequency + norm)
        return scores

    def search(self, query_tokens: Iterable[str], *, k: int) -> list[int]:
        if k < 1:
            raise ValueError("k must be positive")
        scores = self._scores(query_tokens)
        return [
            self.ids[position]
            for position in heapq.nlargest(
                k, scores, key=lambda position: (scores[position], -position)
            )
        ]

    def search_with_gold_rank(
        self, query_tokens: Iterable[str], positive_ids: Iterable[int], *, k: int
    ) -> tuple[list[int], int | None]:
        """Top-K plus exact first gold rank without materializing top 5,000.

        Only documents sharing at least one token are scored by BM25. A gold
        document without lexical overlap has no rank in this retrieval policy.
        """
        if k < 1:
            raise ValueError("k must be positive")
        scores = self._scores(query_tokens)
        top_ids = [
            self.ids[position]
            for position in heapq.nlargest(
                k, scores, key=lambda position: (scores[position], -position)
            )
        ]
        positive_set = set(positive_ids)
        for rank, lemma_id in enumerate(top_ids, start=1):
            if lemma_id in positive_set:
                return top_ids, rank
        gold_positions = (
            self.id_to_position[lemma_id]
            for lemma_id in positive_set
            if lemma_id in self.id_to_position
        )
        best_gold = max(
            ((scores[position], -position) for position in gold_positions if position in scores),
            default=None,
        )
        if best_gold is None:
            return top_ids, None
        rank = 1 + sum(
            (score, -position) > best_gold for position, score in scores.items()
        )
        return top_ids, rank


def hybrid_top_k(
    bm25_ids: Sequence[int],
    gnn_ids: Sequence[int],
    *,
    k: int,
    bm25_quota: int,
) -> list[int]:
    """Fill fixed candidate budget with BM25 first, then distinct GNN hits."""
    if k < 1 or not 0 <= bm25_quota <= k:
        raise ValueError("k must be positive and bm25_quota must lie in [0, k]")
    combined = list(dict.fromkeys(bm25_ids[:bm25_quota]))
    seen = set(combined)
    for source in (gnn_ids, bm25_ids[bm25_quota:]):
        for lemma_id in source:
            if lemma_id not in seen:
                combined.append(lemma_id)
                seen.add(lemma_id)
            if len(combined) == k:
                return combined
    return combined


def reciprocal_rank_fusion(
    bm25_ids: Sequence[int], gnn_ids: Sequence[int], *, k: int, constant: int = 60
) -> list[int]:
    """Combine both rank orders without using validation labels or fitted weights."""
    if k < 1 or constant < 1:
        raise ValueError("k and constant must be positive")
    scores: dict[int, float] = defaultdict(float)
    for ranking in (bm25_ids, gnn_ids):
        for rank, lemma_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[lemma_id] += 1 / (constant + rank)
    return heapq.nlargest(k, scores, key=lambda lemma_id: scores[lemma_id])


def hit_overlap(
    bm25_rows: Sequence[Sequence[int]],
    gnn_rows: Sequence[Sequence[int]],
    positive_rows: Sequence[Sequence[int]],
) -> dict[str, int]:
    """Partition labeled queries by retrieval hit in either top-200 pool."""
    if not len(bm25_rows) == len(gnn_rows) == len(positive_rows):
        raise ValueError("rankings and positives must have equal row counts")
    counts = {"both": 0, "bm25_only": 0, "gnn_only": 0, "neither": 0}
    for bm25, gnn, positives in zip(bm25_rows, gnn_rows, positive_rows):
        if not positives:
            continue
        gold = set(positives)
        bm25_hit = bool(gold.intersection(bm25))
        gnn_hit = bool(gold.intersection(gnn))
        if bm25_hit and gnn_hit:
            counts["both"] += 1
        elif bm25_hit:
            counts["bm25_only"] += 1
        elif gnn_hit:
            counts["gnn_only"] += 1
        else:
            counts["neither"] += 1
    return counts


def text_graph_hit_overlap(
    text_rows: Sequence[Sequence[int]],
    graph_rows: Sequence[Sequence[int]],
    positive_rows: Sequence[Sequence[int]],
) -> dict[str, int]:
    """Name hit categories for text-vs-graph queries, not BM25-vs-GNN."""
    counts = hit_overlap(text_rows, graph_rows, positive_rows)
    return {
        "both": counts["both"],
        "text_only": counts["bm25_only"],
        "graph_only": counts["gnn_only"],
        "neither": counts["neither"],
    }
