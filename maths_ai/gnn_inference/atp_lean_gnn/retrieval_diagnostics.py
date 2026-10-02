"""Dependency-free lexical baseline over the same declaration corpus as FAISS."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

from .lemma_corpus import LemmaRecord


_TOKEN = re.compile(r"[a-z][a-z0-9]*|[0-9]+", re.IGNORECASE)
_STRUCTURAL = frozenset({"state", "goal", "hyp", "app", "forall", "lambda", "unk"})


def lexical_tokens(text: str) -> list[str]:
    """Split qualified names and snake case, retaining type identifiers."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STRUCTURAL]


class PremiseBM25:
    """Inverted-index BM25 (k1=1.2, b=0.75); no new server dependencies."""

    def __init__(self, records: Sequence[LemmaRecord]) -> None:
        self.ids = [record.lemma_id for record in records]
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for position, record in enumerate(records):
            counts = Counter(lexical_tokens(f"{record.name} {record.statement}"))
            self.lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                self.postings[token].append((position, frequency))
        self.average_length = sum(self.lengths) / max(len(records), 1)

    def search(self, query_tokens: Iterable[str], *, k: int) -> list[int]:
        if k < 1:
            raise ValueError("k must be positive")
        scores: dict[int, float] = defaultdict(float)
        count = len(self.ids)
        for token in set(query_tokens):
            hits = self.postings.get(token, ())
            if not hits:
                continue
            idf = math.log1p((count - len(hits) + 0.5) / (len(hits) + 0.5))
            for position, frequency in hits:
                norm = 1.2 * (0.25 + 0.75 * self.lengths[position] / max(self.average_length, 1))
                scores[position] += idf * frequency * 2.2 / (frequency + norm)
        return [
            self.ids[position]
            for position in heapq.nlargest(
                k, scores, key=lambda position: (scores[position], -position)
            )
        ]
