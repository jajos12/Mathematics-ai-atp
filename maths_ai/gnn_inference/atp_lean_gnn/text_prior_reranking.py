"""Experimental text relevance prior for a frozen mixed-pool graph pointer.

Only library logits receive an adjustment.  Normalize text similarities within
each retrieved library pool; otherwise text cosine and pointer dot scores have
unrelated scales.  Local pointer logits remain numerically unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .premise_pool import CandidateRef, CandidateSource


def blend_library_scores(
    pointer_scores: np.ndarray,
    candidates: Sequence[CandidateRef],
    *,
    alpha: float,
) -> np.ndarray:
    """Add ``alpha * zscore(text_cosine)`` to library logits only."""
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("text prior strength must be finite and nonnegative")
    scores = np.asarray(pointer_scores, dtype=np.float32)
    if scores.ndim != 1 or len(scores) != len(candidates) or not np.isfinite(scores).all():
        raise ValueError("pointer scores must be finite and match the candidate pool")
    selected = scores.copy()
    library = [position for position, candidate in enumerate(candidates)
               if candidate.source is CandidateSource.LIBRARY]
    if not library:
        return selected
    text_scores = np.asarray(
        [candidates[position].retrieval_score for position in library], dtype=np.float32
    )
    if not np.isfinite(text_scores).all():
        raise ValueError("every library candidate needs a finite text retrieval score")
    if alpha:
        centered = text_scores - text_scores.mean()
        std = float(text_scores.std())
        if std > 1e-8:
            selected[library] += alpha * (centered / std)
    return selected


def rank_of(scores: np.ndarray, position: int, *, eligible: Sequence[int] | None = None) -> int:
    """One-indexed rank with deterministic lower-position tie breaking."""
    positions = list(range(len(scores))) if eligible is None else list(eligible)
    if position not in positions:
        raise ValueError("gold position must be eligible")
    order = sorted(positions, key=lambda index: (-float(scores[index]), index))
    return order.index(position) + 1
