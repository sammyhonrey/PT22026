"""Recommendation service: strategy interface, candidate filter, orchestrator.

The recommendation layer sits behind a strategy interface so strategies can be
added or swapped without changing HTTP routes. The orchestrator (and routes)
never contain model/tensor or raw-SQL logic — that lives inside the concrete
strategies in ``app/ml/recommenders/``.

Design:
- ``RecommendationStrategy`` — interface; concrete strategies (NCF demo,
  popularity, content-based) are constructed with whatever they need to rank.
- ``RecommendationItem`` — what a strategy returns: the resolved movie, a
  score, a stable ``source`` label, and a truthful ``reason``.
- ``CandidateFilter`` — shared exclusion (rated / watchlisted) + de-duplication.
  Pure and dependency-free so it is unit-testable without a DB or model.
- ``RecommendationService`` — wires a strategy to the filter and caps results.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import ClassVar

from app.db.models.movie import Movie

# Known ``source`` labels (plain strings so new strategies can extend freely).
SOURCE_NCF = "ncf"
SOURCE_POPULARITY = "popularity"
SOURCE_CONTENT_BASED = "content_based"
SOURCE_COLLABORATIVE = "collaborative"
SOURCE_HYBRID = "hybrid"


def normalize_score(value: float, *, scale: float | None = None) -> float:
    """Scale a raw score into the [-1, 1] range while preserving direction."""
    if scale is None:
        scale = max(abs(float(value)), 1.0)
    if scale == 0:
        return 0.0
    return float(value) / scale


def combine_score_components(
    content_score: float | None,
    collaborative_score: float | None,
    *,
    content_weight: float,
    collaborative_weight: float,
) -> float:
    """Combine normalized content and collaborative components."""
    if content_score is None and collaborative_score is None:
        return 0.0
    if content_score is None:
        return float(collaborative_score or 0.0)
    if collaborative_score is None:
        return float(content_score)
    total_weight = content_weight + collaborative_weight
    if total_weight <= 0:
        return float(content_score)
    return (
        (content_weight * float(content_score)) + (collaborative_weight * float(collaborative_score))
    ) / total_weight


@dataclass(frozen=True)
class RecommendationItem:
    """A single ranked recommendation produced by a strategy."""

    movie: Movie
    score: float
    source: str
    reason: str


class RecommendationStrategy(ABC):
    """Interface every recommendation strategy implements.

    A strategy is constructed with its dependencies (DB session, the loaded
    model, the target user, ...) and :meth:`recommend` only takes a ``limit``,
    returning raw, ranked candidates. Exclusion of already-rated / watchlisted
    movies and de-duplication are the orchestrator's responsibility.
    """

    source: ClassVar[str] = "unknown"

    @abstractmethod
    def recommend(self, *, limit: int) -> list[RecommendationItem]:
        """Return up to ``limit`` ranked candidates (unfiltered, order = rank)."""
        raise NotImplementedError


@dataclass
class CandidateFilter:
    """Shared exclusion and de-duplication for recommendations.

    Pure (no DB, no model) so it can be unit-tested in isolation. Removes
    candidates whose movie is in the rated/watchlisted sets (when enabled) and
    drops duplicates by movie id, keeping the first occurrence and preserving
    incoming order.
    """

    exclude_rated: bool = True
    exclude_watchlisted: bool = True

    def apply(
        self,
        items: Iterable[RecommendationItem],
        *,
        rated_movie_ids: Collection[uuid.UUID] = (),
        watchlisted_movie_ids: Collection[uuid.UUID] = (),
    ) -> list[RecommendationItem]:
        excluded: set[uuid.UUID] = set()
        if self.exclude_rated:
            excluded.update(rated_movie_ids)
        if self.exclude_watchlisted:
            excluded.update(watchlisted_movie_ids)

        result: list[RecommendationItem] = []
        seen: set[uuid.UUID] = set()
        for item in items:
            movie_id = item.movie.id
            if movie_id in excluded or movie_id in seen:
                continue
            seen.add(movie_id)
            result.append(item)
        return result


class RecommendationService:
    """Orchestrates a strategy and the shared candidate filter."""

    def __init__(self, candidate_filter: CandidateFilter | None = None) -> None:
        self.candidate_filter = candidate_filter or CandidateFilter()

    def recommend(
        self,
        strategy: RecommendationStrategy,
        *,
        limit: int,
        rated_movie_ids: Collection[uuid.UUID] = (),
        watchlisted_movie_ids: Collection[uuid.UUID] = (),
    ) -> list[RecommendationItem]:
        """Return up to ``limit`` filtered, ranked recommendations.

        Over-fetches by the number of excluded movies so that filtering (which
        removes candidates) can still fill the page when enough candidates
        exist.
        """
        excluded_count = len(set(rated_movie_ids) | set(watchlisted_movie_ids))
        candidates = strategy.recommend(limit=limit + excluded_count)
        filtered = self.candidate_filter.apply(
            candidates,
            rated_movie_ids=rated_movie_ids,
            watchlisted_movie_ids=watchlisted_movie_ids,
        )
        return filtered[:limit]
