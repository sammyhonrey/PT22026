"""Recommendation service + candidate filter tests (T8.1 acceptance).

These run with no database and no model: the filter is pure, and strategies are
stubbed with lightweight ``Movie`` objects (plain Python objects, never flushed).
"""

from __future__ import annotations

import uuid

import pytest

from app.db.models.movie import Movie
from app.services.recommendation_service import (
    CandidateFilter,
    RecommendationItem,
    RecommendationService,
    RecommendationStrategy,
    combine_score_components,
    normalize_score,
)

_A = uuid.UUID("00000000-0000-0000-0000-000000000001")
_B = uuid.UUID("00000000-0000-0000-0000-000000000002")
_C = uuid.UUID("00000000-0000-0000-0000-000000000003")


def _item(movie_id: uuid.UUID, *, score: float = 1.0) -> RecommendationItem:
    return RecommendationItem(
        movie=Movie(id=movie_id, title="t"), score=score, source="test", reason="test"
    )


def test_filter_excludes_rated_and_watchlisted():
    items = [_item(_A), _item(_B), _item(_C)]
    out = CandidateFilter().apply(items, rated_movie_ids={_A}, watchlisted_movie_ids={_C})
    assert [i.movie.id for i in out] == [_B]


def test_filter_flags_are_configurable():
    items = [_item(_A), _item(_B)]
    f = CandidateFilter(exclude_rated=False, exclude_watchlisted=False)
    out = f.apply(items, rated_movie_ids={_A}, watchlisted_movie_ids={_B})
    assert [i.movie.id for i in out] == [_A, _B]


def test_filter_deduplicates_keeping_first_and_order():
    items = [_item(_A, score=5.0), _item(_B, score=4.0), _item(_A, score=3.0)]
    out = CandidateFilter().apply(items)
    assert [i.movie.id for i in out] == [_A, _B]
    assert [i.score for i in out] == [5.0, 4.0]


def test_filter_preserves_input_order():
    items = [_item(_C), _item(_A), _item(_B)]
    assert [i.movie.id for i in CandidateFilter().apply(items)] == [_C, _A, _B]


def test_filter_empty_input():
    assert CandidateFilter().apply([]) == []


def test_service_applies_filter_and_caps_limit():
    class Stub(RecommendationStrategy):
        source = "stub"

        def recommend(self, *, limit):
            return [_item(_A, score=1), _item(_B, score=2), _item(_C, score=3)][:limit]

    out = RecommendationService().recommend(Stub(), limit=2, rated_movie_ids={_A})
    assert [i.movie.id for i in out] == [_B, _C]


def test_service_overfetches_so_filtering_still_fills_page():
    ids = [uuid.UUID(f"00000000-0000-0000-0000-{i:012d}") for i in range(10)]

    class Stub(RecommendationStrategy):
        source = "stub"

        def recommend(self, *, limit):
            return [_item(ids[i], score=float(i)) for i in range(limit)]

    # Excluding one movie makes the service request limit+1 candidates so it
    # can still return a full page of 3 after filtering.
    out = RecommendationService().recommend(Stub(), limit=3, rated_movie_ids={ids[0]})
    assert [i.movie.id for i in out] == ids[1:4]


def test_normalize_score_preserves_direction():
    assert normalize_score(4.0) == 1.0
    assert normalize_score(-4.0) == -1.0
    assert normalize_score(2.0, scale=4.0) == 0.5


def test_combine_score_components_weights_inputs():
    score = combine_score_components(1.0, 0.5, content_weight=0.7, collaborative_weight=0.3)
    assert score == pytest.approx(0.85)
