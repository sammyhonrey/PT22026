"""Tests for app.ml.recommenders.ncf_recommender (T3.3 acceptance criteria)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.ml.model_loader import load_model
from app.ml.recommenders.ncf_recommender import NCFRecommender, UnknownMovieLensIdError

ARTIFACT_DIR = Path("artifacts")

pytestmark = pytest.mark.skipif(
    not ARTIFACT_DIR.is_dir(), reason="real artifacts not present"
)


@pytest.fixture(scope="module")
def recommender() -> NCFRecommender:
    return NCFRecommender(load_model(ARTIFACT_DIR))


def test_single_prediction_is_finite_and_in_range(recommender):
    # Known user 1, known movie 1 (both present in mappings).
    score = recommender.predict(1, 1)
    assert math.isfinite(score)
    assert 0.5 <= score <= 5.0


def test_batch_matches_single_predictions(recommender):
    pairs = [(1, 1), (2, 3), (5, 10)]
    single = [recommender.predict(u, m) for u, m in pairs]
    batch = recommender.predict_batch(pairs)
    assert batch == pytest.approx(single, abs=1e-5)


def test_unknown_user_raises_not_index_error(recommender):
    with pytest.raises(UnknownMovieLensIdError) as exc:
        recommender.predict(99999999, 1)
    assert exc.value.kind == "user"


def test_unknown_movie_raises_not_index_error(recommender):
    with pytest.raises(UnknownMovieLensIdError) as exc:
        recommender.predict(1, 99999999)
    assert exc.value.kind == "movie"


def test_score_candidates_returns_sorted_input_order(recommender):
    movies = [1, 2, 3]
    scored = recommender.score_candidates(1, movies)
    assert [m for m, _ in scored] == movies
    assert all(math.isfinite(s) for _, s in scored)


def test_known_id_lookup(recommender):
    assert recommender.has_user(1)
    assert recommender.has_movie(1)
    assert not recommender.has_user(99999999)
    assert not recommender.has_movie(99999999)
    assert "1" in recommender.known_user_ids()
    assert "1" in recommender.known_movie_ids()
