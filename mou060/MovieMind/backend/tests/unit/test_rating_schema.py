"""Rating schema validation (T7.1 acceptance: score scale + half-step)."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.schemas.rating import RatingCreate

MOVIE_ID = uuid.uuid4()


def _create(score: float) -> RatingCreate:
    return RatingCreate(movie_id=MOVIE_ID, score=score)


@pytest.mark.parametrize(
    "score",
    [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
)
def test_valid_half_steps(score: float):
    assert _create(score).score == score


@pytest.mark.parametrize(
    "score",
    [0.0, 0.25, 0.7, 1.1, 2.3, 3.3, 4.6, 5.5, 6.0, -1.0],
)
def test_invalid_scores_rejected(score: float):
    with pytest.raises(ValidationError):
        _create(score)


def test_missing_movie_id_rejected():
    with pytest.raises(ValidationError):
        RatingCreate(score=4.0)  # type: ignore[call-arg]
