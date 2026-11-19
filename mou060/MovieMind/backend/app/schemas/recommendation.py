"""Pydantic schemas for recommendations and demo predictions."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.movie import MovieOut


class RecommendationOut(BaseModel):
    """A single ranked recommendation item."""

    movie: MovieOut
    score: float
    source: str
    reason: str


class RecommendationsOut(BaseModel):
    """A page of recommendations plus strategy metadata."""

    items: list[RecommendationOut]
    source: str
    is_fallback: bool
    limit: int


class PredictionOut(BaseModel):
    """A single NCF demo prediction for one (raw user, raw movie) pair."""

    user_id: str
    movie_id: str
    score: float
