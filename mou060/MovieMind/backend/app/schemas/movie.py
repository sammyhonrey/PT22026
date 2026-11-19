"""Pydantic schemas for the movie catalogue."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, ConfigDict


class MovieOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    movielens_id: int | None
    title: str
    genres: list[str]
    overview: str | None
    release_year: int | None
    poster_url: str | None
    average_rating: float | None
    ratings_count: int | None


class MovieListOut(BaseModel):
    items: list[MovieOut]
    total: int
    limit: int
    offset: int


class GenreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
