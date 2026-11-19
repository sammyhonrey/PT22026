"""Pydantic schemas for ratings.

The application rating scale is ``0.5–5.0`` in ``0.5`` increments to match the
NCF model's output range (owner decision #5 — recommended default).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.movie import MovieOut


class RatingCreate(BaseModel):
    movie_id: uuid.UUID
    score: float = Field(ge=0.5, le=5.0)

    @field_validator("score")
    @classmethod
    def _half_step(cls, value: float) -> float:
        """Reject anything that is not a multiple of 0.5 (e.g. 0.7, 3.3)."""
        steps = Decimal(str(value)) * 10
        if steps % 5 != 0:
            raise ValueError("score must be in 0.5 increments (e.g. 0.5, 1.0, 4.5)")
        return value


class RatingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    movie_id: uuid.UUID
    score: float
    movie: MovieOut
    created_at: datetime
    updated_at: datetime


class RatingListOut(BaseModel):
    items: list[RatingOut]
    total: int
    limit: int
    offset: int
