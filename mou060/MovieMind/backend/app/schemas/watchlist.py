"""Pydantic schemas for the watchlist."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.movie import MovieOut


class WatchlistItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    movie: MovieOut
    created_at: datetime


class WatchlistListOut(BaseModel):
    items: list[WatchlistItemOut]
    total: int
    limit: int
    offset: int
