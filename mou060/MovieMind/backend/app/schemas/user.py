"""Pydantic schemas for users."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    username: str | None
    display_name: str | None
    role: str
    is_active: bool
    created_at: datetime


class UserUpdate(BaseModel):
    """Allow-listed profile fields a user may change about themselves."""

    display_name: str | None = Field(None, max_length=100)
    username: str | None = Field(None, min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_.-]+$")
