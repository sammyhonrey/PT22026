from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models.user_preference import UserPreference


def get_for_user(db: Session, *, user_id: uuid.UUID) -> UserPreference | None:
    return db.scalar(select(UserPreference).where(UserPreference.user_id == user_id))


def upsert(
    db: Session,
    *,
    user_id: uuid.UUID,
    favorite_genres: list[str],
    favorite_movie_ids: list[str],
    description: str | None,
) -> UserPreference:
    existing = get_for_user(db, user_id=user_id)
    now = utcnow()
    if existing is not None:
        existing.favorite_genres = favorite_genres
        existing.favorite_movie_ids = favorite_movie_ids
        existing.description = description
        existing.updated_at = now
        db.commit()
        return existing

    item = UserPreference(
        user_id=user_id,
        favorite_genres=favorite_genres,
        favorite_movie_ids=favorite_movie_ids,
        description=description,
        created_at=now,
        updated_at=now,
    )
    db.add(item)
    db.commit()
    return item
