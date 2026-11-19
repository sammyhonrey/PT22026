"""Ratings repository.

All queries are parameterized via the SQLAlchemy expression API — no raw SQL
from user input. ``movie_id`` throughout the API is the internal UUID.

The write operations are atomic: the upsert uses PostgreSQL's ``ON CONFLICT``
so re-rating the same movie updates in place instead of inserting a duplicate.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, selectinload

from app.db.base import utcnow
from app.db.models.movie import Movie, MovieGenre
from app.db.models.rating import Rating


def _rating_select() -> select:
    """Rating with its movie (and genres) eagerly loaded for serialization."""
    return select(Rating).options(
        selectinload(Rating.movie).selectinload(Movie.genre_links).selectinload(MovieGenre.genre)
    )


def get_rating(db: Session, *, user_id: uuid.UUID, movie_id: uuid.UUID) -> Rating | None:
    return db.scalar(_rating_select().where(Rating.user_id == user_id, Rating.movie_id == movie_id))


def upsert(db: Session, *, user_id: uuid.UUID, movie_id: uuid.UUID, score: float) -> Rating:
    """Insert or update the user's rating for a movie, transaction-safely.

    ``updated_at`` is refreshed on update; ``created_at`` is preserved.
    """
    now = utcnow()
    stmt = pg_insert(Rating).values(
        id=uuid.uuid4(),
        user_id=user_id,
        movie_id=movie_id,
        score=score,
        created_at=now,
        updated_at=now,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[Rating.user_id, Rating.movie_id],
        set_={"score": stmt.excluded.score, "updated_at": now},
    )
    db.execute(stmt)
    db.commit()

    rating = get_rating(db, user_id=user_id, movie_id=movie_id)
    if rating is None:  # pragma: no cover — must exist after insert/update
        raise RuntimeError("rating upsert failed to return a row")
    return rating


def list_ratings(
    db: Session, *, user_id: uuid.UUID, limit: int, offset: int
) -> tuple[list[Rating], int]:
    """Return ``(ratings, total)`` for a user, most recently updated first."""
    total = (
        db.scalar(select(func.count()).select_from(Rating).where(Rating.user_id == user_id)) or 0
    )
    stmt = (
        _rating_select()
        .where(Rating.user_id == user_id)
        .order_by(Rating.updated_at.desc(), Rating.id)
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).unique()), total


def list_movie_ids(db: Session, *, user_id: uuid.UUID) -> list[uuid.UUID]:
    """Return the movie ids the user has rated (for recommendation exclusion)."""
    return list(db.scalars(select(Rating.movie_id).where(Rating.user_id == user_id)))


def delete(db: Session, *, user_id: uuid.UUID, movie_id: uuid.UUID) -> bool:
    """Delete the user's rating for a movie; return False if none exists."""
    rating = db.scalar(select(Rating).where(Rating.user_id == user_id, Rating.movie_id == movie_id))
    if rating is None:
        return False
    db.delete(rating)
    db.commit()
    return True
