"""Watchlist repository.

All queries are parameterized via the SQLAlchemy expression API. ``movie_id``
throughout the API is the internal UUID. Duplicate adds are idempotent (the
DB-level unique constraint is the backstop; the repository returns the existing
row instead of raising).
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models.movie import Movie, MovieGenre
from app.db.models.watchlist import WatchlistItem


def _watchlist_select() -> select:
    """Watchlist item with its movie (and genres) eagerly loaded."""
    return select(WatchlistItem).options(
        selectinload(WatchlistItem.movie)
        .selectinload(Movie.genre_links)
        .selectinload(MovieGenre.genre)
    )


def get_item(db: Session, *, user_id: uuid.UUID, movie_id: uuid.UUID) -> WatchlistItem | None:
    return db.scalar(
        _watchlist_select().where(
            WatchlistItem.user_id == user_id, WatchlistItem.movie_id == movie_id
        )
    )


def add(db: Session, *, user_id: uuid.UUID, movie_id: uuid.UUID) -> tuple[bool, WatchlistItem]:
    """Add a movie to the user's watchlist (idempotent).

    Returns ``(created, item)`` so the endpoint can report 201 vs 200.
    """
    existing = get_item(db, user_id=user_id, movie_id=movie_id)
    if existing is not None:
        return False, existing

    item = WatchlistItem(user_id=user_id, movie_id=movie_id)
    db.add(item)
    db.commit()

    loaded = get_item(db, user_id=user_id, movie_id=movie_id)
    if loaded is None:  # pragma: no cover — must exist after insert
        raise RuntimeError("watchlist insert failed to return a row")
    return True, loaded


def list_items(
    db: Session, *, user_id: uuid.UUID, limit: int, offset: int
) -> tuple[list[WatchlistItem], int]:
    """Return ``(items, total)`` for a user, most recently added first."""
    total = (
        db.scalar(
            select(func.count()).select_from(WatchlistItem).where(WatchlistItem.user_id == user_id)
        )
        or 0
    )
    stmt = (
        _watchlist_select()
        .where(WatchlistItem.user_id == user_id)
        .order_by(WatchlistItem.created_at.desc(), WatchlistItem.id)
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt).unique()), total


def list_movie_ids(db: Session, *, user_id: uuid.UUID) -> list[uuid.UUID]:
    """Return the movie ids on the user's watchlist (for recommendation exclusion)."""
    return list(db.scalars(select(WatchlistItem.movie_id).where(WatchlistItem.user_id == user_id)))


def remove(db: Session, *, user_id: uuid.UUID, movie_id: uuid.UUID) -> bool:
    """Remove a movie from the user's watchlist; return False if absent."""
    item = db.scalar(
        select(WatchlistItem).where(
            WatchlistItem.user_id == user_id, WatchlistItem.movie_id == movie_id
        )
    )
    if item is None:
        return False
    db.delete(item)
    db.commit()
    return True
