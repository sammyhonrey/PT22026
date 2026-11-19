"""Movie catalogue repository.

All queries are parameterized via the SQLAlchemy expression API — no raw SQL
from user input. ``movie_id`` throughout the API is the internal UUID; the
external MovieLens id is ``movielens_id``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import asc, desc, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.db.models.movie import Genre, Movie, MovieGenre

SORT_FIELDS = {"title", "release_year", "average_rating", "ratings_count", "created_at"}


def _base_select() -> select:
    return select(Movie).options(
        selectinload(Movie.genre_links).selectinload(MovieGenre.genre)
    )


def list_movies(
    db: Session,
    *,
    limit: int,
    offset: int,
    q: str | None = None,
    genre: str | None = None,
    year: int | None = None,
    sort: str = "title",
    direction: str = "asc",
) -> tuple[list[Movie], int]:
    """Return ``(movies, total)`` for the given filters.

    ``sort``/``direction`` are validated by the endpoint against allow-lists;
    here they are mapped to column expressions.
    """
    stmt = _base_select()

    if q:
        matching_genres = (
            select(MovieGenre.movie_id)
            .join(Genre, Genre.id == MovieGenre.genre_id)
            .where(Genre.name.ilike(f"%{q}%"))
        )
        stmt = stmt.where(or_(Movie.title.ilike(f"%{q}%"), Movie.id.in_(matching_genres)))
    if year is not None:
        stmt = stmt.where(Movie.release_year == year)
    if genre:
        stmt = stmt.where(
            Movie.id.in_(
                select(MovieGenre.movie_id)
                .join(Genre, Genre.id == MovieGenre.genre_id)
                .where(func.lower(Genre.name) == genre.lower())
            )
        )

    # Total count before pagination.
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    order_col = {
        "title": Movie.title,
        "release_year": Movie.release_year,
        "average_rating": Movie.average_rating,
        "ratings_count": Movie.ratings_count,
        "created_at": Movie.created_at,
    }[sort]
    # Deterministic tie-breaker for stable pagination.
    order_expr = desc(order_col) if direction == "desc" else asc(order_col)
    stmt = stmt.order_by(order_expr, Movie.id)

    movies = list(db.scalars(stmt.limit(limit).offset(offset)).unique())
    return movies, total


def get_movie(db: Session, movie_id: uuid.UUID) -> Movie | None:
    return db.scalar(_base_select().where(Movie.id == movie_id))


def get_movie_by_movielens_id(db: Session, movielens_id: int) -> Movie | None:
    return db.scalar(_base_select().where(Movie.movielens_id == movielens_id))


def list_genres(db: Session) -> list[Genre]:
    return list(db.scalars(select(Genre).order_by(Genre.name)).all())


def popular_movies(db: Session, limit: int) -> list[Movie]:
    """Movies ranked by cached popularity signals (empty until users rate).

    Ordering is deterministic: ratings_count desc, average_rating desc, title.
    """
    stmt = (
        _base_select()
        .order_by(
            desc(Movie.ratings_count),
            desc(Movie.average_rating),
            asc(Movie.title),
            Movie.id,
        )
        .limit(limit)
    )
    return list(db.scalars(stmt).unique())


def search_suggest(db: Session, q: str, limit: int) -> list[Movie]:
    """Loose title search (used for quick suggestions)."""
    stmt = _base_select().where(
        or_(Movie.title.ilike(f"{q}%"), Movie.title.ilike(f"%{q}%"))
    )
    return list(db.scalars(stmt.limit(limit)).unique())
