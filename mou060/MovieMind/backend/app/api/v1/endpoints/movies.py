"""Movie catalogue endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from app.api.dependencies import DbSession
from app.core.errors import NotFoundApiError, ValidationApiError
from app.repositories import movies as movie_repo
from app.repositories.movies import SORT_FIELDS
from app.schemas.movie import GenreOut, MovieListOut, MovieOut

router = APIRouter()


@router.get("/movies", response_model=MovieListOut, summary="Paginated movie listing")
def list_movies(
    db: DbSession,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: str | None = Query(None, max_length=200),
    genre: str | None = Query(None, max_length=100),
    year: int | None = Query(None, ge=1888, le=2100),
    sort: str = Query("title"),
    direction: str = Query("asc", pattern="^(asc|desc)$"),
) -> MovieListOut:
    if sort not in SORT_FIELDS:
        raise ValidationApiError(
            f"Invalid sort field. Allowed: {', '.join(sorted(SORT_FIELDS))}"
        )
    movies, total = movie_repo.list_movies(
        db,
        limit=limit,
        offset=offset,
        q=q,
        genre=genre,
        year=year,
        sort=sort,
        direction=direction,
    )
    return MovieListOut(
        items=[MovieOut.model_validate(m) for m in movies],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/movies/genres", response_model=list[GenreOut], summary="Available genres")
def list_genres(db: DbSession) -> list[GenreOut]:
    return [GenreOut.model_validate(g) for g in movie_repo.list_genres(db)]


@router.get("/movies/popular", response_model=list[MovieOut], summary="Popular movies")
def popular_movies(
    db: DbSession,
    limit: int = Query(20, ge=1, le=100),
) -> list[MovieOut]:
    return [MovieOut.model_validate(m) for m in movie_repo.popular_movies(db, limit)]


@router.get("/movies/{movie_id}", response_model=MovieOut, summary="Movie details")
def get_movie(movie_id: uuid.UUID, db: DbSession) -> MovieOut:
    movie = movie_repo.get_movie(db, movie_id)
    if movie is None:
        raise NotFoundApiError("Movie not found")
    return MovieOut.model_validate(movie)
