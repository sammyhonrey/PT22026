"""Rating endpoints (create/update, list, get, delete).

Ownership is always derived from the authenticated user in the token; a user
can only read or delete their own ratings.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, DbSession
from app.core.errors import NotFoundApiError
from app.repositories import movies as movie_repo
from app.repositories import ratings as rating_repo
from app.schemas.rating import RatingCreate, RatingListOut, RatingOut

router = APIRouter(prefix="/ratings", tags=["ratings"])


def _ensure_movie_exists(db: DbSession, movie_id: uuid.UUID) -> None:
    if movie_repo.get_movie(db, movie_id) is None:
        raise NotFoundApiError("Movie not found")


@router.post("", response_model=RatingOut, status_code=201, summary="Create or update a rating")
def upsert_rating(payload: RatingCreate, user: CurrentUser, db: DbSession) -> RatingOut:
    _ensure_movie_exists(db, payload.movie_id)
    rating = rating_repo.upsert(db, user_id=user.id, movie_id=payload.movie_id, score=payload.score)
    return RatingOut.model_validate(rating)


@router.get("/me", response_model=RatingListOut, summary="List my ratings")
def list_my_ratings(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> RatingListOut:
    ratings, total = rating_repo.list_ratings(db, user_id=user.id, limit=limit, offset=offset)
    return RatingListOut(
        items=[RatingOut.model_validate(r) for r in ratings],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/me/{movie_id}", response_model=RatingOut, summary="Get my rating for a movie")
def get_my_rating(movie_id: uuid.UUID, user: CurrentUser, db: DbSession) -> RatingOut:
    rating = rating_repo.get_rating(db, user_id=user.id, movie_id=movie_id)
    if rating is None:
        raise NotFoundApiError("Rating not found")
    return RatingOut.model_validate(rating)


@router.delete("/me/{movie_id}", status_code=204, summary="Delete my rating")
def delete_my_rating(movie_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    if not rating_repo.delete(db, user_id=user.id, movie_id=movie_id):
        raise NotFoundApiError("Rating not found")
