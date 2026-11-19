"""Watchlist endpoints (add, list, remove).

Ownership is always derived from the authenticated user in the token; a user
can only see and modify their own watchlist.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query, Response

from app.api.dependencies import CurrentUser, DbSession
from app.core.errors import NotFoundApiError
from app.repositories import movies as movie_repo
from app.repositories import watchlist as watchlist_repo
from app.schemas.watchlist import WatchlistItemOut, WatchlistListOut

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


@router.post("/{movie_id}", response_model=WatchlistItemOut, summary="Add movie to watchlist")
def add_to_watchlist(
    movie_id: uuid.UUID, user: CurrentUser, db: DbSession, response: Response
) -> WatchlistItemOut:
    if movie_repo.get_movie(db, movie_id) is None:
        raise NotFoundApiError("Movie not found")
    created, item = watchlist_repo.add(db, user_id=user.id, movie_id=movie_id)
    response.status_code = 201 if created else 200
    return WatchlistItemOut.model_validate(item)


@router.get("", response_model=WatchlistListOut, summary="List my watchlist")
def list_watchlist(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> WatchlistListOut:
    items, total = watchlist_repo.list_items(db, user_id=user.id, limit=limit, offset=offset)
    return WatchlistListOut(
        items=[WatchlistItemOut.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.delete("/{movie_id}", status_code=204, summary="Remove movie from watchlist")
def remove_from_watchlist(movie_id: uuid.UUID, user: CurrentUser, db: DbSession) -> None:
    if not watchlist_repo.remove(db, user_id=user.id, movie_id=movie_id):
        raise NotFoundApiError("Watchlist item not found")
