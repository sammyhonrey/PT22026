"""Current-user profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import CurrentUser, DbSession
from app.core.errors import ConflictApiError, NotImplementedApiError
from app.repositories import users as user_repo
from app.schemas.user import UserOut, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserOut, summary="Get current profile")
def get_me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)


@router.patch("/me", response_model=UserOut, summary="Update allowed profile fields")
def update_me(payload: UserUpdate, user: CurrentUser, db: DbSession) -> UserOut:
    if payload.username is not None:
        existing = user_repo.get_by_username(db, payload.username)
        if existing is not None and existing.id != user.id:
            raise ConflictApiError("Username already taken")
        user.username = payload.username
    if payload.display_name is not None:
        user.display_name = payload.display_name

    db.commit()
    db.refresh(user)
    return UserOut.model_validate(user)


@router.delete("/me", status_code=501, summary="Delete account (deferred)")
def delete_me() -> None:
    raise NotImplementedApiError(
        "Account deletion is deferred until a retention/deletion policy is defined"
    )
