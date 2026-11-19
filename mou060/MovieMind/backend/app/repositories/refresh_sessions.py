"""Persistence helpers for opaque, rotating refresh tokens."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models.refresh_session import RefreshSession
from app.db.models.user import User
from app.repositories import users as user_repo


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create(db: Session, user: User, lifetime: timedelta) -> str:
    """Create a new refresh-token family and return its opaque token once."""
    token = secrets.token_urlsafe(48)
    db.add(
        RefreshSession(
            token_hash=_token_hash(token),
            family_id=uuid.uuid4(),
            user_id=user.id,
            expires_at=utcnow() + lifetime,
        )
    )
    db.commit()
    return token


def rotate(db: Session, token: str, lifetime: timedelta) -> tuple[User, str] | None:
    """Consume a refresh token and issue its replacement in the same family."""
    now = utcnow()
    current = db.scalar(
        select(RefreshSession)
        .where(RefreshSession.token_hash == _token_hash(token))
        .with_for_update()
    )
    if current is None:
        return None

    if current.revoked_at is not None or current.expires_at <= now:
        _revoke_family(db, current.family_id, now)
        return None

    user = user_repo.get_by_id(db, current.user_id)
    if user is None or not user.is_active:
        _revoke_family(db, current.family_id, now)
        return None

    next_token = secrets.token_urlsafe(48)
    current.revoked_at = now
    db.add(
        RefreshSession(
            token_hash=_token_hash(next_token),
            family_id=current.family_id,
            user_id=user.id,
            expires_at=now + lifetime,
        )
    )
    db.commit()
    return user, next_token


def revoke(db: Session, token: str) -> None:
    """Revoke every token in the supplied token's family."""
    current = db.scalar(
        select(RefreshSession)
        .where(RefreshSession.token_hash == _token_hash(token))
        .with_for_update()
    )
    if current is not None:
        _revoke_family(db, current.family_id, utcnow())


def _revoke_family(db: Session, family_id: uuid.UUID, at: datetime) -> None:
    db.execute(
        update(RefreshSession)
        .where(RefreshSession.family_id == family_id, RefreshSession.revoked_at.is_(None))
        .values(revoked_at=at)
    )
    db.commit()
