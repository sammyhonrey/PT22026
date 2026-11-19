"""User repository."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.user import User


def get_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email))


def get_by_username(db: Session, username: str) -> User | None:
    return db.scalar(select(User).where(User.username == username))


def get_by_id(db: Session, user_id: uuid.UUID) -> User | None:
    return db.scalar(select(User).where(User.id == user_id))


def create(
    db: Session,
    *,
    email: str,
    password_hash: str,
    username: str | None = None,
    display_name: str | None = None,
) -> User:
    user = User(
        email=email,
        password_hash=password_hash,
        username=username,
        display_name=display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
