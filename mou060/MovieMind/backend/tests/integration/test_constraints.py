"""Database constraint & integrity audit (T4.4 acceptance criteria).

Verifies the schema invariants the API relies on — uniqueness, FK cascade
delete, the role check constraint, and UTC timestamp autopopulation — directly
against PostgreSQL. All assertions run inside a rolled-back transaction.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db.models.movie import Movie
from app.db.models.rating import Rating
from app.db.models.user import User
from app.db.models.watchlist import WatchlistItem


def _email(prefix: str = "test") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}@example.com"


def _user(email: str | None = None) -> User:
    return User(email=email or _email(), password_hash="x")


def _movie(title: str = "Test Movie", movielens_id: int | None = None) -> Movie:
    return Movie(title=title, movielens_id=movielens_id or uuid.uuid4().int % 100000000)


def test_duplicate_email_rejected(db):
    email = _email("dup")
    db.add(_user(email))
    db.flush()
    db.add(_user(email))
    with pytest.raises(IntegrityError):
        db.flush()


def test_duplicate_rating_rejected(db):
    user = _user()
    movie = _movie()
    db.add_all([user, movie])
    db.flush()
    db.add(Rating(user_id=user.id, movie_id=movie.id, score=4.0))
    db.flush()
    db.add(Rating(user_id=user.id, movie_id=movie.id, score=5.0))
    with pytest.raises(IntegrityError):
        db.flush()


def test_duplicate_watchlist_rejected(db):
    user = _user()
    movie = _movie()
    db.add_all([user, movie])
    db.flush()
    db.add(WatchlistItem(user_id=user.id, movie_id=movie.id))
    db.flush()
    db.add(WatchlistItem(user_id=user.id, movie_id=movie.id))
    with pytest.raises(IntegrityError):
        db.flush()


def test_user_delete_cascades(db):
    user = _user()
    movie = _movie()
    db.add_all([user, movie])
    db.flush()
    db.add(Rating(user_id=user.id, movie_id=movie.id, score=3.5))
    db.add(WatchlistItem(user_id=user.id, movie_id=movie.id))
    db.flush()

    assert db.scalar(select(func.count()).select_from(Rating)) == 1
    assert db.scalar(select(func.count()).select_from(WatchlistItem)) == 1

    db.delete(user)
    db.flush()

    assert db.scalar(select(func.count()).select_from(Rating)) == 0
    assert db.scalar(select(func.count()).select_from(WatchlistItem)) == 0


def test_invalid_role_rejected(db):
    user = User(email=_email("role"), password_hash="x", role="superuser")
    db.add(user)
    with pytest.raises(IntegrityError):
        db.flush()


def test_timestamps_autopopulated(db):
    user = _user()
    db.add(user)
    db.flush()
    assert user.created_at is not None
    assert user.updated_at is not None


def test_movie_movielens_id_unique(db):
    mlid = uuid.uuid4().int % 100000000
    db.add(_movie("A", movielens_id=mlid))
    db.flush()
    db.add(_movie("B", movielens_id=mlid))
    with pytest.raises(IntegrityError):
        db.flush()


def test_uuid_primary_key_generated(db):
    user = _user()
    db.add(user)
    db.flush()
    assert isinstance(user.id, uuid.UUID)
