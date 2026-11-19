"""Integration test fixtures.

Tests in this directory run against a real PostgreSQL database (the dev/Neon
database configured via ``DATABASE_URL``). Two kinds of fixtures keep it clean:

- ``db`` — a transactional session that flushes but never commits (rolled back).
- ``register_user`` — registers a real user through the HTTP API and deletes
  that user (plus any cascaded ratings/watchlist) on teardown, so HTTP-level
  tests that must commit leave no residue behind.

The whole directory is skipped when the database is unreachable.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.models.user import User
from app.db.session import get_engine, get_session_factory
from app.main import app


def _db_available() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 — any failure means "skip"
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="PostgreSQL not reachable")


@pytest.fixture
def db() -> Session:
    """Session bound to an uncommitted connection; rolled back after each test."""
    connection = get_engine().connect()
    session = Session(bind=connection, expire_on_commit=False, autoflush=False)
    try:
        yield session
        session.rollback()
    finally:
        session.close()
        connection.close()


@pytest.fixture
def register_user():
    """Return a callable that registers a fresh user and returns its bearer token.

    Every user registered this way is deleted on teardown; the ``ON DELETE
    CASCADE`` foreign keys remove their ratings/watchlist too, so HTTP tests
    that commit against the dev database do not pollute it.
    """
    client = TestClient(app)
    created_ids: list[str] = []

    def _register() -> str:
        r = client.post(
            "/api/v1/auth/register",
            json={"email": f"it-{uuid.uuid4().hex[:12]}@example.com", "password": "password123"},
        )
        assert r.status_code == 201
        body = r.json()
        created_ids.append(body["user"]["id"])
        return body["access_token"]

    yield _register

    with get_session_factory()() as s:
        for raw_id in created_ids:
            user = s.get(User, uuid.UUID(raw_id))
            if user is not None:
                s.delete(user)
        s.commit()
