"""Refresh-session endpoint tests using an isolated in-memory SQLite database."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import reset_rate_limiters
from app.api.v1.endpoints import auth as auth_endpoint
from app.core import security
from app.core.config import Settings
from app.db.base import Base, utcnow
from app.db.models import RefreshSession
from app.db.session import get_db
from app.main import create_app


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, sessionmaker[Session]]]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_sessions = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    settings = Settings(
        APP_ENV="test",
        SECRET_KEY="test-auth-secret-key-that-is-long-enough",
        DATABASE_URL="sqlite+pysqlite:///:memory:",
        _env_file=None,
    )
    monkeypatch.setattr(auth_endpoint, "get_settings", lambda: settings)
    monkeypatch.setattr(security, "get_settings", lambda: settings)

    def override_get_db() -> Iterator[Session]:
        with test_sessions() as db:
            yield db

    app = create_app()
    app.dependency_overrides[get_db] = override_get_db
    reset_rate_limiters()
    client = TestClient(app)
    try:
        yield client, test_sessions
    finally:
        client.close()
        app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)
        engine.dispose()
        reset_rate_limiters()


def register(client: TestClient) -> tuple[str, str]:
    email = f"refresh-{uuid.uuid4().hex}@example.com"
    response = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "secure-password-123"},
    )
    assert response.status_code == 201
    return response.json()["access_token"], response.json()["csrf_token"]


def test_refresh_rotates_hashed_token_and_requires_csrf(auth_client):
    client, test_sessions = auth_client
    access_token, csrf_token = register(client)
    old_refresh_token = client.cookies.get("moviemind_refresh")

    cookies = client.get("/api/v1/auth/csrf")
    assert cookies.json()["csrf_token"] == csrf_token
    missing_csrf = client.post("/api/v1/auth/refresh")
    assert missing_csrf.status_code == 403

    response = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": csrf_token},
    )
    assert response.status_code == 200
    assert response.json()["access_token"] != access_token
    assert client.cookies.get("moviemind_refresh") != old_refresh_token
    assert response.json()["csrf_token"] != csrf_token

    with test_sessions() as db:
        rows = list(db.scalars(select(RefreshSession)))
    assert len(rows) == 2
    assert all(row.token_hash != old_refresh_token for row in rows)
    assert any(row.token_hash == hashlib.sha256(old_refresh_token.encode()).hexdigest() for row in rows)
    assert sum(row.revoked_at is not None for row in rows) == 1


def test_registration_sets_scoped_httponly_refresh_cookie(auth_client):
    client, _ = auth_client
    email = f"cookie-{uuid.uuid4().hex}@example.com"
    registered = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "secure-password-123"},
    )
    cookie_headers = registered.headers.get_list("set-cookie")
    refresh_cookie = next(value for value in cookie_headers if value.startswith("moviemind_refresh="))
    csrf_cookie = next(value for value in cookie_headers if value.startswith("moviemind_csrf="))

    assert registered.status_code == 201
    assert "httponly" in refresh_cookie.lower()
    assert "path=/api/v1/auth" in refresh_cookie.lower()
    assert "samesite=lax" in refresh_cookie.lower()
    assert "httponly" not in csrf_cookie.lower()
    assert "refresh_token" not in registered.json()


def test_replay_revokes_the_entire_refresh_family(auth_client):
    client, test_sessions = auth_client
    _, old_csrf = register(client)
    old_refresh = client.cookies.get("moviemind_refresh")

    rotated = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": old_csrf})
    assert rotated.status_code == 200
    new_refresh = client.cookies.get("moviemind_refresh")
    new_csrf = rotated.json()["csrf_token"]
    assert new_refresh and new_refresh != old_refresh

    replay = client.post(
        "/api/v1/auth/refresh",
        cookies={"moviemind_refresh": old_refresh, "moviemind_csrf": old_csrf},
        headers={"X-CSRF-Token": old_csrf},
    )
    assert replay.status_code == 401

    client.cookies.set("moviemind_refresh", new_refresh, path="/api/v1/auth")
    client.cookies.set("moviemind_csrf", new_csrf, path="/api/v1/auth")
    rejected_after_replay = client.post(
        "/api/v1/auth/refresh",
        headers={"X-CSRF-Token": new_csrf},
    )
    assert rejected_after_replay.status_code == 401

    with test_sessions() as db:
        rows = list(db.scalars(select(RefreshSession)))
    assert len(rows) == 2
    assert all(row.revoked_at is not None for row in rows)


def test_logout_revokes_refresh_family_and_clears_cookies(auth_client):
    client, test_sessions = auth_client
    _, csrf_token = register(client)

    response = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf_token})

    assert response.status_code == 204
    assert client.cookies.get("moviemind_refresh") is None
    assert client.post("/api/v1/auth/refresh").status_code == 401
    with test_sessions() as db:
        row = db.scalar(select(RefreshSession))
    assert row is not None and row.revoked_at is not None


def test_unknown_refresh_token_is_rejected(auth_client):
    client, _ = auth_client
    unknown_token = "unknown-refresh-token"
    csrf_token = client.get(
        "/api/v1/auth/csrf",
        cookies={"moviemind_refresh": unknown_token},
    ).json()["csrf_token"]

    response = client.post(
        "/api/v1/auth/refresh",
        cookies={
            "moviemind_refresh": unknown_token,
            "moviemind_csrf": csrf_token,
        },
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 401
    assert "refresh_token" not in response.json()


def test_expired_refresh_session_is_rejected(auth_client):
    client, test_sessions = auth_client
    _, csrf_token = register(client)
    with test_sessions() as db:
        row = db.scalar(select(RefreshSession))
        assert row is not None
        row.expires_at = utcnow() - timedelta(seconds=1)
        db.commit()

    response = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": csrf_token})

    assert response.status_code == 401
