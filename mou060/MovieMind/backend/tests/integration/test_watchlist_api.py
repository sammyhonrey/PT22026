"""Watchlist API tests (T7.3 acceptance criteria).

Runs against the real dev database and commits real rows during each request,
so every test uses the ``register_user`` fixture (a fresh user removed on
teardown) plus the pre-imported catalogue movies. The module is skipped when
the catalogue is empty.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.models.movie import Movie
from app.db.session import get_session_factory
from app.main import app

client = TestClient(app)


def _catalog_movie_ids(count: int = 2) -> list[uuid.UUID]:
    try:
        with get_session_factory()() as s:
            return [m.id for m in s.scalars(select(Movie).limit(count))]
    except Exception:  # noqa: BLE001 — no DB means "skip"
        return []


_MOVIE_IDS = _catalog_movie_ids()

pytestmark = pytest.mark.skipif(
    len(_MOVIE_IDS) < 2, reason="need at least 2 movies in the catalogue"
)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_add_and_list_watchlist(register_user):
    token = register_user()
    movie_id = _MOVIE_IDS[0]

    r = client.post(f"/api/v1/watchlist/{movie_id}", headers=_auth(token))
    assert r.status_code == 201
    assert r.json()["movie"]["id"] == str(movie_id)

    body = client.get("/api/v1/watchlist", headers=_auth(token)).json()
    assert body["total"] == 1
    assert body["items"][0]["movie"]["id"] == str(movie_id)


def test_duplicate_add_is_idempotent(register_user):
    token = register_user()
    headers = _auth(token)
    movie_id = _MOVIE_IDS[0]

    assert client.post(f"/api/v1/watchlist/{movie_id}", headers=headers).status_code == 201
    assert client.post(f"/api/v1/watchlist/{movie_id}", headers=headers).status_code == 200

    assert client.get("/api/v1/watchlist", headers=headers).json()["total"] == 1


def test_remove_from_watchlist(register_user):
    token = register_user()
    headers = _auth(token)
    movie_id = _MOVIE_IDS[0]

    client.post(f"/api/v1/watchlist/{movie_id}", headers=headers)
    assert client.delete(f"/api/v1/watchlist/{movie_id}", headers=headers).status_code == 204
    assert client.get("/api/v1/watchlist", headers=headers).json()["total"] == 0
    # Removing again is a 404 (already gone).
    assert client.delete(f"/api/v1/watchlist/{movie_id}", headers=headers).status_code == 404


def test_cross_user_isolation(register_user):
    token_a = register_user()
    token_b = register_user()
    movie_id = _MOVIE_IDS[0]

    client.post(f"/api/v1/watchlist/{movie_id}", headers=_auth(token_a))

    assert client.get("/api/v1/watchlist", headers=_auth(token_b)).json()["total"] == 0
    assert client.delete(f"/api/v1/watchlist/{movie_id}", headers=_auth(token_b)).status_code == 404
    assert client.get("/api/v1/watchlist", headers=_auth(token_a)).json()["total"] == 1


def test_unknown_movie_returns_404(register_user):
    token = register_user()
    r = client.post(f"/api/v1/watchlist/{uuid.uuid4()}", headers=_auth(token))
    assert r.status_code == 404


def test_requires_auth():
    movie_id = _MOVIE_IDS[0]
    assert client.get("/api/v1/watchlist").status_code == 401
    assert client.post(f"/api/v1/watchlist/{movie_id}").status_code == 401
