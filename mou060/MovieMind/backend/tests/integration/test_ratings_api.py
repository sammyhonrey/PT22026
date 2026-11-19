"""Ratings API tests (T7.1/T7.2 acceptance criteria).

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


def test_rate_creates_rating_and_get_returns_it(register_user):
    token = register_user()
    movie_id = _MOVIE_IDS[0]

    r = client.post(
        "/api/v1/ratings", json={"movie_id": str(movie_id), "score": 4.0}, headers=_auth(token)
    )
    assert r.status_code == 201
    body = r.json()
    assert body["score"] == 4.0
    assert body["movie_id"] == str(movie_id)
    assert body["movie"]["id"] == str(movie_id)

    g = client.get(f"/api/v1/ratings/me/{movie_id}", headers=_auth(token))
    assert g.status_code == 200
    assert g.json()["score"] == 4.0


def test_rerate_updates_not_duplicates(register_user):
    token = register_user()
    movie_id = _MOVIE_IDS[0]
    headers = _auth(token)

    assert client.post(
        "/api/v1/ratings", json={"movie_id": str(movie_id), "score": 4.0}, headers=headers
    ).status_code == 201
    assert client.post(
        "/api/v1/ratings", json={"movie_id": str(movie_id), "score": 4.5}, headers=headers
    ).status_code == 201

    body = client.get("/api/v1/ratings/me", headers=headers).json()
    assert body["total"] == 1
    assert body["items"][0]["score"] == 4.5


def test_list_my_ratings_includes_movie_details(register_user):
    token = register_user()
    headers = _auth(token)
    for mid in _MOVIE_IDS:
        client.post("/api/v1/ratings", json={"movie_id": str(mid), "score": 3.5}, headers=headers)

    body = client.get("/api/v1/ratings/me", headers=headers).json()
    assert body["total"] == 2
    assert all("title" in item["movie"] for item in body["items"])


def test_delete_rating(register_user):
    token = register_user()
    headers = _auth(token)
    movie_id = _MOVIE_IDS[0]

    client.post("/api/v1/ratings", json={"movie_id": str(movie_id), "score": 2.0}, headers=headers)
    assert client.delete(f"/api/v1/ratings/me/{movie_id}", headers=headers).status_code == 204
    assert client.get(f"/api/v1/ratings/me/{movie_id}", headers=headers).status_code == 404


def test_cross_user_isolation(register_user):
    token_a = register_user()
    token_b = register_user()
    movie_id = _MOVIE_IDS[0]

    client.post(
        "/api/v1/ratings", json={"movie_id": str(movie_id), "score": 4.0}, headers=_auth(token_a)
    )

    # B cannot see or delete A's rating.
    assert client.get(f"/api/v1/ratings/me/{movie_id}", headers=_auth(token_b)).status_code == 404
    assert client.delete(f"/api/v1/ratings/me/{movie_id}", headers=_auth(token_b)).status_code == 404
    # A still has it.
    assert client.get(f"/api/v1/ratings/me/{movie_id}", headers=_auth(token_a)).status_code == 200


def test_unknown_movie_returns_404(register_user):
    token = register_user()
    r = client.post(
        "/api/v1/ratings", json={"movie_id": str(uuid.uuid4()), "score": 4.0}, headers=_auth(token)
    )
    assert r.status_code == 404


def test_invalid_score_returns_422(register_user):
    token = register_user()
    movie_id = _MOVIE_IDS[0]
    for score in (0.0, 0.7, 5.5, 6.0):
        r = client.post(
            "/api/v1/ratings", json={"movie_id": str(movie_id), "score": score}, headers=_auth(token)
        )
        assert r.status_code == 422


def test_requires_auth():
    movie_id = _MOVIE_IDS[0]
    assert client.post(
        "/api/v1/ratings", json={"movie_id": str(movie_id), "score": 4.0}
    ).status_code == 401
    assert client.get("/api/v1/ratings/me").status_code == 401
