"""Recommendation API tests (T8.2–T8.6 acceptance criteria).

- ``/recommendations/me``: cold-start users get an honest popularity fallback;
  users with enough ratings get content-based recommendations.
- Demo NCF endpoints: finite, sorted predictions for known MovieLens ids;
  unknown ids -> 404; gated by ``ENABLE_DEMO_ENDPOINTS``.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models.movie import Movie
from app.db.session import get_session_factory
from app.main import app
from app.services import model_service

client = TestClient(app)


def _catalog_movie_ids(count: int = 6) -> list[uuid.UUID]:
    try:
        with get_session_factory()() as s:
            return [m.id for m in s.scalars(select(Movie).limit(count))]
    except Exception:  # noqa: BLE001 — no DB means "skip"
        return []


_MOVIE_IDS = _catalog_movie_ids()


def _demo_recommender():
    try:
        return model_service.get_ncf_recommender()
    except Exception:  # noqa: BLE001 — no model means "skip"
        return None


_DEMO = _demo_recommender()
demo_mark = pytest.mark.skipif(_DEMO is None, reason="NCF model not available")


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- /recommendations/me orchestration (T8.4) ---------------------------------


def test_me_new_user_gets_popularity_fallback(register_user):
    token = register_user()
    r = client.get("/api/v1/recommendations/me", headers=_auth(token))
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "popularity"
    assert body["is_fallback"] is True
    assert len(body["items"]) <= body["limit"]
    assert all(i["source"] == "popularity" for i in body["items"])


@pytest.mark.skipif(len(_MOVIE_IDS) < 5, reason="need at least 5 catalogue movies")
def test_me_with_enough_ratings_uses_content_based(register_user):
    token = register_user()
    headers = _auth(token)
    for mid in _MOVIE_IDS[:5]:
        r = client.post(
            "/api/v1/ratings", json={"movie_id": str(mid), "score": 4.0}, headers=headers
        )
        assert r.status_code == 201

    r = client.get("/api/v1/recommendations/me", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "content_based"
    assert body["is_fallback"] is False


def test_me_requires_auth():
    assert client.get("/api/v1/recommendations/me").status_code == 401


# --- NCF demo endpoints (T8.2) ------------------------------------------------


@demo_mark
def test_demo_prediction_known_pair():
    user_id = _DEMO.known_user_ids()[0]
    movie_id = _DEMO.known_movie_ids()[0]
    r = client.get(
        "/api/v1/predictions/demo/movielens",
        params={"user_id": user_id, "movie_id": movie_id},
    )
    assert r.status_code == 200
    score = r.json()["score"]
    assert 0.5 <= score <= 5.0


@demo_mark
def test_demo_prediction_unknown_ids_404():
    known_movie = _DEMO.known_movie_ids()[0]
    assert (
        client.get(
            "/api/v1/predictions/demo/movielens",
            params={"user_id": "999999999", "movie_id": known_movie},
        ).status_code
        == 404
    )
    known_user = _DEMO.known_user_ids()[0]
    assert (
        client.get(
            "/api/v1/predictions/demo/movielens",
            params={"user_id": known_user, "movie_id": "999999999"},
        ).status_code
        == 404
    )


@demo_mark
def test_demo_recommendations_known_user_sorted_ncf():
    user_id = _DEMO.known_user_ids()[0]
    r = client.get(f"/api/v1/recommendations/demo/movielens/{user_id}", params={"limit": 5})
    assert r.status_code == 200
    items = r.json()
    assert items, "expected at least one demo recommendation"
    assert len(items) <= 5
    assert all(i["source"] == "ncf" for i in items)
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True)


@demo_mark
def test_demo_recommendations_unknown_user_404():
    assert client.get("/api/v1/recommendations/demo/movielens/999999999").status_code == 404


def test_demo_endpoints_disabled_returns_404(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "ENABLE_DEMO_ENDPOINTS", False)
    assert (
        client.get(
            "/api/v1/predictions/demo/movielens", params={"user_id": "1", "movie_id": "1"}
        ).status_code
        == 404
    )
    assert client.get("/api/v1/recommendations/demo/movielens/1").status_code == 404
