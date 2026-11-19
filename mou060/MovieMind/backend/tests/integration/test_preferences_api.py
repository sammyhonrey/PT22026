from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.models.movie import Movie
from app.db.session import get_session_factory
from app.main import app

client = TestClient(app)


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _sample_movie_ids(count: int = 2) -> list[str]:
    try:
        with get_session_factory()() as session:
            movies = session.scalars(select(Movie).limit(count)).all()
            return [str(movie.id) for movie in movies]
    except Exception:  # noqa: BLE001
        return []


@pytest.mark.skipif(not _sample_movie_ids(), reason="catalogue movies unavailable")
def test_preferences_are_user_scoped_and_round_trip(register_user):
    token_a = register_user()
    token_b = register_user()
    movie_ids = _sample_movie_ids()

    payload_a = {
        "favorite_genres": ["Science Fiction", "Thriller"],
        "favorite_movie_ids": movie_ids,
        "description": "I want smart, tense science-fiction stories with twists.",
    }

    payload_b = {
        "favorite_genres": ["Romance", "Comedy"],
        "favorite_movie_ids": [],
        "description": "I enjoy warm and witty movies.",
    }

    r = client.put("/api/v1/preferences/me", json=payload_a, headers=_auth(token_a))
    assert r.status_code == 200
    body = r.json()
    assert body["favorite_genres"] == ["Science Fiction", "Thriller"]
    assert body["favorite_movie_ids"] == movie_ids
    assert body["description"] == payload_a["description"]

    r = client.get("/api/v1/preferences/me", headers=_auth(token_a))
    assert r.status_code == 200
    assert r.json()["favorite_genres"] == ["Science Fiction", "Thriller"]

    r = client.put("/api/v1/preferences/me", json=payload_b, headers=_auth(token_b))
    assert r.status_code == 200

    r = client.get("/api/v1/preferences/me", headers=_auth(token_a))
    assert r.status_code == 200
    assert r.json()["favorite_genres"] == ["Science Fiction", "Thriller"]

    r = client.get("/api/v1/preferences/me", headers=_auth(token_b))
    assert r.status_code == 200
    assert r.json()["favorite_genres"] == ["Romance", "Comedy"]


@pytest.mark.skipif(not _sample_movie_ids(), reason="catalogue movies unavailable")
def test_preferences_recommendations_use_saved_profile(register_user):
    token = register_user()
    movie_ids = _sample_movie_ids()

    payload = {
        "favorite_genres": ["Science Fiction", "Thriller"],
        "favorite_movie_ids": movie_ids,
        "description": "Intelligent science fiction and big ideas.",
    }

    r = client.post("/api/v1/preferences/me/recommend", json=payload, headers=_auth(token))
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "preference_based"
    assert body["is_fallback"] is False
    assert len(body["items"]) <= body["limit"]
    assert body["items"]
    assert all(item["source"] == "preference_based" for item in body["items"])


def test_preferences_require_authentication():
    assert client.get("/api/v1/preferences/me").status_code == 401
    assert client.put("/api/v1/preferences/me", json={"favorite_genres": ["Drama"]}).status_code == 401
    assert client.post("/api/v1/preferences/me/recommend", json={"favorite_genres": ["Drama"]}).status_code == 401
