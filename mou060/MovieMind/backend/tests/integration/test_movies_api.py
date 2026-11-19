"""Movie catalogue API tests (T5.3 acceptance criteria).

These run against the imported catalogue (committed data) and skip when no
movies are present. The integration directory is already skipped when the DB
is unreachable (see conftest.py).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.models.movie import Movie
from app.db.session import get_session_factory
from app.main import app

client = TestClient(app)


def _first_movie_id():
    with get_session_factory()() as s:
        movie = s.scalar(select(Movie).limit(1))
        return movie.id if movie else None


_FIRST_MOVIE_ID = _first_movie_id()

pytestmark = pytest.mark.skipif(
    _FIRST_MOVIE_ID is None, reason="no movies imported into the catalogue"
)


def test_list_movies_returns_page_with_meta():
    r = client.get("/api/v1/movies", params={"limit": 5, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 5
    assert body["offset"] == 0
    assert len(body["items"]) <= 5
    assert body["total"] >= len(body["items"])
    item = body["items"][0]
    assert set(item.keys()) >= {"id", "title", "genres", "movielens_id"}


def test_search_is_case_insensitive():
    r = client.get("/api/v1/movies", params={"q": "toy story"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items
    assert all("toy story" in m["title"].lower() for m in items)


def test_genre_filter():
    r = client.get("/api/v1/movies", params={"genre": "Drama", "limit": 20})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items
    assert all("Drama" in m["genres"] for m in items)


def test_year_filter():
    r = client.get("/api/v1/movies", params={"year": 1995, "limit": 20})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items
    assert all(m["release_year"] == 1995 for m in items)


def test_invalid_sort_returns_422():
    r = client.get("/api/v1/movies", params={"sort": "bogus"})
    assert r.status_code == 422


def test_limit_validation():
    assert client.get("/api/v1/movies", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/movies", params={"limit": 1000}).status_code == 422


def test_invalid_year_returns_422():
    r = client.get("/api/v1/movies", params={"year": 1800})
    assert r.status_code == 422


def test_get_movie_by_id():
    r = client.get(f"/api/v1/movies/{_FIRST_MOVIE_ID}")
    assert r.status_code == 200
    assert r.json()["id"] == str(_FIRST_MOVIE_ID)


def test_get_unknown_movie_returns_404():
    from uuid import uuid4

    r = client.get(f"/api/v1/movies/{uuid4()}")
    assert r.status_code == 404


def test_genres_endpoint():
    r = client.get("/api/v1/movies/genres")
    assert r.status_code == 200
    names = {g["name"] for g in r.json()}
    assert names
    assert "Drama" in names


def test_popular_endpoint():
    r = client.get("/api/v1/movies/popular", params={"limit": 10})
    assert r.status_code == 200
    items = r.json()
    assert len(items) <= 10
