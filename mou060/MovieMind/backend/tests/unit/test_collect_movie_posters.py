from __future__ import annotations

import csv
import http.client
import json
from io import BytesIO
from pathlib import Path

from scripts.collect_movie_posters import (
    _fetch_with_retries,
    collect_posters,
)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_collect_posters_includes_all_movies_and_reuses_cached_tmdb_results(tmp_path, monkeypatch):
    movies_csv = tmp_path / "movies.csv"
    links_csv = tmp_path / "links.csv"
    output_path = tmp_path / "movie-posters.json"
    cache_path = tmp_path / "cache.json"
    write_csv(
        movies_csv,
        ["movieId", "title", "genres"],
        [
            {"movieId": "1", "title": "First", "genres": "Drama"},
            {"movieId": "2", "title": "Second", "genres": "Comedy"},
            {"movieId": "3", "title": "Third", "genres": "Action"},
        ],
    )
    write_csv(
        links_csv,
        ["movieId", "imdbId", "tmdbId"],
        [
            {"movieId": "1", "imdbId": "1", "tmdbId": "101"},
            {"movieId": "2", "imdbId": "2", "tmdbId": "101"},
            {"movieId": "3", "imdbId": "3", "tmdbId": ""},
        ],
    )
    calls: list[int] = []

    def fetch(tmdb_id: int, api_key: str, retries: int) -> str:
        assert api_key == "test-key"
        assert retries == 4
        calls.append(tmdb_id)
        return "/verified-poster.jpg"

    from scripts import collect_movie_posters

    monkeypatch.setattr(collect_movie_posters, "_fetch_with_retries", fetch)
    count, errors = collect_posters(
        movies_csv,
        links_csv,
        output_path,
        cache_path,
        "test-key",
        request_delay=0,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))["moviePosters"]
    assert count == 3
    assert errors == []
    assert calls == [101]
    assert result == [
        {
            "movieId": 1,
            "tmdbId": 101,
            "posterUrl": "https://image.tmdb.org/t/p/w342/verified-poster.jpg",
        },
        {
            "movieId": 2,
            "tmdbId": 101,
            "posterUrl": "https://image.tmdb.org/t/p/w342/verified-poster.jpg",
        },
        {"movieId": 3, "tmdbId": None, "posterUrl": None},
    ]

    second_output_path = tmp_path / "second-movie-posters.json"
    collect_posters(
        movies_csv,
        links_csv,
        second_output_path,
        cache_path,
        "test-key",
        request_delay=0,
    )
    assert calls == [101]


def test_write_catalog_manifest_keeps_every_movie_without_claiming_verified_posters(tmp_path):
    from scripts.collect_movie_posters import write_catalog_manifest

    movies_csv = tmp_path / "movies.csv"
    links_csv = tmp_path / "links.csv"
    output_path = tmp_path / "movie-posters.json"
    write_csv(
        movies_csv,
        ["movieId", "title", "genres"],
        [
            {"movieId": "1", "title": "First", "genres": "Drama"},
            {"movieId": "2", "title": "Second", "genres": "Comedy"},
        ],
    )
    write_csv(
        links_csv,
        ["movieId", "tmdbId"],
        [{"movieId": "1", "tmdbId": "101"}, {"movieId": "2", "tmdbId": ""}],
    )

    count = write_catalog_manifest(movies_csv, links_csv, output_path)

    assert count == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "moviePosters": [
            {"movieId": 1, "tmdbId": 101, "posterUrl": None},
            {"movieId": 2, "tmdbId": None, "posterUrl": None},
        ]
    }


def test_write_posters_from_cache_writes_verified_cached_urls_and_nulls(tmp_path):
    from scripts.collect_movie_posters import write_posters_from_cache

    movies_csv = tmp_path / "movies.csv"
    links_csv = tmp_path / "links.csv"
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "movie-posters.json"
    write_csv(
        movies_csv,
        ["movieId", "title", "genres"],
        [
            {"movieId": "1", "title": "First", "genres": "Drama"},
            {"movieId": "2", "title": "Second", "genres": "Comedy"},
        ],
    )
    write_csv(
        links_csv,
        ["movieId", "tmdbId"],
        [{"movieId": "1", "tmdbId": "101"}, {"movieId": "2", "tmdbId": "102"}],
    )
    cache_path.write_text(
        json.dumps({"posters": {"101": "/verified.jpg", "102": None}}),
        encoding="utf-8",
    )

    count = write_posters_from_cache(movies_csv, links_csv, cache_path, output_path)

    assert count == 2
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "moviePosters": [
            {
                "movieId": 1,
                "tmdbId": 101,
                "posterUrl": "https://image.tmdb.org/t/p/w342/verified.jpg",
            },
            {"movieId": 2, "tmdbId": 102, "posterUrl": None},
        ]
    }


def test_fetch_retries_when_tmdb_closes_the_connection(monkeypatch):
    from scripts import collect_movie_posters

    responses = iter([BytesIO(b'{"poster_path": "/verified.jpg"}')])
    attempts = 0

    def close_then_respond(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http.client.RemoteDisconnected("connection closed")
        return next(responses)

    monkeypatch.setattr(collect_movie_posters, "urlopen", close_then_respond)
    monkeypatch.setattr(collect_movie_posters.time, "sleep", lambda _: None)

    assert _fetch_with_retries(101, "test-key", retries=1) == "/verified.jpg"
    assert attempts == 2
