"""Idempotent MovieLens movie-metadata import.

Usage (from the ``backend/`` directory, with the venv active):

    python scripts/import_movielens.py --source artifacts/movies.json
    python scripts/import_movielens.py --source path/to/movies.csv

Supported sources (auto-detected by extension):
- ``*.csv``  — MovieLens ``movies.csv`` (``movieId,title,genres``).
- ``*.json`` — the model bundle's ``movies.json`` (list of
  ``{movieId, title, genres}``).

Behavior:
- Upserts by ``movielens_id`` so re-runs never duplicate records.
- Parses the release year from the ``"Title (YYYY)"`` suffix.
- Splits the ``|``-separated genre string into normalized ``Genre`` rows.
- Does **not** import ratings — MovieLens training ratings are never treated
  as MovieMind user ratings.
- Reports imported / updated / skipped / failed counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Allow running as `python scripts/import_movielens.py` from backend/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models.movie import Genre, Movie, MovieGenre
from app.db.session import get_session_factory

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _parse_year(title: str) -> int | None:
    m = _YEAR_RE.search(title)
    if not m:
        return None
    year = int(m.group(1))
    return year if 1888 <= year <= 2100 else None


def _load_rows(source: Path) -> list[dict]:
    """Return a list of {movie_id, title, genres} regardless of format."""
    if source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"{source} is not a JSON array of movie records")
        return [
            {
                "movie_id": int(rec["movieId"]),
                "title": str(rec["title"]).strip(),
                "genres": str(rec.get("genres", "")).strip(),
            }
            for rec in data
        ]
    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [
                {
                    "movie_id": int(row["movieId"]),
                    "title": str(row["title"]).strip(),
                    "genres": str(row.get("genres", "")).strip(),
                }
                for row in reader
            ]
    raise ValueError(f"Unsupported source format: {source.suffix} (expected .csv or .json)")


def _genre_names(raw: str) -> list[str]:
    if not raw or raw == "(no genres listed)":
        return []
    return [g.strip() for g in raw.split("|") if g.strip()]


def import_movies(session: Session, rows: list[dict]) -> dict[str, int]:
    """Upsert movies + genres. Returns counts dict."""
    counts = {"imported": 0, "updated": 0, "skipped": 0, "failed": 0}

    # Preload existing movies (with genres) and all genres for O(1) lookups.
    existing = {
        m.movielens_id: m
        for m in session.scalars(
            select(Movie)
            .where(Movie.movielens_id.is_not(None))
            .options(selectinload(Movie.genre_links).selectinload(MovieGenre.genre))
        ).all()
    }
    genres: dict[str, Genre] = {g.name: g for g in session.scalars(select(Genre)).all()}

    def get_genre(name: str) -> Genre:
        g = genres.get(name)
        if g is None:
            g = Genre(name=name)
            session.add(g)
            genres[name] = g
        return g

    for row in rows:
        try:
            movie_id = row["movie_id"]
            title = row["title"]
            desired = _genre_names(row["genres"])
            year = _parse_year(title)

            movie = existing.get(movie_id)
            if movie is None:
                movie = Movie(movielens_id=movie_id, title=title, release_year=year)
                session.add(movie)
                existing[movie_id] = movie
                counts["imported"] += 1
            else:
                changed = False
                if movie.title != title:
                    movie.title = title
                    changed = True
                if movie.release_year != year:
                    movie.release_year = year
                    changed = True
                counts["updated" if changed else "skipped"] += 1

            # Reconcile genre links.
            current = {link.genre.name for link in movie.genre_links}
            for name in desired:
                if name not in current:
                    movie.genre_links.append(MovieGenre(genre=get_genre(name)))
            for link in list(movie.genre_links):
                if link.genre.name not in desired:
                    movie.genre_links.remove(link)
        except Exception:  # noqa: BLE001 — record and continue on malformed rows
            counts["failed"] += 1

    session.commit()
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="artifacts/movies.json",
        help="Path to movies.json (bundle) or movies.csv (MovieLens).",
    )
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_file():
        print(f"ERROR: source file not found: {source}", file=sys.stderr)
        return 1

    rows = _load_rows(source)
    print(f"Loaded {len(rows)} rows from {source}")

    session_factory = get_session_factory()
    with session_factory() as session:
        counts = import_movies(session, rows)

    print(
        f"imported={counts['imported']} updated={counts['updated']} "
        f"skipped={counts['skipped']} failed={counts['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
