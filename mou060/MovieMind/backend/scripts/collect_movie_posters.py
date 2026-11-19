from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"
TMDB_API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
CACHE_SAVE_INTERVAL = 25
DEFAULT_TIMEOUT_SECONDS = 20


class TmdbRequestError(RuntimeError):
    pass


class TmdbTemporaryError(TmdbRequestError):
    pass


class TmdbAuthenticationError(TmdbRequestError):
    pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing a CSV header")
        return list(reader)


def _parse_id(value: str | None, field: str, *, optional: bool = False) -> int | None:
    if value is None or not value.strip():
        if optional:
            return None
        raise ValueError(f"Missing {field} in MovieLens data")

    try:
        result = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid {field} value: {value!r}") from error

    if result <= 0:
        raise ValueError(f"Invalid {field} value: {value!r}")
    return result


def load_movie_catalog(movies_csv: Path, links_csv: Path) -> list[dict[str, int | None]]:
    links: dict[int, int | None] = {}
    for row in _read_csv(links_csv):
        if "movieId" not in row or "tmdbId" not in row:
            raise ValueError(f"{links_csv} must contain movieId and tmdbId columns")
        movie_id = _parse_id(row["movieId"], "movieId")
        tmdb_id = _parse_id(row["tmdbId"], "tmdbId", optional=True)
        if movie_id is None:
            raise ValueError(f"Missing movieId in {links_csv}")
        if movie_id in links:
            raise ValueError(f"Duplicate movieId {movie_id} in {links_csv}")
        links[movie_id] = tmdb_id

    movies: list[dict[str, int | None]] = []
    seen_movie_ids: set[int] = set()
    for row in _read_csv(movies_csv):
        if "movieId" not in row:
            raise ValueError(f"{movies_csv} must contain a movieId column")
        movie_id = _parse_id(row["movieId"], "movieId")
        if movie_id is None:
            raise ValueError(f"Missing movieId in {movies_csv}")
        if movie_id in seen_movie_ids:
            raise ValueError(f"Duplicate movieId {movie_id} in {movies_csv}")
        seen_movie_ids.add(movie_id)
        movies.append({"movieId": movie_id, "tmdbId": links.get(movie_id)})
    return movies


def write_catalog_manifest(movies_csv: Path, links_csv: Path, output_path: Path) -> int:
    movies = load_movie_catalog(movies_csv, links_csv)
    _write_json(
        output_path,
        {
            "moviePosters": [
                {"movieId": movie["movieId"], "tmdbId": movie["tmdbId"], "posterUrl": None}
                for movie in movies
            ]
        },
    )
    return len(movies)


def write_posters_from_cache(
    movies_csv: Path,
    links_csv: Path,
    cache_path: Path,
    output_path: Path,
) -> int:
    movies = load_movie_catalog(movies_csv, links_csv)
    cache = load_cache(cache_path)
    _write_json(output_path, {"moviePosters": build_poster_records(movies, cache)})
    return len(movies)


def load_cache(cache_path: Path) -> dict[str, str | None]:
    if not cache_path.exists():
        return {}

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Poster cache is not valid JSON: {cache_path}") from error

    posters = data.get("posters") if isinstance(data, dict) else None
    if not isinstance(posters, dict):
        raise ValueError(f"Poster cache has an invalid structure: {cache_path}")

    cache: dict[str, str | None] = {}
    for tmdb_id, poster_path in posters.items():
        if not isinstance(tmdb_id, str) or (
            poster_path is not None
            and (not isinstance(poster_path, str) or not poster_path.startswith("/"))
        ):
            raise ValueError(f"Poster cache contains an invalid entry: {cache_path}")
        cache[tmdb_id] = poster_path
    return cache


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
        ) as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
            temporary_path = Path(file.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_cache(cache_path: Path, cache: dict[str, str | None]) -> None:
    _write_json(cache_path, {"posters": cache})


def build_poster_records(
    movies: list[dict[str, int | None]],
    cache: dict[str, str | None],
) -> list[dict[str, int | str | None]]:
    return [
        {
            "movieId": movie["movieId"],
            "tmdbId": movie["tmdbId"],
            "posterUrl": (
                f"{IMAGE_BASE_URL}{cache[str(movie['tmdbId'])]}"
                if movie["tmdbId"] is not None and cache.get(str(movie["tmdbId"]))
                else None
            ),
        }
        for movie in movies
    ]


def fetch_poster_path(tmdb_id: int, api_key: str) -> str | None:
    query = urlencode({"api_key": api_key})
    request = Request(
        f"{TMDB_API_URL.format(tmdb_id=tmdb_id)}?{query}",
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except HTTPError as error:
        if error.code == 404:
            return None
        if error.code in {401, 403}:
            raise TmdbAuthenticationError(
                "TMDB rejected the API key; check TMDB_API_KEY."
            ) from None
        if error.code == 429 or error.code >= 500:
            raise TmdbTemporaryError(
                f"TMDB returned a temporary HTTP {error.code} error for ID {tmdb_id}."
            ) from None
        raise TmdbRequestError(f"TMDB returned HTTP {error.code} for ID {tmdb_id}.") from None
    except (TimeoutError, URLError, ConnectionError, http.client.HTTPException):
        raise TmdbTemporaryError(f"Could not reach TMDB for ID {tmdb_id}.") from None
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TmdbRequestError(f"TMDB returned invalid JSON for ID {tmdb_id}.") from None

    if not isinstance(payload, dict):
        raise TmdbRequestError(f"TMDB returned an invalid response for ID {tmdb_id}.")
    poster_path = payload.get("poster_path")
    if poster_path is None:
        return None
    if not isinstance(poster_path, str) or not poster_path.startswith("/"):
        raise TmdbRequestError(f"TMDB returned an invalid poster path for ID {tmdb_id}.")
    return poster_path


def _fetch_with_retries(tmdb_id: int, api_key: str, retries: int) -> str | None:
    for attempt in range(retries + 1):
        try:
            return fetch_poster_path(tmdb_id, api_key)
        except TmdbTemporaryError:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("Retry loop exited unexpectedly")


def collect_posters(
    movies_csv: Path,
    links_csv: Path,
    output_path: Path,
    cache_path: Path,
    api_key: str,
    *,
    request_delay: float = 0.1,
    retries: int = 4,
) -> tuple[int, list[str]]:
    movies = load_movie_catalog(movies_csv, links_csv)
    cache = load_cache(cache_path)
    errors: dict[str, str] = {}
    new_lookups = 0

    try:
        for movie in movies:
            tmdb_id = movie["tmdbId"]
            if tmdb_id is not None:
                cache_key = str(tmdb_id)
                if cache_key not in cache and cache_key not in errors:
                    try:
                        cache[cache_key] = _fetch_with_retries(tmdb_id, api_key, retries)
                    except TmdbAuthenticationError:
                        raise
                    except TmdbRequestError as error:
                        errors[cache_key] = str(error)
                    new_lookups += 1
                    if new_lookups % CACHE_SAVE_INTERVAL == 0:
                        save_cache(cache_path, cache)
                        _write_json(
                            output_path, {"moviePosters": build_poster_records(movies, cache)}
                        )
                    if request_delay > 0:
                        time.sleep(request_delay)

    finally:
        save_cache(cache_path, cache)
        _write_json(output_path, {"moviePosters": build_poster_records(movies, cache)})

    error_summaries = [f"TMDB ID {tmdb_id}: {message}" for tmdb_id, message in errors.items()]
    return len(movies), error_summaries


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Collect static TMDB poster URLs for a MovieLens catalogue."
    )
    parser.add_argument("--movies-csv", type=Path, required=True)
    parser.add_argument("--links-csv", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "frontend" / "src" / "data" / "movie-posters.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=project_root / "backend" / "artifacts" / "tmdb_poster_cache.json",
    )
    parser.add_argument("--request-delay", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Write all MovieLens IDs with null URLs without making TMDB requests.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Write known poster URLs from the cache without making TMDB requests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    api_key = os.environ.get("TMDB_API_KEY")
    if not api_key and not args.catalog_only and not args.cache_only:
        parser.error("TMDB_API_KEY must be set in the environment; never pass or commit it.")
    if args.request_delay < 0:
        parser.error("--request-delay cannot be negative")
    if args.retries < 0:
        parser.error("--retries cannot be negative")

    try:
        if args.catalog_only:
            count = write_catalog_manifest(args.movies_csv, args.links_csv, args.output)
            print(
                f"Wrote {count} catalogue records with unverified null poster URLs to "
                f"{args.output}; rerun without --catalog-only to collect posters."
            )
            return 0

        if args.cache_only:
            count = write_posters_from_cache(
                args.movies_csv,
                args.links_csv,
                args.cache,
                args.output,
            )
            cache_size = len(load_cache(args.cache))
            print(
                f"Wrote {count} catalogue records using {cache_size} cached TMDB lookups "
                f"to {args.output}; rerun without --cache-only to collect missing posters."
            )
            return 0

        if api_key is None:
            parser.error("TMDB_API_KEY must be set in the environment.")
        count, errors = collect_posters(
            args.movies_csv,
            args.links_csv,
            args.output,
            args.cache,
            api_key,
            request_delay=args.request_delay,
            retries=args.retries,
        )
    except (OSError, ValueError, TmdbAuthenticationError) as error:
        print(f"Poster collection failed: {error}", file=sys.stderr)
        return 2

    print(f"Wrote poster records for {count} MovieLens movies to {args.output}.")
    if errors:
        print(f"{len(errors)} TMDB lookups failed; rerun to retry them.", file=sys.stderr)
        for error in errors[:20]:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
