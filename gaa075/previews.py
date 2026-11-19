"""Legitimate audio previews for recommended songs.

Provider: the **public iTunes Search API** (``itunes.apple.com/search``).

Why this provider
-----------------
* No API key, no OAuth, no account — it works from a plain HTTP GET.
* It returns official ~30-second AAC/M4A previews (verified on this machine: a
  30-second ``m4a`` served from Apple's CDN with ``audio/x-m4p``).
* Apple serves these previews publicly for promotional listening, and this app
  **streams** them from Apple's CDN in the user's browser. Nothing is downloaded,
  stored, or redistributed by this project.
* Deezer's public API was unreachable from this network (request timeout), and
  Spotify's Web API requires OAuth client credentials and no longer exposes 30s
  preview URLs for newly registered applications — so neither is used here.

What is guaranteed
------------------
A preview is returned **only** when a search result verifiably matches both the
artist and the title; otherwise the caller gets ``None`` and the UI must show
"Preview unavailable" plus a search link for the song. The module never invents an
audio URL.

Only the standard library is used here (``urllib``), so the app keeps working
without extra dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
APPLE_MUSIC_SEARCH_URL = "https://music.apple.com/search"
APPLE_MUSIC_HOME = "https://music.apple.com"

PROVIDER = "Apple iTunes preview"
PREVIEW_SECONDS = 30
REQUEST_TIMEOUT = 6.0
MAX_CANDIDATES = 8
MIN_TITLE_SIMILARITY = 0.62
USER_AGENT = "MusicRecommendationSystem/1.0 (university project)"

#: Words that mark a candidate as a cover/tribute act rather than the real artist.
COVER_MARKERS = frozenset(
    {"tribute", "karaoke", "cover", "covers", "impersonator", "soundalike", "parody"}
)


@dataclass(frozen=True)
class Preview:
    """A verified, legally streamable preview for one catalogue track."""

    url: str
    provider: str
    external_url: str
    artwork_url: str
    matched_title: str
    matched_artist: str


def _normalise(text: str) -> str:
    """Lower-case, strip punctuation/whitespace noise for matching."""
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in str(text).lower())
    return " ".join(cleaned.split())


def _primary_artist(artists: str) -> str:
    """First credited artist (dataset credits are semicolon-separated)."""
    return str(artists).split(";")[0].strip()


def title_similarity(left: str, right: str) -> float:
    """Similarity of two titles in [0, 1], tolerant of remaster/parenthetical noise."""
    a, b = _normalise(left), _normalise(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    if a in b or b in a:
        ratio = max(ratio, 0.9)
    return ratio


def artist_matches(expected: str, candidate: str) -> bool:
    """True when the catalogue artist and the API artist clearly refer to the same act.

    Exact token sets match, and one-sided extra tokens are tolerated when they are
    ordinary credit words (``"Sir Elton John"``, ``"Elton John & Britney Spears"``).
    Extra tokens that mark a cover or tribute act are rejected, so a tribute band
    can never supply a preview for the real artist.
    """
    expected_tokens = set(_normalise(expected).split())
    candidate_tokens = set(_normalise(candidate).split())
    if not expected_tokens or not candidate_tokens:
        return False
    if expected_tokens == candidate_tokens:
        return True

    smaller, larger = sorted((expected_tokens, candidate_tokens), key=len)
    if not smaller.issubset(larger):
        return False
    extra = larger - smaller
    return not (extra & COVER_MARKERS)


def _http_json(url: str, timeout: float = REQUEST_TIMEOUT) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed host
        payload = response.read()
    return json.loads(payload.decode("utf-8", errors="replace"))


def search_url(title: str, artists: str) -> str:
    """A legitimate external listening link for a song (search, not audio)."""
    query = urllib.parse.quote_plus(f"{_primary_artist(artists)} {title}".strip())
    return f"{APPLE_MUSIC_SEARCH_URL}?term={query}"


def _pick_candidate(results: list[dict], title: str, artists: str) -> dict | None:
    """Choose the best verifiable match; return None if nothing matches honestly."""
    expected_artist = _primary_artist(artists)
    best: tuple[float, dict] | None = None

    for candidate in results:
        candidate_title = candidate.get("trackName") or ""
        candidate_artist = candidate.get("artistName") or ""
        preview_url = candidate.get("previewUrl") or ""
        if not preview_url:
            continue
        if not artist_matches(expected_artist, candidate_artist):
            continue
        similarity = title_similarity(title, candidate_title)
        if similarity < MIN_TITLE_SIMILARITY:
            continue
        if best is None or similarity > best[0]:
            best = (similarity, candidate)

    return best[1] if best is not None else None


@lru_cache(maxsize=1024)
def find_preview(title: str, artists: str) -> Preview | None:
    """Look up a verified 30-second preview, or ``None`` when none is available.

    Failures are swallowed deliberately: a missing preview must never break the
    recommendation flow. Results are cached per (title, artists) for the process
    lifetime.
    """
    if not str(title).strip() or not str(artists).strip():
        return None

    query = urllib.parse.urlencode(
        {
            "term": f"{_primary_artist(artists)} {title}",
            "media": "music",
            "entity": "song",
            "limit": MAX_CANDIDATES,
        }
    )
    try:
        payload = _http_json(f"{ITUNES_SEARCH_URL}?{query}")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None

    results = payload.get("results") or []
    candidate = _pick_candidate(results, title, artists)
    if candidate is None:
        return None

    artwork = str(candidate.get("artworkUrl100") or "").replace("100x100", "300x300")
    return Preview(
        url=str(candidate["previewUrl"]),
        provider=PROVIDER,
        external_url=str(candidate.get("trackViewUrl") or search_url(title, artists)),
        artwork_url=artwork,
        matched_title=str(candidate.get("trackName") or title),
        matched_artist=str(candidate.get("artistName") or artists),
    )


def resolve_many(items: list[tuple[str, str, str]], max_workers: int = 6) -> dict[str, Preview | None]:
    """Resolve previews for many tracks concurrently.

    Parameters
    ----------
    items:
        ``(key, title, artists)`` triples; ``key`` is usually the ``track_id``.

    Returns
    -------
    dict
        ``key -> Preview | None``. Never raises: unavailable previews are ``None``.
    """
    keys = [key for key, _, _ in items]

    def work(item: tuple[str, str, str]) -> Preview | None:
        _, title, artists = item
        try:
            return find_preview(title, artists)
        except Exception:  # noqa: BLE001 - a preview must never break recommendations
            return None

    workers = max(1, min(max_workers, len(items) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved = list(pool.map(work, items))

    return dict(zip(keys, resolved, strict=True))
