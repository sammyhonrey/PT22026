"""Arabic metadata catalogue and metadata-based discovery.

This module is deliberately separate from the international audio engine.

* The international catalogue (``preprocessing`` + ``recommender``) ranks tracks by
  **cosine similarity over nine audio features**.
* This module has **no audio features at all**. It ranks Arabic tracks by verified
  metadata: same credited artist, same curated genre, same region.

The two scores are never compared or blended: one is an audio similarity between
feature vectors, the other is a metadata match between catalogue fields. Any UI
showing both must label them differently.

Source and licence
------------------
Metadata comes from **MusicBrainz**, released under **CC0 1.0**. Region and genre
are editorial labels chosen by this project (documented per row). No audio is
stored or redistributed, and no music platform is scraped.

Typical use::

    from arabic_catalogue import load_arabic_catalogue

    catalogue = load_arabic_catalogue()
    hits = catalogue.search("فيروز")
    results = catalogue.recommend(hits.iloc[0]["track_id"], limit=10)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CATALOGUE_PATH = DATA_DIR / "arabic_catalogue.csv"
PROVENANCE_PATH = DATA_DIR / "arabic_catalogue_provenance.json"

#: Additional catalogue files merged into the default load (additive; the base
#: file is never modified).
EXTRA_CATALOGUE_PATHS = (DATA_DIR / "arabic_catalogue_yemeni.csv",)

#: Phase 12.1 corrected Yemeni catalogue (verified MBIDs only). When present it
#: REPLACES the Phase 12 Yemeni file in the default load; the Phase 12 files are
#: retained on disk untouched.
CORRECTED_YEMENI_PATH = DATA_DIR / "arabic_catalogue_yemeni_v2.csv"
CORRECTED_YEMENI_PROVENANCE = DATA_DIR / "arabic_catalogue_yemeni_v2_provenance.json"


def extra_catalogue_paths() -> tuple[Path, ...]:
    """Which extra catalogue files the default loader should merge."""
    if CORRECTED_YEMENI_PATH.is_file():
        return (CORRECTED_YEMENI_PATH,)
    return EXTRA_CATALOGUE_PATHS

#: MusicBrainz artist aliases (CC0), keyed by artist MBID.
ALIASES_PATH = DATA_DIR / "arabic_artist_aliases.json"

REQUIRED_COLUMNS = (
    "track_id",
    "title",
    "artist",
    "album",
    "genre",
    "region",
    "source",
    "source_url",
)

#: Arabic diacritics/tatweel that must be removed before matching.
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_NON_WORD = re.compile(r"[^\w\u0600-\u06FF]+")

#: Metadata match weights (documented, not audio similarity).
WEIGHT_ARTIST = 0.60
WEIGHT_GENRE = 0.30
WEIGHT_REGION = 0.10


def normalize_text(text: object) -> str:
    """Normalise Arabic/Latin text for search and comparison.

    * Unicode NFKC, case-folded for Latin
    * Arabic diacritics and tatweel removed
    * alef variants (أ إ آ ٱ) folded to ا, ى → ي, ة → ه
    * punctuation collapsed to single spaces
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = _DIACRITICS.sub("", value).replace(_TATWEEL, "")
    value = (
        value.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ٱ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    value = _NON_WORD.sub(" ", value)
    return " ".join(value.split())


def is_arabic_script(text: object) -> bool:
    """True when the text contains at least one Arabic-script character."""
    return bool(re.search(r"[\u0600-\u06FF]", str(text or "")))


@dataclass(frozen=True)
class ArabicStats:
    """Catalogue facts shown in the UI and the report."""

    tracks: int
    artists: int
    genres: int
    regions: int
    arabic_titles: int
    arabic_artists: int


class ArabicCatalogueError(RuntimeError):
    """Raised when the Arabic catalogue is missing or malformed."""


class ArabicCatalogue:
    """Metadata-only Arabic discovery.

    Parameters
    ----------
    frame:
        Validated catalogue rows (see :func:`load_arabic_catalogue`).
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ArabicCatalogueError(f"Arabic catalogue is missing columns: {missing}")

        working = frame.copy()
        working["title_key"] = working["title"].map(normalize_text)
        working["artist_key"] = working["artist"].map(normalize_text)
        working["album_key"] = working["album"].map(normalize_text)
        # Latin transliteration of the seed artist, for English-language search.
        if "artist_latin" in working.columns:
            working["artist_latin_key"] = working["artist_latin"].map(normalize_text)
        else:
            working["artist_latin_key"] = ""

        working = working[(working["title_key"] != "") & (working["artist_key"] != "")]
        working = working.drop_duplicates(subset=["title_key", "artist_key"], keep="first")
        working = working.reset_index(drop=True)

        self._tracks = working
        self._by_id = working.set_index("track_id", drop=False)

    # -- facts --------------------------------------------------------------- #

    @property
    def tracks(self) -> pd.DataFrame:
        return self._tracks

    def stats(self) -> ArabicStats:
        frame = self._tracks
        return ArabicStats(
            tracks=len(frame),
            artists=int(frame["artist"].nunique()),
            genres=int(frame["genre"].nunique()),
            regions=int(frame["region"].nunique()),
            arabic_titles=int(frame["title"].map(is_arabic_script).sum()),
            arabic_artists=int(frame["artist"].map(is_arabic_script).sum()),
        )

    def genres(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks["genre"].dropna().unique()))

    def regions(self) -> tuple[str, ...]:
        return tuple(sorted(self._tracks["region"].dropna().unique()))

    def artists(self, limit: int | None = None) -> tuple[str, ...]:
        values = tuple(sorted(self._tracks["artist"].dropna().unique()))
        return values[:limit] if limit else values

    # -- search -------------------------------------------------------------- #

    def search(self, query: str, limit: int = 25) -> pd.DataFrame:
        """Search by title or artist, in Arabic or Latin script.

        An empty query returns a deterministic sample (most common artists), so
        the page always offers something to pick.
        """
        key = normalize_text(query)
        if not key:
            return self._tracks.head(limit)

        tokens = [token for token in key.split() if token]
        titles = self._tracks["title_key"]
        artists = self._tracks["artist_key"]
        latin = self._tracks["artist_latin_key"]

        mask = titles.str.contains(key, regex=False) | artists.str.contains(key, regex=False)
        if not mask.any():
            # all-token match, tolerant of word order and transliteration
            mask = pd.Series(True, index=self._tracks.index)
            for token in tokens:
                mask &= (
                    titles.str.contains(token, regex=False)
                    | artists.str.contains(token, regex=False)
                    | latin.str.contains(token, regex=False)
                )
            if not mask.any():
                mask = latin.str.contains(key, regex=False)

        hits = self._tracks.loc[mask]
        return hits.head(limit)

    def get(self, track_id: str) -> pd.Series:
        """One catalogue row by id (raises ``KeyError`` if unknown)."""
        if track_id not in self._by_id.index:
            raise KeyError(f"unknown Arabic track_id: {track_id!r}")
        return self._by_id.loc[track_id]

    # -- metadata-based discovery -------------------------------------------- #

    def recommend(
        self,
        track_id: str,
        limit: int = 10,
        genres: tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Find tracks by **metadata** similarity to the reference.

        Scoring (0–1) is explicit and fully metadata-based:

        * same credited artist → +0.60
        * same curated genre  → +0.30
        * same region         → +0.10

        The result frame carries a transparent ``explanation`` and is ordered by
        score then title, so it is deterministic. It is **not** an audio score and
        must never be presented as one.
        """
        reference = self.get(track_id)
        candidates = self._tracks[self._tracks["track_id"] != track_id]
        if genres:
            candidates = candidates[candidates["genre"].isin(genres)]
        if candidates.empty:
            return pd.DataFrame(columns=self._result_columns())

        rows: list[tuple[int, float, str]] = []
        for idx, row in candidates.iterrows():
            score = 0.0
            reasons = []
            if row["artist_key"] == reference["artist_key"]:
                score += WEIGHT_ARTIST
                reasons.append(f"same artist: {row['artist']}")
            if row["genre"] == reference["genre"]:
                score += WEIGHT_GENRE
                reasons.append(f"same genre: {row['genre']}")
            if row["region"] == reference["region"]:
                score += WEIGHT_REGION
                reasons.append(f"same region: {row['region']}")
            if reasons:
                rows.append((idx, round(score, 2), "; ".join(reasons)))

        if not rows:
            return pd.DataFrame(columns=self._result_columns())

        result = candidates.loc[[idx for idx, _, _ in rows]].copy()
        result["metadata_score"] = [score for _, score, _ in rows]
        result["explanation"] = [text for _, _, text in rows]
        result = result.sort_values(
            ["metadata_score", "title_key", "artist_key"], ascending=[False, True, True]
        ).head(limit)

        return pd.DataFrame(
            {
                "rank": range(1, len(result) + 1),
                "track_id": result["track_id"].to_numpy(),
                "title": result["title"].to_numpy(),
                "artist": result["artist"].to_numpy(),
                "album": result["album"].to_numpy(),
                "genre": result["genre"].to_numpy(),
                "region": result["region"].to_numpy(),
                "metadata_score": result["metadata_score"].to_numpy(),
                "explanation": result["explanation"].to_numpy(),
            }
        )

    @staticmethod
    def _result_columns() -> list[str]:
        return [
            "rank",
            "track_id",
            "title",
            "artist",
            "album",
            "genre",
            "region",
            "metadata_score",
            "explanation",
        ]


def load_arabic_catalogue(path: str | Path | None = None) -> ArabicCatalogue:
    """Load and validate the Arabic metadata catalogue.

    With no ``path`` the base catalogue is loaded **and merged** with any extra
    catalogue files that exist (Phase 12 Yemeni expansion). Passing an explicit
    path loads exactly that file.

    Raises
    ------
    ArabicCatalogueError
        If the file is missing, empty, or lacks required columns.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CATALOGUE_PATH
    if not csv_path.is_file():
        raise ArabicCatalogueError(
            f"Arabic catalogue not found at {csv_path}. "
            "Build it with: python scripts/fetch_arabic_metadata.py"
        )
    try:
        frame = pd.read_csv(csv_path, encoding="utf-8")
        if path is None:
            for extra in extra_catalogue_paths():
                if extra.is_file():
                    extra_frame = pd.read_csv(extra, encoding="utf-8")
                    shared = [c for c in frame.columns if c in extra_frame.columns]
                    frame = pd.concat(
                        [frame[shared], extra_frame[shared]], ignore_index=True
                    )
    except Exception as exc:  # noqa: BLE001
        raise ArabicCatalogueError(f"Could not read the Arabic catalogue: {exc}") from exc

    if frame.empty:
        raise ArabicCatalogueError("Arabic catalogue is empty.")
    return ArabicCatalogue(frame)


def load_artist_aliases() -> dict[str, tuple[str, ...]]:
    """MusicBrainz aliases per artist MBID (empty dict when the file is absent)."""
    import json

    if not ALIASES_PATH.is_file():
        return {}
    try:
        payload = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    result: dict[str, tuple[str, ...]] = {}
    for mbid, info in (payload.get("artists") or {}).items():
        aliases = tuple(a for a in (info.get("aliases") or ()) if str(a).strip())
        if aliases:
            result[str(mbid)] = aliases
    return result


def provenance() -> dict:
    """Return the recorded dataset provenance (licence, source, counts)."""
    import json

    if not PROVENANCE_PATH.is_file():
        return {}
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
