"""Fetch a licence-clean Arabic music metadata catalogue from MusicBrainz (CC0).

What it does
------------
For a curated seed list of Arabic artists (each labelled with a region and a
curated genre category), it resolves the MusicBrainz artist, downloads their
releases with track lists, and writes a flat CSV plus a provenance JSON.

Sources and licence
-------------------
MusicBrainz core data is released under **CC0 1.0** (public domain dedication).
Attribution is not required but is recorded anyway. No audio is downloaded, no
music platform is scraped, and nothing copyrighted is redistributed.

Usage
-----
    .venv\\Scripts\\python.exe scripts\\fetch_arabic_metadata.py

Output (both git-ignored through ``data/*.csv`` and ``data/*.json``):
    data/arabic_catalogue.csv
    data/arabic_catalogue_provenance.json
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "arabic_catalogue.csv"
PROVENANCE_PATH = DATA_DIR / "arabic_catalogue_provenance.json"

MB = "https://musicbrainz.org/ws/2"
UA = "MusicRecommendationSystem/1.0 (university project; educational use)"
PAUSE = 1.15
RELEASE_LIMIT = 100

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


def norm(text: str) -> str:
    """Case/punctuation/diacritic-insensitive key used for name comparison."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = ARABIC_DIACRITICS.sub("", text)
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text)
    return " ".join(text.split())


#: Curated seed list. ``region`` and ``genre`` are **editorial labels chosen by the
#: project** (not MusicBrainz facts) and are recorded as such in the CSV
#: (``region_source``/``genre_source``). ``search`` overrides the query when the
#: Latin name is ambiguous in MusicBrainz. Everything else comes from MusicBrainz.
SEED = [
    ("Amr Diab", "Egypt", "Arabic pop", None),
    ("Mohamed Mounir", "Egypt", "Arabic pop", None),
    ("Tamer Hosny", "Egypt", "Arabic pop", None),
    ("Sherine", "Egypt", "Arabic pop", None),
    ("Angham", "Egypt", "Arabic pop", None),
    ("Mohamed Hamaki", "Egypt", "Arabic pop", None),
    ("Umm Kulthum", "Egypt", "Arabic classical", "أم كلثوم"),
    ("Abdel Halim Hafez", "Egypt", "Arabic classical", None),
    ("Mohammed Abdel Wahab", "Egypt", "Arabic classical", None),
    ("Mohammed Abdu", "Saudi Arabia", "Khaleeji", None),
    ("Talal Maddah", "Saudi Arabia", "Khaleeji", None),
    ("Abdul Majeed Abdullah", "Saudi Arabia", "Khaleeji", None),
    ("Rashed Al Majed", "Kuwait", "Khaleeji", None),
    ("Majid Al Muhandis", "Kuwait", "Khaleeji", None),
    ("Hussain Al Jassmi", "United Arab Emirates", "Khaleeji", None),
    ("Ahlam", "United Arab Emirates", "Khaleeji", "أحلام"),
    ("Balqees", "United Arab Emirates", "Khaleeji", None),
    ("Kadim Al Sahir", "Iraq", "Iraqi", "كاظم الساهر"),
    ("Rida Al Abdullah", "Iraq", "Iraqi", None),
    ("Ali Al Essa", "Iraq", "Iraqi", None),
    ("Fairuz", "Lebanon", "Levantine", "فيروز"),
    ("Wael Kfoury", "Lebanon", "Levantine", None),
    ("Elissa", "Lebanon", "Levantine", None),
    ("Ragheb Alama", "Lebanon", "Levantine", None),
    ("Nancy Ajram", "Lebanon", "Levantine", None),
    ("Majida El Roumi", "Lebanon", "Levantine", None),
    ("Georges Wassouf", "Syria", "Levantine", None),
    ("Assala Nasri", "Syria", "Levantine", None),
    ("Sabah Fakhri", "Syria", "Arabic classical", "صباح فخري"),
    ("Saad Lamjarred", "Morocco", "Moroccan pop", None),
    ("Cheb Khaled", "Algeria", "Rai", "الشاب خالد"),
    ("Cheb Mami", "Algeria", "Rai", None),
    ("Warda", "Algeria", "Arabic classical", "وردة الجزائرية"),
    ("Latifa", "Tunisia", "Tunisian pop", None),
    ("Saber Rebaï", "Tunisia", "Tunisian pop", None),
    ("Abu Bakr Salem", "Yemen", "Yemeni", None),
]


def mb_get(url: str, retries: int = 5) -> dict | None:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": UA, "Accept": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001 - transient 5xx/429, retry
            time.sleep(2 + attempt * 2)
    return None


#: MusicBrainz areas that count as Arabic-music regions for this project.
ARAB_AREAS = {
    "Egypt", "Lebanon", "Syria", "Iraq", "Saudi Arabia", "Kuwait", "United Arab Emirates",
    "Qatar", "Bahrain", "Oman", "Yemen", "Jordan", "Palestine", "Morocco", "Algeria",
    "Tunisia", "Libya", "Sudan", "Mauritania", "Comoros", "Djibouti", "Somalia",
}


def resolve_artist(name: str, search: str | None = None) -> dict | None:
    """Resolve a seed name to a MusicBrainz artist, preferring an Arabic area.

    MusicBrainz relevance is the primary signal, but a candidate is only accepted
    outright when its area is an Arabic region; otherwise a very high relevance
    score is required. This rejects namesakes such as the Swedish act "Elissa".
    """
    query = urllib.parse.quote_plus(search or name)
    payload = mb_get(f"{MB}/artist?query={query}&fmt=json&limit=6")
    time.sleep(PAUSE)
    if not payload:
        return None

    def area_of(candidate: dict) -> str:
        return ((candidate.get("area") or {}).get("name") or "").strip()

    def similarity(candidate: dict) -> float:
        cand_name = candidate.get("name", "")
        return max(
            SequenceMatcher(None, norm(name), norm(cand_name)).ratio(),
            SequenceMatcher(None, norm(search or name), norm(cand_name)).ratio(),
        )

    candidates = [c for c in payload.get("artists", []) if int(c.get("score", 0)) >= 88]
    if not candidates:
        return None

    arabic = [c for c in candidates if area_of(c) in ARAB_AREAS]
    pool = arabic or [c for c in candidates if int(c.get("score", 0)) >= 97]
    if not pool:
        return None
    return max(pool, key=lambda c: (int(c.get("score", 0)), similarity(c)))


def fetch_releases(artist_mbid: str) -> list[dict]:
    url = (
        f"{MB}/release?artist={artist_mbid}"
        "&inc=recordings+artist-credits&limit=%d&fmt=json" % RELEASE_LIMIT
    )
    payload = mb_get(url)
    time.sleep(PAUSE)
    return (payload or {}).get("releases", [])


def artist_credit(credit: list[dict]) -> str:
    return "".join(part.get("name", "") for part in credit) or ""


def is_seed_track(credited: str, allowed: set[str]) -> bool:
    """Only keep tracks actually credited to the seed artist (not compilations)."""
    key = norm(credited)
    if not key:
        return False
    return any(name and name in key for name in allowed if len(name) >= 3)


def main() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    rows: list[dict] = []
    resolved: list[dict] = []
    failed: list[str] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for index, (name, region, genre, search) in enumerate(SEED, start=1):
        print(f"[{index}/{len(SEED)}] {name} …", end=" ", flush=True)
        artist = resolve_artist(name, search)
        if not artist:
            failed.append(name)
            print("not resolved")
            continue

        allowed = {norm(name), norm(search or ""), norm(artist.get("name", ""))}
        releases = fetch_releases(artist["id"])
        before = len(rows)
        for release in releases:
            language = release.get("language") or ""
            country = release.get("country") or ""
            for medium in release.get("media", []):
                for track in medium.get("tracks", []):
                    title = (track.get("title") or "").strip()
                    if not title:
                        continue
                    credited = artist_credit(
                        (track.get("artist-credit") or release.get("artist-credit") or [])
                    ) or artist.get("name", name)
                    if not is_seed_track(credited, allowed):
                        continue
                    rows.append(
                        {
                            "title": title,
                            "artist": credited,
                            "artist_latin": name,
                            "album": (release.get("title") or "").strip(),
                            "release_date": release.get("date") or "",
                            "language": language,
                            "country": country,
                            "region": region,
                            "genre": genre,
                            "region_source": "curated (project editorial label)",
                            "genre_source": "curated (project editorial label)",
                            "mb_artist_id": artist["id"],
                            "mb_recording_id": track.get("recording", {}).get("id", ""),
                            "mb_release_id": release.get("id", ""),
                            "source": "MusicBrainz",
                            "source_url": f"https://musicbrainz.org/release/{release.get('id', '')}",
                            "retrieved_at": now,
                        }
                    )
        added = len(rows) - before
        resolved.append(
            {
                "seed_name": name,
                "mb_name": artist.get("name"),
                "mb_id": artist["id"],
                "area": (artist.get("area") or {}).get("name"),
                "releases": len(releases),
                "tracks": added,
            }
        )
        print(f"{added} tracks from {len(releases)} releases")

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No data collected — aborting without writing files.")
        return 1

    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["mb_recording_id", "title", "artist"], keep="first")
    frame = frame.drop_duplicates(subset=["title", "artist"], keep="first")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "track_id", ["arb%06d" % i for i in range(1, len(frame) + 1)])
    frame.to_csv(CSV_PATH, index=False, encoding="utf-8")

    provenance = {
        "source": "MusicBrainz",
        "source_url": "https://musicbrainz.org/",
        "licence": "CC0 1.0 Universal (public domain dedication)",
        "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": "Metadata from MusicBrainz (CC0). Region and genre are "
        "editorial labels chosen by this project.",
        "retrieved_at": now,
        "user_agent": UA,
        "seed_artists": len(SEED),
        "resolved_artists": len(resolved),
        "failed_artists": failed,
        "rows_before_dedup": before_dedup,
        "tracks": int(len(frame)),
        "dedup_removed": int(before_dedup - len(frame)),
        "language_available": int((frame["language"].fillna("") != "").sum()),
        "regions": sorted(frame["region"].unique().tolist()),
        "curated_genres": sorted(frame["genre"].unique().tolist()),
        "columns": list(frame.columns),
        "artist_details": resolved,
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"tracks: {len(frame)} (dedup removed {before_dedup - len(frame)}) | "
          f"artists: {frame['artist'].nunique()} | regions: {frame['region'].nunique()} | "
          f"genres: {frame['genre'].nunique()} | with language: {provenance['language_available']}")
    print(f"failed: {failed or 'none'}")
    print(f"written: {CSV_PATH.name}, {PROVENANCE_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
