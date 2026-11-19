"""Additive Phase 12 data jobs (MusicBrainz, CC0).

1. Artist aliases for the Arabic seed artists  -> data/arabic_artist_aliases.json
2. Yemeni music expansion                     -> data/arabic_catalogue_yemeni.csv

Both files are new; the existing catalogue is never modified or overwritten.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = DATA / "arabic_catalogue.csv"
ALIASES = DATA / "arabic_artist_aliases.json"
YEMENI = DATA / "arabic_catalogue_yemeni.csv"
PROV = DATA / "arabic_catalogue_yemeni_provenance.json"

MB = "https://musicbrainz.org/ws/2"
UA = "MusicRecommendationSystem/1.0 (university project; educational use)"
PAUSE = 1.15
LIMIT = 100
ARABIC = re.compile(r"[\u0600-\u06FF]")
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

#: Yemeni seeds with curated tradition labels (project editorial labels).
YEMEN_SEEDS = [
    ("Abu Bakr Salem", "أبو بكر سالم", "Yemeni traditional"),
    ("Ayoub Tarish", "أيوب طارش", "Yemeni Sanaani"),
    ("Mohammed Morshed Naji", "محمد مرشد ناجي", "Yemeni Sanaani"),
    ("Fouad Al-Kibsi", "فؤاد الكبسي", "Yemeni Sanaani"),
    ("Ahmed Fathi", "أحمد فتحي", "Yemeni Adeni"),
    ("Ali bin Ali Al-Ansi", "علي بن علي الآنسي", "Yemeni Sanaani"),
    ("Ahmed Al-Sunaidar", "أحمد السنيدار", "Yemeni Adeni"),
    ("Ali Al-Sumaie", "علي السميعة", "Yemeni Hadhrami"),
    ("Mohammed Al-Harithi", "محمد الحارثي", "Yemeni modern"),
    ("Faisal Al-Radhi", "فيصل الرضي", "Yemeni modern"),
    ("Huda Al-Yamani", "هدى اليماني", "Yemeni modern"),
]


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = DIACRITICS.sub("", value).replace("\u0640", "")
    value = re.sub(r"[^\w\u0600-\u06FF]+", " ", value)
    return " ".join(value.split())


def get(url: str, retries: int = 5) -> dict | None:
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            time.sleep(2 + attempt * 2)
    return None


def resolve_yemeni(name: str, arabic: str) -> dict | None:
    for query in (arabic, name):
        payload = get(f"{MB}/artist?query={urllib.parse.quote_plus(query)}&fmt=json&limit=5")
        time.sleep(PAUSE)
        if not payload:
            continue
        for candidate in payload.get("artists", []):
            area = ((candidate.get("area") or {}).get("name") or "").strip()
            if area == "Yemen" and int(candidate.get("score", 0)) >= 85:
                return candidate
    return None


def fetch_releases(artist_id: str) -> list[dict]:
    payload = get(f"{MB}/release?artist={artist_id}&inc=recordings+artist-credits&limit={LIMIT}&fmt=json")
    time.sleep(PAUSE)
    return (payload or {}).get("releases", [])


def credit(parts: list[dict]) -> str:
    return "".join(part.get("name", "") for part in parts) or ""


def main() -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base = pd.read_csv(BASE)

    # 1) aliases -----------------------------------------------------------
    alias_map: dict[str, dict] = {}
    for _, row in base.drop_duplicates("mb_artist_id").iterrows():
        mbid = row["mb_artist_id"]
        payload = get(f"{MB}/artist/{mbid}?inc=aliases&fmt=json")
        time.sleep(PAUSE)
        if not payload:
            print(f"  aliases: {row['artist_latin']} -> failed")
            continue
        aliases = []
        for alias in payload.get("aliases", []):
            value = (alias.get("name") or "").strip()
            if value and value not in aliases and len(value) >= 3:
                aliases.append(value)
        alias_map[mbid] = {
            "name": payload.get("name"),
            "seed": row["artist_latin"],
            "aliases": aliases,
        }
        print(f"  aliases: {row['artist_latin']:<24} {len(aliases)}")

    ALIASES.write_text(
        json.dumps(
            {
                "source": "MusicBrainz artist aliases",
                "source_url": "https://musicbrainz.org/",
                "licence": "CC0 1.0 Universal (public domain dedication)",
                "retrieved_at": now,
                "artists": alias_map,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 2) Yemeni expansion --------------------------------------------------
    rows: list[dict] = []
    resolved: list[dict] = []
    for name, arabic, tradition in YEMEN_SEEDS:
        artist = resolve_yemeni(name, arabic)
        if not artist:
            print(f"  yemen: {name} -> not resolved")
            continue
        allowed = {norm(name), norm(arabic), norm(artist.get("name", ""))}
        releases = fetch_releases(artist["id"])
        before = len(rows)
        for release in releases:
            for medium in release.get("media", []):
                for track in medium.get("tracks", []):
                    title = (track.get("title") or "").strip()
                    if not title:
                        continue
                    credited = credit(track.get("artist-credit") or release.get("artist-credit") or []) or artist.get("name", name)
                    key = norm(credited)
                    if not any(a and a in key for a in allowed if len(a) >= 3):
                        continue
                    rows.append(
                        {
                            "title": title,
                            "artist": credited,
                            "artist_latin": name,
                            "album": (release.get("title") or "").strip(),
                            "release_date": release.get("date") or "",
                            "language": "",
                            "country": release.get("country") or "",
                            "region": "Yemen",
                            "genre": tradition,
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
        resolved.append({"seed": name, "mb_name": artist.get("name"), "mb_id": artist["id"],
                         "releases": len(releases), "tracks": len(rows) - before})
        print(f"  yemen: {name:<24} {len(rows) - before} tracks from {len(releases)} releases")

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No Yemeni rows collected; leaving existing files untouched.")
        return 1

    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["mb_recording_id", "title", "artist"], keep="first")
    frame = frame.drop_duplicates(subset=["title", "artist"], keep="first")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "track_id", ["yem%05d" % i for i in range(1, len(frame) + 1)])
    frame.to_csv(YEMENI, index=False, encoding="utf-8")

    PROV.write_text(
        json.dumps(
            {
                "source": "MusicBrainz",
                "licence": "CC0 1.0 Universal (public domain dedication)",
                "retrieved_at": now,
                "seed_artists": len(YEMEN_SEEDS),
                "resolved_artists": len(resolved),
                "rows_before_dedup": before_dedup,
                "tracks": int(len(frame)),
                "regions": ["Yemen"],
                "traditions": sorted(frame["genre"].unique().tolist()),
                "artist_details": resolved,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nYemeni tracks written: {len(frame)} (dedup removed {before_dedup - len(frame)})")
    print("traditions:", frame["genre"].value_counts().to_dict())
    print("artists:", frame["artist"].nunique())
    print("written:", YEMENI.name, PROV.name, ALIASES.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
