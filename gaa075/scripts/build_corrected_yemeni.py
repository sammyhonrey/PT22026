"""Build the CORRECTED Yemeni catalogue (Phase 12.1).

Only seeds whose MusicBrainz identity is verifiable by *explicit MBID* are
included. The three misattributed seeds from Phase 12
(Fouad Al-Kibsi, Ali bin Ali Al-Ansi, Faisal Al-Radhi) have no MusicBrainz record
of their own, so they are left unresolved and their Phase 12 rows are excluded.

New output files only; the Phase 12 files are never modified.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_CSV = DATA / "arabic_catalogue_yemeni_v2.csv"
OUT_PROV = DATA / "arabic_catalogue_yemeni_v2_provenance.json"

import sys

sys.path.insert(0, str(ROOT))

from arabic_catalogue import normalize_text as catalogue_norm  # same key the loader uses

MB = "https://musicbrainz.org/ws/2"
UA = "MusicRecommendationSystem/1.0 (Phase 12.1 identity verification)"
PAUSE = 1.2
LIMIT = 100
ARABIC = re.compile(r"[\u0600-\u06FF]")
DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

#: Verified seeds: (label, arabic label, MBID, expected MB name, area, curated tradition)
VERIFIED = [
    ("Abu Bakr Salem", "أبو بكر سالم", "b0b889b4-e694-41ec-9092-e510edd51bc9",
     "ابو بكر سالم بلفقيه", "Yemen", "Yemeni traditional"),
    ("Mohammed Morshed Naji", "محمد مرشد ناجي", "a4bfc9b8-087b-4756-9c25-2cbc1d0cfc5a",
     "محمد مرشد ناجي", "Yemen", "Yemeni Sanaani"),
    ("Mohammed Al-Harithi", "محمد الحارثي", "4bd7a78f-b2de-4ba8-ae73-d532f852fc7e",
     "Muhammad Hamud Al‐Harithi", "Yemen", "Yemeni traditional"),
]

#: Seeds the audit flagged, kept explicitly unresolved (no fabricated MBID).
UNRESOLVED = [
    ("Fouad Al-Kibsi", "فؤاد الكبسي", "no MusicBrainz artist record for this name"),
    ("Ali bin Ali Al-Ansi", "علي بن علي الآنسي", "MusicBrainz only has the different artist علي بن محمد"),
    ("Faisal Al-Radhi", "فيصل الرضي", "MusicBrainz only has the different artist فيصل علوي"),
    ("Ayoub Tarish", "أيوب طارش", "resolved earlier but no releases"),
    ("Ahmed Fathi", "أحمد فتحي", "resolved earlier but no releases"),
    ("Ahmed Al-Sunaidar", "أحمد السنيدار", "resolved earlier but no releases"),
    ("Ali Al-Sumaie", "علي السميعة", "no verifiable MusicBrainz artist"),
    ("Huda Al-Yamani", "هدى اليماني", "no verifiable MusicBrainz artist"),
]


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = DIACRITICS.sub("", value).replace("\u0640", "")
    value = re.sub(r"[^\w\u0600-\u06FF]+", " ", value)
    return " ".join(value.split())


def get(url: str, retries: int = 5):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            time.sleep(2 + attempt * 2)
    return None


def credit_names(credit: list[dict]) -> list[str]:
    """Individual artist names from a MusicBrainz artist-credit array."""
    names = []
    for part in credit:
        name = (part.get("name") or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def main() -> int:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: list[dict] = []
    verified_records: list[dict] = []

    for label, arabic, mbid, expected, area, tradition in VERIFIED:
        artist = get(f"{MB}/artist/{mbid}?fmt=json")
        time.sleep(PAUSE)
        if not artist or artist.get("id") != mbid:
            print(f"  {label}: MBID not resolvable — skipped")
            continue
        actual_name = artist.get("name")
        actual_area = (artist.get("area") or {}).get("name")
        # identity gate: MBID must be the record we verified in the audit
        if actual_name != expected or actual_area != area:
            print(f"  {label}: identity gate FAILED ({actual_name!r}, {actual_area!r}) — skipped")
            continue
        print(f"  {label}: verified MBID {mbid} -> {actual_name!r} ({actual_area})")

        releases = get(f"{MB}/release?artist={mbid}&inc=recordings+artist-credits&limit={LIMIT}&fmt=json")
        time.sleep(PAUSE)
        releases = (releases or {}).get("releases", [])
        allowed = {norm(label), norm(arabic), norm(actual_name)}
        before = len(rows)

        for release in releases:
            for medium in release.get("media", []):
                for track in medium.get("tracks", []):
                    title = (track.get("title") or "").strip()
                    if not title:
                        continue
                    names = credit_names(track.get("artist-credit") or release.get("artist-credit") or [])
                    if not names:
                        names = [actual_name]
                    # keep only tracks genuinely credited to this artist
                    if not any(norm(n) in allowed or any(a in norm(n) for a in allowed if len(a) >= 3) for n in names):
                        continue
                    rows.append(
                        {
                            "title": title,
                            "artist": "; ".join(names),          # individual names preserved
                            "artist_latin": label,
                            "album": (release.get("title") or "").strip(),
                            "release_date": release.get("date") or "",
                            "language": "",
                            "country": release.get("country") or "",
                            "region": "Yemen",
                            "genre": tradition,
                            "region_source": "curated (project editorial label)",
                            "genre_source": "curated (project editorial label)",
                            "mb_artist_id": mbid,
                            "mb_recording_id": track.get("recording", {}).get("id", ""),
                            "mb_release_id": release.get("id", ""),
                            "source": "MusicBrainz",
                            "source_url": f"https://musicbrainz.org/release/{release.get('id', '')}",
                            "retrieved_at": now,
                            "identity_verification": f"MBID {mbid} verified: {actual_name} / {actual_area}",
                        }
                    )
        verified_records.append(
            {"label": label, "arabic": arabic, "mbid": mbid, "mb_name": actual_name,
             "area": actual_area, "releases": len(releases), "tracks": len(rows) - before}
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No verified rows — aborting without writing files.")
        return 1

    before_dedup = len(frame)
    frame = frame.drop_duplicates(subset=["mb_recording_id", "title", "artist"], keep="first")
    frame = frame.drop_duplicates(subset=["title", "artist"], keep="first")
    # Arabic normalisation merges spelling/diacritic variants, so dedupe on the
    # same key the catalogue class uses.
    frame["_title_key"] = frame["title"].map(catalogue_norm)
    frame["_artist_key"] = frame["artist"].map(catalogue_norm)
    frame = frame.drop_duplicates(subset=["_title_key", "_artist_key"], keep="first")
    frame = frame.drop(columns=["_title_key", "_artist_key"])
    frame = frame.reset_index(drop=True)
    frame.insert(0, "track_id", ["ymv2%05d" % i for i in range(1, len(frame) + 1)])
    frame.to_csv(OUT_CSV, index=False, encoding="utf-8")

    OUT_PROV.write_text(
        json.dumps(
            {
                "source": "MusicBrainz",
                "licence": "CC0 1.0 Universal (public domain dedication)",
                "retrieved_at": now,
                "purpose": "Corrected Yemeni catalogue (Phase 12.1): verified MBIDs only",
                "verified_seeds": verified_records,
                "unresolved_seeds": [{"label": l, "arabic": a, "reason": r} for l, a, r in UNRESOLVED],
                "rows_before_dedup": before_dedup,
                "tracks": int(len(frame)),
                "dedup_removed": int(before_dedup - len(frame)),
                "multi_artist_credit_rows": int(frame["artist"].str.contains(";").sum()),
                "regions": ["Yemen"],
                "traditions": sorted(frame["genre"].unique().tolist()),
                "region_source": "curated (project editorial label)",
                "genre_source": "curated (project editorial label)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"tracks written: {len(frame)} (dedup removed {before_dedup - len(frame)})")
    print("artists (credited):", frame["artist"].nunique(), "| seed labels:", frame["artist_latin"].nunique())
    print("multi-artist credit rows:", int(frame["artist"].str.contains(";").sum()))
    print("traditions:", frame["genre"].value_counts().to_dict())
    print("written:", OUT_CSV.name, OUT_PROV.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
