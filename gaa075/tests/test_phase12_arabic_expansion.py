"""Phase 12 tests: artist aliases, Yemeni expansion, preservation guarantees."""

import json

import pandas as pd
import pytest

import app
import arabic_catalogue as ac
import previews

requires_aliases = pytest.mark.skipif(not ac.ALIASES_PATH.is_file(), reason="aliases not built")
requires_yemeni = pytest.mark.skipif(
    not (ac.DATA_DIR / "arabic_catalogue_yemeni.csv").is_file(), reason="Yemeni file not built"
)


@pytest.fixture(autouse=True)
def clear_caches():
    previews.find_preview.cache_clear()
    app.cached_arabic_preview.clear()
    app.cached_aliases.clear()
    yield
    previews.find_preview.cache_clear()
    app.cached_arabic_preview.clear()
    app.cached_aliases.clear()


# --------------------------------------------------------------------------- #
# Aliases
# --------------------------------------------------------------------------- #


@requires_aliases
def test_aliases_file_is_cc0_and_records_source():
    payload = json.loads(ac.ALIASES_PATH.read_text(encoding="utf-8"))
    assert "CC0" in payload["licence"]
    assert payload["source"].startswith("MusicBrainz")
    assert payload["artists"]


@requires_aliases
def test_alias_candidates_are_added_and_deduplicated():
    base = app.preview_query_candidates("Makane", "Khaled", "Cheb Khaled")
    with_alias = app.preview_query_candidates("Makane", "Khaled", "Cheb Khaled", ("Cheb Khaled", "الشاب خالد"))
    assert len(with_alias) >= len(base)
    assert len(with_alias) == len(set(with_alias))
    assert all(len(pair) == 2 for pair in with_alias)


def test_candidate_pairs_are_capped():
    aliases = tuple(f"Alias {i}" for i in range(20))
    pairs = app.preview_query_candidates("Taybat", "Hussain Al Jassmi", "", aliases)
    assert len(pairs) <= app._MAX_QUERY_PAIRS


def test_alias_only_spelling_can_verify(monkeypatch):
    """An alias is just another spelling: it must still pass find_preview."""
    verified = previews.Preview(
        url="https://audio/x.m4a",
        provider=previews.PROVIDER,
        external_url="https://music.apple.com/x",
        artwork_url="https://img/x.jpg",
        matched_title="Makane Alah",
        matched_artist="Khaled",
    )

    def fake_find(title, artist):
        return verified if artist == "Khaled" else None

    monkeypatch.setattr(app, "find_preview", fake_find)
    result = app.cached_arabic_preview("Makane", "الشاب خالد", "Cheb Khaled", ("Khaled",))
    assert result is verified


def test_alias_cannot_bypass_artist_verification(monkeypatch):
    payload = {
        "results": [
            {"trackName": "Makane", "artistName": "Some Other Artist",
             "previewUrl": "https://audio/x.m4a", "trackViewUrl": "u",
             "artworkUrl100": "https://img/100x100.jpg"}
        ]
    }
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: payload)
    assert app.cached_arabic_preview("Makane", "الشاب خالد", "Cheb Khaled", ("Khaled",)) is None


# --------------------------------------------------------------------------- #
# Yemeni expansion
# --------------------------------------------------------------------------- #


@requires_yemeni
def test_yemeni_tracks_are_merged_without_losing_base():
    base = ac.load_arabic_catalogue(ac.DEFAULT_CATALOGUE_PATH).tracks
    merged = ac.load_arabic_catalogue().tracks

    assert len(merged) > len(base)                       # additive, nothing lost
    assert set(merged["track_id"]).issuperset(set(base["track_id"]))

    yemen = merged[merged["region"] == "Yemen"]
    assert len(yemen) > 200
    assert yemen["artist"].nunique() >= 10
    assert {g for g in yemen["genre"] if str(g).startswith("Yemeni")}


@requires_yemeni
def test_yemeni_provenance_and_classification():
    payload = json.loads((ac.DATA_DIR / "arabic_catalogue_yemeni_provenance.json").read_text(encoding="utf-8"))
    assert "CC0" in payload["licence"]
    assert payload["tracks"] > 0
    assert "Yemeni Sanaani" in payload["traditions"]

    yemen = ac.load_arabic_catalogue().tracks
    yemen = yemen[yemen["region"] == "Yemen"]
    assert (yemen["source"] == "MusicBrainz").all()
    assert (yemen["region_source"].str.startswith("curated")).all()
    assert (yemen["genre_source"].str.startswith("curated")).all()
    assert yemen["source_url"].str.startswith("https://musicbrainz.org/").all()


@requires_yemeni
def test_merged_catalogue_has_no_duplicate_tracks():
    merged = ac.load_arabic_catalogue().tracks
    assert not merged.duplicated(subset=["title_key", "artist_key"]).any()


@requires_yemeni
def test_explicit_path_loads_only_that_file():
    base_only = ac.load_arabic_catalogue(ac.DEFAULT_CATALOGUE_PATH).tracks
    merged = ac.load_arabic_catalogue().tracks
    assert len(base_only) < len(merged)
    assert (base_only["region"] == "Yemen").sum() < (merged["region"] == "Yemen").sum()


# --------------------------------------------------------------------------- #
# International engine unaffected
# --------------------------------------------------------------------------- #


def test_international_resolution_path_is_still_resolve_many(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "resolve_many", lambda items: calls.append(list(items)) or {})
    app.cached_previews.clear()
    assert app.cached_previews((("id", "Unholy", "Sam Smith;Kim Petras"),)) == {}
    assert calls == [[("id", "Unholy", "Sam Smith;Kim Petras")]]


def test_arabic_rows_without_preview_still_get_fallback(monkeypatch):
    """No verified match -> None, so the UI keeps 'Listen in Apple Music'."""
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: {"results": []})
    assert app.cached_arabic_preview("غير موجود", "فنان", "Artist", ("Alias",)) is None
