"""Phase 12.3 tests: Arabic/Yemeni preview resolution fix and cache policy.

These exercise the real resolver logic (``previews._http_json`` payloads go through
``artist_matches`` / ``title_similarity``), not a mocked-out matcher.
"""

import pandas as pd
import pytest

import app
import arabic_catalogue as ac
import previews

requires_catalogue = pytest.mark.skipif(
    not ac.DEFAULT_CATALOGUE_PATH.is_file(), reason="Arabic catalogue not built"
)
requires_aliases = pytest.mark.skipif(not ac.ALIASES_PATH.is_file(), reason="aliases not built")


@pytest.fixture(autouse=True)
def clear_caches():
    previews.find_preview.cache_clear()
    app.cached_arabic_preview.clear()
    app.cached_aliases.clear()
    yield
    previews.find_preview.cache_clear()
    app.cached_arabic_preview.clear()
    app.cached_aliases.clear()


def payload(track_name, artist_name, preview="https://audio/x.m4a"):
    return {
        "resultCount": 1,
        "results": [
            {
                "trackName": track_name,
                "artistName": artist_name,
                "previewUrl": preview,
                "trackViewUrl": "https://music.apple.com/x",
                "artworkUrl100": "https://img/100x100.jpg",
            }
        ],
    }


# --------------------------------------------------------------------------- #
# 1. Recommendation rows must resolve through the complete catalogue row
# --------------------------------------------------------------------------- #


class FakeCatalogue:
    def __init__(self, rows):
        self.rows = {row["track_id"]: row for row in rows}

    def get(self, track_id):
        if track_id not in self.rows:
            raise KeyError(f"unknown Arabic track_id: {track_id!r}")
        return self.rows[track_id]


def test_recommendation_rows_resolve_with_full_identity(monkeypatch):
    """The fix: a recommendation row alone cannot match; the full row can."""
    row = {
        "track_id": "arb001521",
        "title": "Hadeeth Al Rouh",
        "artist": "أم كلثوم",
        "artist_latin": "Umm Kulthum",
        "mb_artist_id": "",
    }
    results = pd.DataFrame(
        [{"rank": 1, "track_id": "arb001521", "title": "Hadeeth Al Rouh", "artist": "أم كلثوم",
          "album": "", "genre": "Arabic classical", "region": "Egypt",
          "metadata_score": 1.0, "explanation": "same artist"}]
    )
    verified = previews.Preview(
        url="https://audio/x.m4a", provider=previews.PROVIDER,
        external_url="https://music.apple.com/x", artwork_url="https://img/x.jpg",
        matched_title="Hadeeth Al Rouh", matched_artist="Umm Kulthum",
    )
    calls = []

    def fake_find(title, artist):
        calls.append((title, artist))
        return verified if artist == "Umm Kulthum" else None

    monkeypatch.setattr(app, "find_preview", fake_find)
    truncated = app.arabic_preview(
        {"title": results.iloc[0]["title"], "artist": results.iloc[0]["artist"]}
    )
    full = app.arabic_previews_for(results, FakeCatalogue([row]))

    assert truncated is None
    assert full["arb001521"] is verified
    assert ("Hadeeth Al Rouh", "Umm Kulthum") in calls


def test_missing_catalogue_row_is_skipped(monkeypatch):
    results = pd.DataFrame([{"rank": 1, "track_id": "gone", "title": "x", "artist": "y"}])

    monkeypatch.setattr(app, "find_preview", lambda t, a: pytest.fail("must not be called"))
    catalogue = FakeCatalogue([])

    assert app.arabic_previews_for(results, catalogue) == {}


@requires_catalogue
def test_real_recommendation_uses_full_row(monkeypatch):
    catalogue = ac.load_arabic_catalogue()
    reference = catalogue.search("Hadeeth Al Rouh").iloc[0]
    results = catalogue.recommend(str(reference["track_id"]), limit=3)
    assert "artist_latin" not in results.columns          # the defect that caused the bug

    verified = previews.Preview(
        url="https://audio/x.m4a", provider=previews.PROVIDER,
        external_url="https://music.apple.com/x", artwork_url="https://img/x.jpg",
        matched_title="t", matched_artist="Umm Kulthum",
    )

    def fake_find(title, artist):
        return verified if artist == "Umm Kulthum" else None

    monkeypatch.setattr(app, "find_preview", fake_find)
    resolved = app.arabic_previews_for(results, catalogue)
    assert all(value is verified for value in resolved.values())
    assert len(resolved) == len(results)


# --------------------------------------------------------------------------- #
# 2. Transliteration / alias behaviour through the real matcher
# --------------------------------------------------------------------------- #


def test_latin_artist_name_is_required_for_latin_catalogue_entry(monkeypatch):
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: payload("Hadeeth Al Rouh", "Umm Kulthum"))
    previews.find_preview.cache_clear()
    assert previews.find_preview("Hadeeth Al Rouh", "أم كلثوم") is None
    previews.find_preview.cache_clear()
    assert previews.find_preview("Hadeeth Al Rouh", "Umm Kulthum") is not None


@requires_aliases
def test_fairuz_alias_carries_the_apple_spelling():
    aliases = ac.load_artist_aliases()
    catalogue = ac.load_arabic_catalogue()
    fairuz = catalogue.tracks[catalogue.tracks["artist_latin"] == "Fairuz"].iloc[0]
    alias_values = aliases.get(fairuz["mb_artist_id"], ())
    assert "Fairouz" in alias_values

    pairs = app.preview_query_candidates(
        "أرجعي يا ألف ليلة", "فيروز", "Fairuz", tuple(alias_values)
    )
    assert any(artist == "Fairouz" for _, artist in pairs)


def test_fairouz_alias_verifies_through_the_matcher(monkeypatch):
    monkeypatch.setattr(previews, "_http_json",
                        lambda url, timeout=0: payload("Argeay Ya Alf Laila", "Fairouz"))
    previews.find_preview.cache_clear()
    assert previews.find_preview("Arjii ya alf leila", "Fairouz") is not None


# --------------------------------------------------------------------------- #
# 3. Honest rejections
# --------------------------------------------------------------------------- #


def test_arabic_title_without_apple_match_is_rejected(monkeypatch):
    monkeypatch.setattr(previews, "_http_json",
                        lambda url, timeout=0: payload("Allem Alby", "Amr Diab"))
    previews.find_preview.cache_clear()
    assert previews.find_preview("توبة", "Amr Diab") is None


def test_tribute_and_unrelated_artists_are_rejected(monkeypatch):
    for name in ("Umm Kulthum Tribute Band", "Some Unrelated Artist"):
        monkeypatch.setattr(previews, "_http_json",
                            lambda url, timeout=0, n=name: payload("Hadeeth Al Rouh", n))
        previews.find_preview.cache_clear()
        assert previews.find_preview("Hadeeth Al Rouh", "Umm Kulthum") is None


def test_wrong_title_is_rejected(monkeypatch):
    monkeypatch.setattr(previews, "_http_json",
                        lambda url, timeout=0: payload("Completely Different Song", "Umm Kulthum"))
    previews.find_preview.cache_clear()
    assert previews.find_preview("Hadeeth Al Rouh", "Umm Kulthum") is None


def test_missing_preview_url_is_rejected(monkeypatch):
    monkeypatch.setattr(previews, "_http_json",
                        lambda url, timeout=0: payload("Hadeeth Al Rouh", "Umm Kulthum", preview=""))
    previews.find_preview.cache_clear()
    assert previews.find_preview("Hadeeth Al Rouh", "Umm Kulthum") is None


@requires_catalogue
def test_yemeni_arabic_titles_do_not_match_latin_recordings(monkeypatch):
    catalogue = ac.load_arabic_catalogue()
    yemeni = catalogue.tracks[catalogue.tracks["region"] == "Yemen"]
    assert not yemeni.empty
    row = yemeni.iloc[0]

    monkeypatch.setattr(previews, "_http_json",
                        lambda url, timeout=0: payload("Some Other Yemeni Song", "Unrelated"))
    previews.find_preview.cache_clear()
    assert previews.find_preview(str(row["title"]), str(row["artist"])) is None


# --------------------------------------------------------------------------- #
# 4. Cache policy: successes cached, temporary failures retried
# --------------------------------------------------------------------------- #


def test_confirmed_preview_is_cached(monkeypatch):
    verified = previews.Preview(
        url="https://audio/x.m4a", provider=previews.PROVIDER,
        external_url="https://music.apple.com/x", artwork_url="https://img/x.jpg",
        matched_title="T", matched_artist="A",
    )
    calls = []
    monkeypatch.setattr(app, "find_preview", lambda t, a: calls.append((t, a)) or verified)

    assert app.cached_arabic_preview("T", "A", "A") is verified
    assert app.cached_arabic_preview("T", "A", "A") is verified
    assert len(calls) == 1


def test_negative_result_is_short_lived_and_retried(monkeypatch):
    state = {"fail": True, "calls": 0}
    verified = previews.Preview(
        url="https://audio/x.m4a", provider=previews.PROVIDER,
        external_url="https://music.apple.com/x", artwork_url="https://img/x.jpg",
        matched_title="T", matched_artist="A",
    )

    def fake_find(title, artist):
        state["calls"] += 1
        return None if state["fail"] else verified

    monkeypatch.setattr(app, "find_preview", fake_find)
    monkeypatch.setattr(previews.find_preview, "cache_clear", lambda: None)

    assert app.cached_arabic_preview("T", "A", "A") is None
    assert app.cached_arabic_preview("T", "A", "A") is None     # within cooldown: no retry
    assert state["calls"] == 1

    key = app._preview_cache_key("T", "A", "A", ())
    app._PREVIEW_FAILED_AT[key] -= app.NEGATIVE_PREVIEW_TTL_SECONDS + 1
    state["fail"] = False

    assert app.cached_arabic_preview("T", "A", "A") is verified  # retried after cooldown
    assert state["calls"] == 2
    assert app.cached_arabic_preview("T", "A", "A") is verified  # success now cached
    assert state["calls"] == 2


def test_cache_clear_resets_both_maps():
    app._PREVIEW_SUCCESS_CACHE[("k",)] = object()
    app._PREVIEW_FAILED_AT[("k",)] = 1.0
    app.cached_arabic_preview.clear()
    assert app._PREVIEW_SUCCESS_CACHE == {}
    assert app._PREVIEW_FAILED_AT == {}


# --------------------------------------------------------------------------- #
# 5. International behaviour unchanged
# --------------------------------------------------------------------------- #


def test_international_still_uses_resolve_many(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "resolve_many", lambda items: calls.append(list(items)) or {})
    app.cached_previews.clear()
    assert app.cached_previews((("id", "Unholy", "Sam Smith"),)) == {}
    assert calls == [[("id", "Unholy", "Sam Smith")]]
