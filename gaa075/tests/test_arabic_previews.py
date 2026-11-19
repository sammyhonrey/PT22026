"""Tests for the Arabic preview query-candidate fix.

The fix only changes *which strings are sent to Apple*; verification stays in
``previews.find_preview`` (strict artist + title rules, unchanged). These tests
pin that split: candidate construction is pure and deterministic, and acceptance
is decided solely by the existing matcher.
"""

import pytest

import app
import previews


@pytest.fixture(autouse=True)
def clear_caches():
    previews.find_preview.cache_clear()
    app.cached_arabic_preview.clear()
    yield
    previews.find_preview.cache_clear()
    app.cached_arabic_preview.clear()


# --------------------------------------------------------------------------- #
# Label cleaning
# --------------------------------------------------------------------------- #


def test_clean_label_strips_annotation_tail_from_artist():
    assert app.clean_label("Hussain Al Jassmi = حسين الجسمي") == "Hussain Al Jassmi"
    assert app.clean_label("محمد منير = Mohamed Mounir") == "محمد منير"


def test_clean_label_strips_annotation_tail_from_title():
    assert app.clean_label("ميال = Mayyal") == "ميال"
    assert app.clean_label("Taybat = تعبت") == "Taybat"


def test_clean_label_removes_bracketed_gloss():
    assert app.clean_label("قمرين (Amarain)") == "قمرين"
    assert app.clean_label("Experience [Live]") == "Experience"


def test_clean_label_drops_trailing_translation_only_when_arabic_head():
    assert app.clean_label("قمرين (Amarain) - Zwei Monde") == "قمرين"
    # a Latin title keeps its dash segment, which is not a translation
    assert "Live" in app.clean_label("Love - Live")


def test_clean_label_is_whitespace_stable():
    assert app.clean_label("  Taybat =  تعبت ") == "Taybat"
    assert app.clean_label(None) == ""


# --------------------------------------------------------------------------- #
# Candidate lists
# --------------------------------------------------------------------------- #


def test_title_candidates_preserve_latin_parenthetical_as_candidate():
    candidates = app.label_candidates("قمرين (Amarain) - Zwei Monde")
    assert candidates[0] == "قمرين (Amarain) - Zwei Monde"     # original first
    assert "قمرين" in candidates
    assert "Amarain" in candidates


def test_title_candidates_extract_latin_from_annotation():
    candidates = app.label_candidates("ميال = Mayyal")
    assert "ميال" in candidates
    assert "Mayyal" in candidates


def test_artist_candidates_include_both_scripts():
    candidates = app.label_candidates("Hussain Al Jassmi = حسين الجسمي")
    assert candidates[0] == "Hussain Al Jassmi = حسين الجسمي"
    assert "Hussain Al Jassmi" in candidates
    assert "حسين الجسمي" in candidates


def test_candidates_are_deduplicated_and_capped():
    candidates = app.label_candidates("Mayyal = Mayyal = Mayyal")
    assert len(candidates) == len(set(candidates))
    assert len(candidates) <= app._MAX_TITLE_CANDIDATES


def test_query_pairs_start_with_original_then_cleaned_then_latin():
    pairs = app.preview_query_candidates(
        "ميال = Mayyal", "عمرو دياب = Amr Diab", "Amr Diab"
    )
    assert pairs[0] == ("ميال = Mayyal", "عمرو دياب = Amr Diab")
    assert ("ميال", "عمرو دياب") in pairs
    assert any(artist == "Amr Diab" for _, artist in pairs)
    # deterministic: same input, same order
    assert pairs == app.preview_query_candidates("ميال = Mayyal", "عمرو دياب = Amr Diab", "Amr Diab")


def test_query_pairs_are_unique():
    pairs = app.preview_query_candidates("Taybat", "Hussain Al Jassmi", "Hussain Al Jassmi")
    assert len(pairs) == len(set(pairs))


# --------------------------------------------------------------------------- #
# Acceptance still comes from the strict matcher
# --------------------------------------------------------------------------- #


def test_accepts_only_a_verified_preview(monkeypatch):
    """Our loop must not accept on its own: it returns whatever find_preview verifies."""
    verified = previews.Preview(
        url="https://audio/track.m4a",
        provider=previews.PROVIDER,
        external_url="https://music.apple.com/x",
        artwork_url="https://img/x.jpg",
        matched_title="Taybat",
        matched_artist="Hussain Al Jassmi",
    )
    calls = []

    def fake_find(title, artist):
        calls.append((title, artist))
        return verified if artist == "Hussain Al Jassmi" and title == "Taybat" else None

    monkeypatch.setattr(app, "find_preview", fake_find)
    result = app.cached_arabic_preview(
        "Taybat = تعبت", "Hussain Al Jassmi = حسين الجسمي", "Hussain Al Jassmi"
    )
    assert result is verified
    assert ("Taybat", "Hussain Al Jassmi") in calls
    # no duplicate attempts
    assert len(calls) == len(set(calls))


def test_incorrect_artist_never_passes(monkeypatch):
    """A result for a different artist is rejected by the matcher, not by us."""
    payload = {
        "resultCount": 1,
        "results": [
            {
                "trackName": "Taybat",
                "artistName": "Some Cover Band",
                "previewUrl": "https://audio/x.m4a",
                "trackViewUrl": "https://music.apple.com/x",
                "artworkUrl100": "https://img/100x100.jpg",
            }
        ],
    }
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: payload)
    assert app.cached_arabic_preview("Taybat", "Hussain Al Jassmi", "Hussain Al Jassmi") is None


def test_incorrect_title_never_passes(monkeypatch):
    payload = {
        "resultCount": 1,
        "results": [
            {
                "trackName": "Completely Different Song",
                "artistName": "Hussain Al Jassmi",
                "previewUrl": "https://audio/x.m4a",
                "trackViewUrl": "https://music.apple.com/x",
                "artworkUrl100": "https://img/100x100.jpg",
            }
        ],
    }
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: payload)
    assert app.cached_arabic_preview("Taybat", "Hussain Al Jassmi", "Hussain Al Jassmi") is None


def test_tribute_band_is_rejected(monkeypatch):
    payload = {
        "resultCount": 1,
        "results": [
            {
                "trackName": "Taybat",
                "artistName": "Hussain Al Jassmi Tribute Band",
                "previewUrl": "https://audio/x.m4a",
                "trackViewUrl": "https://music.apple.com/x",
                "artworkUrl100": "https://img/100x100.jpg",
            }
        ],
    }
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: payload)
    assert app.cached_arabic_preview("Taybat", "Hussain Al Jassmi", "Hussain Al Jassmi") is None


def test_missing_preview_url_is_rejected(monkeypatch):
    payload = {
        "resultCount": 1,
        "results": [
            {
                "trackName": "Taybat",
                "artistName": "Hussain Al Jassmi",
                "previewUrl": "",
                "trackViewUrl": "https://music.apple.com/x",
                "artworkUrl100": "https://img/100x100.jpg",
            }
        ],
    }
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: payload)
    assert app.cached_arabic_preview("Taybat", "Hussain Al Jassmi", "Hussain Al Jassmi") is None


def test_verified_match_after_cleaning(monkeypatch):
    """The annotated query would fail; the cleaned candidate succeeds."""
    payload = {
        "resultCount": 1,
        "results": [
            {
                "trackName": "Taybat",
                "artistName": "Hussain Al Jassmi",
                "previewUrl": "https://audio/x.m4a",
                "trackViewUrl": "https://music.apple.com/x",
                "artworkUrl100": "https://img/100x100.jpg",
            }
        ],
    }
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: payload)
    result = app.cached_arabic_preview(
        "Taybat = تعبت", "Hussain Al Jassmi = حسين الجسمي", "Hussain Al Jassmi"
    )
    assert result is not None
    assert result.matched_artist == "Hussain Al Jassmi"


def test_no_verified_match_returns_none_for_fallback(monkeypatch):
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: {"results": []})
    assert app.cached_arabic_preview("غير موجود", "فنان غير موجود", "Unknown Artist") is None


# --------------------------------------------------------------------------- #
# International path unchanged
# --------------------------------------------------------------------------- #


def test_international_resolution_still_uses_resolve_many(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "resolve_many", lambda items: calls.append(list(items)) or {})
    app.cached_previews.clear()
    result = app.cached_previews((("id1", "Unholy", "Sam Smith;Kim Petras"),))
    assert result == {}
    assert calls == [[("id1", "Unholy", "Sam Smith;Kim Petras")]]
