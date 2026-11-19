"""Tests for the preview resolver (previews.py).

No network is used: the HTTP layer is monkeypatched, so these tests verify the
*matching rules* — that a preview is only returned for a verifiable artist and
title match, and that failures degrade to ``None``.
"""

import pytest

import previews


@pytest.fixture(autouse=True)
def clear_cache():
    """The lookup is lru_cached; clear it between tests."""
    previews.find_preview.cache_clear()
    yield
    previews.find_preview.cache_clear()


def _payload(track_name, artist_name, preview_url="https://audio.example/x.m4a", artwork="https://img/100x100bb.jpg"):
    return {
        "resultCount": 1,
        "results": [
            {
                "trackName": track_name,
                "artistName": artist_name,
                "previewUrl": preview_url,
                "trackViewUrl": "https://music.apple.com/album/1?i=2",
                "artworkUrl100": artwork,
            }
        ],
    }


# --------------------------------------------------------------------------- #
# Normalisation / matching primitives
# --------------------------------------------------------------------------- #


def test_normalise_strips_punctuation_and_case():
    assert previews._normalise("  Rocket-Man (Live!) ") == "rocket man live"


def test_primary_artist_takes_first_of_semicolon_list():
    assert previews._primary_artist("Elton John;Kiki Dee") == "Elton John"


def test_title_similarity_exact_and_variants():
    assert previews.title_similarity("Rocket Man", "Rocket Man") == 1.0
    assert previews.title_similarity(
        "Rocket Man", "Rocket Man (I Think It's Going to Be a Long Long Time)"
    ) >= 0.9
    assert previews.title_similarity("Rocket Man", "Bohemian Rhapsody") < 0.5


def test_artist_matches():
    assert previews.artist_matches("Elton John", "Elton John")
    assert previews.artist_matches("Elton John", "Sir Elton John")
    assert not previews.artist_matches("Elton John", "Elton John Tribute Band")


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #


def test_pick_candidate_accepts_verified_match():
    payload = _payload("Rocket Man", "Elton John")
    picked = previews._pick_candidate(payload["results"], "Rocket Man", "Elton John")
    assert picked is not None
    assert picked["previewUrl"].startswith("https://audio.example")


def test_pick_candidate_rejects_wrong_artist():
    payload = _payload("Rocket Man", "Some Cover Band")
    assert previews._pick_candidate(payload["results"], "Rocket Man", "Elton John") is None


def test_pick_candidate_rejects_wrong_title():
    payload = _payload("Completely Different Song", "Elton John")
    assert previews._pick_candidate(payload["results"], "Rocket Man", "Elton John") is None


def test_pick_candidate_rejects_missing_preview_url():
    payload = _payload("Rocket Man", "Elton John", preview_url="")
    assert previews._pick_candidate(payload["results"], "Rocket Man", "Elton John") is None


def test_pick_candidate_prefers_best_title_match():
    results = [
        {"trackName": "Rocket Man (Live)", "artistName": "Elton John", "previewUrl": "https://a/1"},
        {"trackName": "Rocket Man", "artistName": "Elton John", "previewUrl": "https://a/2"},
    ]
    picked = previews._pick_candidate(results, "Rocket Man", "Elton John")
    assert picked["previewUrl"] == "https://a/2"


# --------------------------------------------------------------------------- #
# find_preview behaviour
# --------------------------------------------------------------------------- #


def test_find_preview_returns_verified_preview(monkeypatch):
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: _payload("Rocket Man", "Elton John"))
    preview = previews.find_preview("Rocket Man", "Elton John")

    assert preview is not None
    assert preview.provider == previews.PROVIDER
    assert preview.url.startswith("https://audio.example")
    assert preview.artwork_url == "https://img/300x300bb.jpg"
    assert preview.external_url.startswith("https://music.apple.com")


def test_find_preview_returns_none_on_network_failure(monkeypatch):
    def boom(url, timeout=0):
        raise OSError("network down")

    monkeypatch.setattr(previews, "_http_json", boom)
    assert previews.find_preview("Anything", "Anyone") is None


def test_find_preview_returns_none_for_empty_input():
    assert previews.find_preview("", "Artist") is None
    assert previews.find_preview("Title", "") is None


def test_find_preview_returns_none_when_no_honest_match(monkeypatch):
    monkeypatch.setattr(previews, "_http_json", lambda url, timeout=0: _payload("Other", "Other"))
    assert previews.find_preview("Rocket Man", "Elton John") is None


def test_find_preview_is_cached(monkeypatch):
    calls = {"n": 0}

    def counted(url, timeout=0):
        calls["n"] += 1
        return _payload("Rocket Man", "Elton John")

    monkeypatch.setattr(previews, "_http_json", counted)
    previews.find_preview("Rocket Man", "Elton John")
    previews.find_preview("Rocket Man", "Elton John")
    assert calls["n"] == 1


def test_search_url_is_an_external_search_link():
    url = previews.search_url("Rocket Man", "Elton John")
    assert url.startswith("https://music.apple.com/search?term=")
    assert "Elton+John" in url and "Rocket+Man" in url


def test_resolve_many_keeps_keys_and_isolates_failures(monkeypatch):
    good = previews.Preview(
        url="https://audio/x.m4a",
        provider=previews.PROVIDER,
        external_url="https://music.apple.com/x",
        artwork_url="https://img/x.jpg",
        matched_title="Good",
        matched_artist="Artist",
    )

    def fake_find(title, artists):
        if title == "Boom":
            raise RuntimeError("unexpected")
        return good if title == "Good" else None

    monkeypatch.setattr(previews, "find_preview", fake_find)
    resolved = previews.resolve_many([("a", "Good", "Artist"), ("b", "Boom", "Artist"), ("c", "Other", "Artist")])

    assert resolved["a"] is good
    assert resolved["b"] is None      # exception swallowed
    assert resolved["c"] is None      # no match
