"""Tests for the single-page Streamlit interface (app.py).

Covers the pure presentation helpers, the player payload/HTML, and the complete
single-page flow through Streamlit's own ``AppTest`` harness (preview lookups are
disabled in those runs with ``MRS_DISABLE_PREVIEWS=1``, so no network is used).
"""

from pathlib import Path

import pandas as pd
import pytest

import app
import preprocessing as pp

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

requires_dataset = pytest.mark.skipif(
    not pp.DEFAULT_DATA_PATH.is_file(),
    reason="data/dataset.csv not present",
)


@pytest.fixture()
def catalogue():
    rows = []
    for i, (title, artist, genre, popularity) in enumerate(
        [
            ("Alpha Song", "The Betas", "rock", 80),
            ("Beta Song", "The Alphas", "pop", 60),
            ("Gamma", "Solo Act", "jazz", 40),
            ("Delta Tune", "The Betas", "rock", 20),
        ]
    ):
        rows.append(
            {
                "track_id": f"id{i:02d}",
                "track_name": title,
                "artists": artist,
                "album_name": f"Album {i}",
                "genre_set": (genre,),
                "popularity": popularity,
                "duration_ms": 200_000 + i * 1000,
            }
        )
    df = pd.DataFrame(rows)
    df.index = df["track_id"]
    return df


@pytest.fixture()
def results_frame():
    return pd.DataFrame(
        [
            {
                "rank": 1,
                "track_id": "id02",
                "track_name": "Beta Song",
                "artists": "The Alphas",
                "genres": "pop",
                "score": 0.93,
                "preference_score": 0.0,
                "explanation": "shared genre: pop; closest audio traits: valence, tempo",
            },
            {
                "rank": 2,
                "track_id": "id03",
                "track_name": "Gamma",
                "artists": "Solo Act",
                "genres": "jazz",
                "score": 0.81,
                "preference_score": 0.0,
                "explanation": "no shared genre; the audio profile is the only link",
            },
        ]
    )


class FakePreview:
    url = "https://audio.example/x.m4a"
    provider = "Apple iTunes preview"
    external_url = "https://music.apple.com/x"
    artwork_url = "https://img.example/x.jpg"
    matched_title = "Beta Song"
    matched_artist = "The Alphas"


# --------------------------------------------------------------------------- #
# Search helpers
# --------------------------------------------------------------------------- #


def test_search_by_title_is_case_insensitive(catalogue):
    assert app.search_tracks(catalogue, "alpha").iloc[0]["track_name"] == "Alpha Song"


def test_search_by_artist(catalogue):
    hits = app.search_tracks(catalogue, "the betas")
    assert set(hits["track_name"]) == {"Alpha Song", "Delta Tune"}


def test_empty_query_returns_most_popular_first(catalogue):
    hits = app.search_tracks(catalogue, "")
    assert hits.iloc[0]["track_name"] == "Alpha Song"


def test_search_limit_and_no_hits(catalogue):
    assert len(app.search_tracks(catalogue, "", limit=2)) == 2
    assert app.search_tracks(catalogue, "zzz-nothing").empty


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #


def test_format_score_is_raw_cosine_never_percentage():
    assert app.format_score(0.9301) == "0.930"
    assert "%" not in app.format_score(0.93)
    assert app.format_score(float("nan")) == "—"
    assert app.format_score(None) == "—"


def test_track_label_and_genres_display():
    assert app.track_label("Song", "Artist") == "Song — Artist"
    assert app.track_label(None, None) == "Unknown track — Unknown artist"
    assert app.genres_display("rock, pop") == "rock, pop"
    assert app.genres_display(("rock", "pop")) == "rock, pop"


# --------------------------------------------------------------------------- #
# Player payload and markup
# --------------------------------------------------------------------------- #


def test_payload_marks_unavailable_previews(results_frame):
    payload = app.build_result_payload(results_frame, {})
    assert [item["preview_url"] for item in payload] == ["", ""]
    assert all(item["external_url"].startswith("https://music.apple.com/search") for item in payload)
    assert payload[0]["score"] == "0.930"


def test_payload_uses_resolved_preview(results_frame):
    payload = app.build_result_payload(results_frame, {"id02": FakePreview()})
    assert payload[0]["preview_url"] == FakePreview.url
    assert payload[0]["artwork_url"] == FakePreview.artwork_url
    assert payload[0]["external_url"] == FakePreview.external_url
    assert payload[1]["preview_url"] == ""          # second row unresolved


def test_results_html_contains_cards_and_mini_player(results_frame):
    payload = app.build_result_payload(results_frame, {"id02": FakePreview()})
    page = app.render_results_html(payload)

    assert page.count('class="card"') == 2
    assert 'id="mini"' in page
    assert 'id="mini-toggle"' in page and 'id="mini-replay"' in page and 'id="mini-next"' in page
    assert "Preview unavailable" in page          # second card has no preview
    assert 'data-idx="0"' in page                 # play button for the first card
    assert page.count("<audio") == 0 and "new Audio()" in page   # exactly one audio object
    assert 'id="np-progress"' in page and 'id="np-elapsed"' in page and 'id="np-remaining"' in page


def test_now_playing_overlay_structure(results_frame):
    page = app.render_results_html(app.build_result_payload(results_frame, {"id02": FakePreview()}))

    assert 'id="np"' in page
    assert "Now Playing" in page
    assert 'id="np-back"' in page                  # back/close control
    assert 'id="np-art"' in page                   # circular artwork element
    assert 'id="np-toggle"' in page
    assert 'id="np-next"' in page
    assert 'id="np-status"' in page
    assert 'class="np-ring"' in page               # glowing ring
    # hidden until a preview is selected
    assert 'hidden' in page.split('id="np"')[1][:80]


def test_replay_is_an_icon_not_text(results_frame):
    page = app.render_results_html(app.build_result_payload(results_frame, {"id02": FakePreview()}))

    # no visible button label says "Replay" — the control is the circular icon
    assert ">Replay<" not in page
    assert page.count('aria-label="Replay preview from the beginning"') == 2   # overlay + mini
    for chunk in page.split('aria-label="Replay preview from the beginning"')[1:]:
        assert "<svg" in chunk[:120]
    assert page.count("<svg") >= 6


def test_unavailable_track_has_no_play_button(results_frame):
    page = app.render_results_html(app.build_result_payload(results_frame, {}))
    assert page.count('class="play"') == 0          # no play buttons at all
    assert page.count('class="unavailable"') == 2   # both cards say so
    assert page.count('class="listen"') == 2        # and offer a legitimate link
    assert "Listen on Apple Music" in page


def test_artwork_only_when_verified(results_frame):
    with_art = app.render_results_html(app.build_result_payload(results_frame, {"id02": FakePreview()}))
    assert 'src="https://img.example/x.jpg"' in with_art
    assert 'class="art art-fallback"' in with_art              # second card has no art

    without_art = app.render_results_html(app.build_result_payload(results_frame, {}))
    assert '<img class="art"' not in without_art               # no card artwork invented
    assert without_art.count('class="art art-fallback"') == 2


def test_progress_uses_actual_preview_duration(results_frame):
    page = app.render_results_html(app.build_result_payload(results_frame, {}))
    assert "audio.duration" in page          # real duration, not a hardcoded length
    assert "Math.min(real, LIMIT)" in page   # capped at the 30-second preview
    assert "LIMIT = 30" in page


def test_overlay_can_close_without_stopping_audio(results_frame):
    page = app.render_results_html(app.build_result_payload(results_frame, {"id02": FakePreview()}))
    assert "function closeOverlay()" in page
    assert "np.classList.remove" in page      # closes the view
    # audio is deliberately not paused when closing
    close_block = page.split("function closeOverlay()")[1][:220]
    assert "audio.pause" not in close_block


def test_player_is_responsive(results_frame):
    page = app.render_results_html(app.build_result_payload(results_frame, {}))
    assert "@media (max-width: 640px)" in page
    assert "prefers-reduced-motion" in page


def test_dark_violet_theme_config():
    config = (Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml").read_text(
        encoding="utf-8"
    )
    assert 'base = "dark"' in config
    assert "#963BFF" in config and "#090611" in config and "#170D2B" in config


def test_results_html_escapes_untrusted_text(results_frame):
    evil = results_frame.copy()
    evil.loc[0, "track_name"] = "<script>alert(1)</script>"
    evil.loc[0, "explanation"] = "<img src=x onerror=alert(1)>"
    page = app.render_results_html(app.build_result_payload(evil, {}))
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_results_html_empty_payload():
    assert "No recommendations" in app.render_results_html([])


def test_results_height_scales_with_count():
    assert app.results_height(5) < app.results_height(10)
    assert app.results_height(0) > 0


# --------------------------------------------------------------------------- #
# Single-page flow through Streamlit's AppTest
# --------------------------------------------------------------------------- #


@requires_dataset
def test_single_page_renders_without_navigation(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MRS_DISABLE_PREVIEWS", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()

    assert not at.exception, f"app raised: {at.exception}"
    labels = [r.label for r in at.radio]
    assert "Navigation" not in labels            # no page navigation anywhere
    assert any("Arabic Music" in r.options for r in at.radio)   # discovery mode selector
    assert len(at.text_input) >= 2               # search + preferred artist
    assert len(at.selectbox) >= 1                # matching songs
    assert len(at.multiselect) >= 1              # preferred genres
    count_radio = next(r for r in at.radio if r.label == "How many recommendations?")
    options = [str(option) for option in count_radio.options]
    assert "5" in options and "10" in options


@requires_dataset
def test_full_flow_five_and_ten_recommendations(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MRS_DISABLE_PREVIEWS", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=240).run()

    at.text_input[0].set_value("Elton John").run()
    assert not at.exception
    button = next(b for b in at.button if "Find Similar Songs" in b.label)

    def count_radio():
        return next(r for r in at.radio if r.label == "How many recommendations?")

    # 5 recommendations (default)
    button.click().run()
    assert not at.exception, f"app raised: {at.exception}"
    assert len(at.session_state["results"]) == 5

    # 10 recommendations
    count_radio().set_value("10").run()
    next(b for b in at.button if "Find Similar Songs" in b.label).click().run()
    assert not at.exception
    assert len(at.session_state["results"]) == 10


@requires_dataset
def test_preferences_change_results_flow(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MRS_DISABLE_PREVIEWS", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=240).run()
    at.text_input[0].set_value("Elton John").run()

    at.multiselect[0].set_value(["piano"]).run()
    at.text_input[1].set_value("Elton John").run()
    next(b for b in at.button if "Find Similar Songs" in b.label).click().run()

    assert not at.exception
    stored = at.session_state["results_preferences"]
    assert stored.genres == ("piano",)
    assert stored.artists == ("elton john",)


@requires_dataset
def test_unavailable_previews_are_handled(monkeypatch):
    """Preview lookups disabled → results still render, cards show 'unavailable'."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MRS_DISABLE_PREVIEWS", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=240).run()
    at.text_input[0].set_value("Elton John").run()
    next(b for b in at.button if "Find Similar Songs" in b.label).click().run()

    assert not at.exception
    payload = at.session_state.get("results")
    assert payload is not None and len(payload) == 5


@requires_dataset
def test_no_search_hits_shows_warning(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MRS_DISABLE_PREVIEWS", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    at.text_input[0].set_value("zzz no such song zzz").run()

    assert not at.exception
    assert any("No song matches" in w.value for w in at.warning)


@requires_dataset
def test_results_are_hidden_after_choosing_another_song(monkeypatch):
    """Old recommendations must not stay on screen under a different reference song."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MRS_DISABLE_PREVIEWS", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=240).run()

    at.text_input[0].set_value("BTS").run()
    next(b for b in at.button if "Find Similar Songs" in b.label).click().run()
    assert any("Songs similar to your reference" in m.value for m in at.markdown)

    at.text_input[0].set_value("Coldplay").run()
    assert not at.exception
    assert not any("Songs similar to your reference" in m.value for m in at.markdown)
    assert any("Find Similar Songs" in i.value for i in at.info)


@requires_dataset
def test_preferred_artist_shows_more_by_artist_section(monkeypatch):
    """Typing an artist in preferences also lists that artist's own songs."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MRS_DISABLE_PREVIEWS", "1")
    at = AppTest.from_file(APP_PATH, default_timeout=240).run()

    at.text_input[0].set_value("Hymn for the Weekend").run()
    artist_box = next(t for t in at.text_input if t.label == "Preferred artist")
    artist_box.set_value("coldplay").run()
    next(b for b in at.button if "Find Similar Songs" in b.label).click().run()

    assert not at.exception, f"app raised: {at.exception}"
    assert any("More by Coldplay" in m.value for m in at.markdown)


def test_arabic_reference_song_has_audio_player(monkeypatch):
    """The Arabic page offers a player for the selected song when a preview is verified."""
    from streamlit.testing.v1 import AppTest

    import previews as pv

    monkeypatch.delenv("MRS_DISABLE_PREVIEWS", raising=False)
    fake = pv.Preview(
        url="https://audio.example/x.m4a",
        provider="Apple iTunes preview",
        external_url="https://music.apple.com/x",
        artwork_url="",
        matched_title="Amarain",
        matched_artist="Amr Diab",
    )
    monkeypatch.setattr(pv, "find_preview", lambda *a, **k: fake)
    at = AppTest.from_file(APP_PATH, default_timeout=240).run()
    mode = next(r for r in at.radio if "Arabic Music" in r.options)
    mode.set_value("Arabic Music").run()
    assert not at.exception, f"app raised: {at.exception}"
    assert any("Listen to your reference song" in c.value for c in at.caption)
