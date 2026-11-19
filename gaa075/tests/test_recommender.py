"""Tests for the Phase 5 recommendation engine and the duplicate-suppression rule."""

import numpy as np
import pandas as pd
import pytest

import preprocessing as pp
import recommender as rec

BASE = {
    "album_name": "Test Album",
    "explicit": False,
    "popularity": 50,
    "duration_ms": 200_000,
    "key": 5,
    "mode": 1,
    "loudness": -8.0,
    "time_signature": 4,
    "danceability": 0.50,
    "energy": 0.60,
    "speechiness": 0.05,
    "acousticness": 0.20,
    "instrumentalness": 0.01,
    "liveness": 0.20,
    "valence": 0.70,
    "tempo": 120.0,
}


def _row(track_id, title, artist, genre, **overrides):
    row = dict(BASE)
    row.update(
        track_id=track_id,
        track_name=title,
        artists=artist,
        track_genre=genre,
    )
    row.update(overrides)
    return row


@pytest.fixture()
def df():
    """Eleven rows covering every duplicate-suppression case."""
    return pd.DataFrame(
        [
            _row("id01", "Reference", "Ref Artist", "rock"),
            # identical recording, different genre label -> suppressed, higher score kept
            _row("id02", "Dup Song", "Dup Artist", "rock", duration_ms=180_000),
            _row("id03", "Dup Song", "Dup Artist", "pop", duration_ms=180_000),
            # same title/artist, different audio -> both variants kept
            _row("id04", "Var Song", "Var Artist", "pop", energy=0.95, duration_ms=210_000),
            _row("id05", "Var Song", "Var Artist", "pop", energy=0.15, duration_ms=210_000),
            # same title/artist and audio, different duration -> both kept
            _row("id06", "Dur Song", "Dur Artist", "jazz", duration_ms=150_000),
            _row("id07", "Dur Song", "Dur Artist", "jazz", duration_ms=151_000),
            # same title, different artist -> not suppressed
            _row("id08", "Shared Title", "Artist One", "blues"),
            _row("id09", "Shared Title", "Artist Two", "blues"),
            # identical features, different songs -> scores tie, ordering by track_id
            _row("id10", "Tie One", "Tie Artist A", "folk", tempo=100.0),
            _row("id11", "Tie Two", "Tie Artist B", "folk", tempo=100.0),
        ]
    )


@pytest.fixture()
def fm(df):
    return pp.build_feature_matrix(df)


@pytest.fixture()
def engine(fm):
    return rec.Recommender(fm)


# --------------------------------------------------------------------------- #
# Core behaviour
# --------------------------------------------------------------------------- #


def test_reference_is_never_recommended(engine):
    results = engine.recommend("id01", k=10)
    assert "id01" not in set(results["track_id"])
    assert results["score"].max() <= 1.0


def test_result_columns_and_rank(engine):
    results = engine.recommend("id01", k=5)
    assert list(results.columns) == rec.RESULT_COLUMNS
    assert results["rank"].tolist() == list(range(1, len(results) + 1))
    assert results["score"].is_monotonic_decreasing


def test_scores_decompose_into_components(engine):
    results = engine.recommend("id01", k=10)
    expected = engine.numeric_weight * results["numeric_score"] + engine.genre_weight * results["genre_score"]
    assert np.allclose(results["score"], expected)


def test_default_weights_are_90_10(engine):
    assert engine.numeric_weight == pytest.approx(0.90)
    assert engine.genre_weight == pytest.approx(0.10)


def test_custom_weights_are_accepted(fm):
    engine = rec.Recommender(fm, numeric_weight=0.5, genre_weight=0.5)
    assert engine.numeric_weight == 0.5


# --------------------------------------------------------------------------- #
# Duplicate-recording suppression
# --------------------------------------------------------------------------- #


def test_identical_recording_is_returned_once(engine):
    results = engine.recommend("id01", k=20)
    ids = set(results["track_id"])
    assert ("id02" in ids) != ("id03" in ids), "exactly one of the identical pair may appear"


def test_highest_scoring_duplicate_is_kept(engine):
    scored = engine.scores("id01").set_index("track_id")
    pair = scored.loc[["id02", "id03"], "score"]
    expected = pair.idxmax()
    ids = set(engine.recommend("id01", k=20)["track_id"])
    assert expected in ids
    assert (pair.drop(expected).index[0]) not in ids


def test_suppression_keeps_catalogue_rows(engine, fm):
    assert engine.n_tracks == fm.n_tracks
    assert {"id02", "id03"}.issubset(set(fm.catalogue["track_id"]))


def test_different_audio_features_keep_both_variants(engine):
    ids = set(engine.recommend("id01", k=20)["track_id"])
    assert {"id04", "id05"}.issubset(ids)


def test_different_duration_keeps_both_variants(engine):
    ids = set(engine.recommend("id01", k=20)["track_id"])
    assert {"id06", "id07"}.issubset(ids)


def test_different_artists_are_not_suppressed(engine):
    ids = set(engine.recommend("id01", k=20)["track_id"])
    assert {"id08", "id09"}.issubset(ids)


def test_suppression_can_be_disabled(fm):
    engine = rec.Recommender(fm, suppress_duplicate_recordings=False)
    ids = set(engine.recommend("id01", k=20)["track_id"])
    assert {"id02", "id03"}.issubset(ids)


def test_suppression_is_case_and_whitespace_insensitive(fm):
    df = pd.DataFrame(
        [
            _row("id01", "Reference", "Ref Artist", "rock"),
            _row("id02", "  Mixed Case  ", "Some Artist", "pop"),
            _row("id03", "mixed case", "some artist", "pop"),
        ]
    )
    engine = rec.Recommender(pp.build_feature_matrix(df))
    ids = set(engine.recommend("id01", k=10)["track_id"])
    assert len({"id02", "id03"} & ids) == 1


def test_duplicate_of_the_reference_recording_is_not_returned(fm):
    """A different track_id holding the same recording must not appear as #1."""
    df = pd.DataFrame(
        [
            _row("id01", "Same Song", "Same Artist", "rock"),
            _row("id02", "Same Song", "Same Artist", "rock"),  # same recording, other id
            _row("id03", "Different", "Other Artist", "rock", energy=0.2),
        ]
    )
    engine = rec.Recommender(pp.build_feature_matrix(df))
    ids = set(engine.recommend("id01", k=5)["track_id"])
    assert "id02" not in ids
    assert ids == {"id03"}


# --------------------------------------------------------------------------- #
# Determinism and limits
# --------------------------------------------------------------------------- #


def test_ordering_is_deterministic(engine):
    first = engine.recommend("id01", k=10)
    second = engine.recommend("id01", k=10)
    pd.testing.assert_frame_equal(first, second)


def test_ties_break_on_track_id(engine):
    """id10 and id11 have identical features, so the smaller id must come first."""
    results = engine.recommend("id01", k=20)
    positions = {tid: pos for pos, tid in enumerate(results["track_id"])}
    assert positions["id10"] < positions["id11"]


@pytest.mark.parametrize("k", [1, 5])
def test_k_is_respected(engine, k):
    assert len(engine.recommend("id01", k=k)) == k


def test_k_larger_than_available_returns_all_candidates(engine):
    """11 rows − 1 reference − 1 suppressed duplicate = 9 distinct candidates."""
    results = engine.recommend("id01", k=10)
    assert len(results) == 9
    assert results["rank"].tolist() == list(range(1, 10))


def test_empty_result_frame_when_no_candidates():
    df = pd.DataFrame([_row("solo", "Solo", "Solo Artist", "rock")])
    engine = rec.Recommender(pp.build_feature_matrix(df))
    results = engine.recommend("solo", k=5)
    assert results.empty
    assert list(results.columns) == rec.RESULT_COLUMNS


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def test_unknown_track_raises(engine):
    with pytest.raises(rec.UnknownTrackError):
        engine.recommend("nope")


@pytest.mark.parametrize("bad_id", [None, "", "   ", 42])
def test_invalid_track_id_raises(engine, bad_id):
    with pytest.raises(rec.InvalidRequestError):
        engine.recommend(bad_id)


@pytest.mark.parametrize("bad_k", [0, -1, rec.MAX_TOP_K + 1, 2.5, True])
def test_invalid_k_raises(engine, bad_k):
    with pytest.raises(rec.InvalidRequestError):
        engine.recommend("id01", k=bad_k)


@pytest.mark.parametrize("weights", [(0.7, 0.7), (1.1, -0.1), (-0.2, 1.2)])
def test_invalid_weights_raise(fm, weights):
    with pytest.raises(rec.InvalidRequestError):
        rec.Recommender(fm, numeric_weight=weights[0], genre_weight=weights[1])


# --------------------------------------------------------------------------- #
# Integration against the real dataset (skipped when absent)
# --------------------------------------------------------------------------- #


requires_dataset = pytest.mark.skipif(
    not pp.DEFAULT_DATA_PATH.is_file(),
    reason="data/dataset.csv not present",
)


# --------------------------------------------------------------------------- #
# User preferences (task 5.4)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def pref_df():
    """Reference is jazz so that pop/rock candidates score no genre similarity.

    id02 (rock) and id09 (pop) have identical audio, so their base scores tie and
    ``track_id`` decides. A pop preference must therefore flip the order.
    """
    return pd.DataFrame(
        [
            _row("id01", "Reference", "Ref Artist", "jazz"),
            _row("id02", "Rock Song", "Rock Artist", "rock"),
            _row("id09", "Pop Song", "Pop Artist", "pop"),
            # same track id twice -> genre set (pop, rock), multi-artist credit
            _row("id03", "Multi Song", "Pop Artist;Guest", "pop"),
            _row("id03", "Multi Song", "Pop Artist;Guest", "rock"),
            _row("id04", "Far Song", "Far Artist", "blues", energy=0.05, tempo=60.0),
        ]
    )


@pytest.fixture()
def pref_engine(pref_df):
    return rec.Recommender(pp.build_feature_matrix(pref_df))


def test_no_preferences_preserves_ranking_and_scores(engine):
    without = engine.recommend("id01", k=10)
    empty = engine.recommend("id01", k=10, preferences=rec.Preferences())
    pd.testing.assert_frame_equal(without, empty)
    assert without["score"].equals(without["base_score"])
    assert (without["preference_score"] == 0).all()
    assert without.attrs["preferences_applied"] is False


def test_genre_preference_changes_the_ranking(pref_engine):
    plain = pref_engine.recommend("id01", k=5)
    boosted = pref_engine.recommend(
        "id01", k=5, preferences=rec.Preferences.build(genres=["pop"])
    )
    assert plain.iloc[0]["track_id"] == "id02"      # rock wins on tie-break alone
    assert boosted.iloc[0]["track_id"] != "id02"
    assert boosted.attrs["preferences_applied"] is True


def test_artist_only_preference_works(pref_engine):
    results = pref_engine.recommend(
        "id01", k=5, preferences=rec.Preferences.build(artists=["Pop Artist"])
    )
    top = results.iloc[0]
    assert "pop artist" in top["artists"].casefold()
    assert top["preference_score"] == pytest.approx(1.0)


def test_combined_preferences_follow_the_formula(pref_engine):
    preferences = rec.Preferences.build(genres=["pop"], artists=["Pop Artist"])
    results = pref_engine.recommend("id01", k=5, preferences=preferences)

    expected = 0.80 * results["base_score"] + 0.20 * results["preference_score"]
    assert np.allclose(results["score"], expected)
    for _, row in results.iterrows():
        assert row["score"] == pytest.approx(
            0.80 * row["base_score"] + 0.20 * row["preference_score"], abs=1e-12
        )


def test_genre_component_is_the_share_of_requested_genres(pref_engine):
    preferences = rec.Preferences.build(genres=["pop", "rock", "jazz"])
    results = pref_engine.recommend("id01", k=5, preferences=preferences)
    multi = results.loc[results["track_id"] == "id03"].iloc[0]
    assert multi["preference_score"] == pytest.approx(2 / 3)


def test_artist_component_uses_any_credited_artist(pref_engine):
    preferences = rec.Preferences.build(artists=["guest"])
    results = pref_engine.recommend("id01", k=5, preferences=preferences)
    multi = results.loc[results["track_id"] == "id03"].iloc[0]
    assert multi["preference_score"] == pytest.approx(1.0)


def test_unknown_genre_is_rejected(pref_engine):
    with pytest.raises(rec.InvalidRequestError):
        pref_engine.recommend(
            "id01", preferences=rec.Preferences.build(genres=["not-a-genre"])
        )


def test_wrong_preferences_type_is_rejected(engine):
    with pytest.raises(rec.InvalidRequestError):
        engine.recommend("id01", preferences=["rock"])


def test_preferences_are_normalised():
    prefs = rec.Preferences.build(genres=["  RoCk  ", "rock"], artists=["  ARTIST "])
    assert prefs.genres == ("rock",)
    assert prefs.artists == ("artist",)
    assert not prefs.is_empty()
    assert rec.Preferences().is_empty()


def test_preferences_do_not_break_suppression(engine):
    results = engine.recommend("id01", k=20, preferences=rec.Preferences.build(genres=["rock"]))
    ids = set(results["track_id"])
    assert ("id02" in ids) != ("id03" in ids)


def test_preferences_never_recommend_the_reference(pref_engine):
    results = pref_engine.recommend(
        "id01", k=5, preferences=rec.Preferences.build(genres=["pop"], artists=["Pop Artist"])
    )
    assert "id01" not in set(results["track_id"])


# --------------------------------------------------------------------------- #
# Explanations (task 5.5)
# --------------------------------------------------------------------------- #


def test_explanation_mentions_shared_genre(engine):
    results = engine.recommend("id01", k=10)
    row = results.loc[results["track_id"] == "id02"].iloc[0]
    assert "shared genre" in row["explanation"]
    assert "rock" in row["explanation"]


def test_explanation_mentions_audio_traits(engine):
    results = engine.recommend("id01", k=10)
    assert all("closest audio traits" in text for text in results["explanation"])


def test_explanation_mentions_preferred_artist(engine):
    results = engine.recommend(
        "id01", k=20, preferences=rec.Preferences.build(artists=["Dup Artist"])
    )
    row = results.loc[results["track_id"] == "id02"].iloc[0]
    assert "matches preferred artist: dup artist" in row["explanation"]


def test_explanation_mentions_preferred_genre(pref_engine):
    results = pref_engine.recommend("id01", k=5, preferences=rec.Preferences.build(genres=["pop"]))
    assert results["explanation"].str.contains("matches preferred genre: pop").any()


def test_explanations_are_deterministic(engine):
    first = engine.recommend("id01", k=10)
    second = engine.recommend("id01", k=10)
    assert first["explanation"].tolist() == second["explanation"].tolist()


def test_every_explanation_is_non_empty(engine):
    results = engine.recommend("id01", k=10)
    assert results["explanation"].str.len().gt(0).all()


# --------------------------------------------------------------------------- #
# Score warning (task 5.6)
# --------------------------------------------------------------------------- #


def test_warning_constant_exists_and_is_attached(engine):
    assert "not probabilities" in rec.SCORE_INTERPRETATION_WARNING
    results = engine.recommend("id01", k=5)
    assert results.attrs["score_warning"] == rec.SCORE_INTERPRETATION_WARNING


def test_warning_is_attached_to_empty_results():
    df = pd.DataFrame([_row("solo", "Solo", "Solo Artist", "rock")])
    results = rec.Recommender(pp.build_feature_matrix(df)).recommend("solo", k=5)
    assert results.empty
    assert results.attrs["score_warning"] == rec.SCORE_INTERPRETATION_WARNING


# --------------------------------------------------------------------------- #
# Integration against the real dataset (skipped when absent)
# --------------------------------------------------------------------------- #


@requires_dataset
def test_real_data_recommendations():
    engine = rec.Recommender(pp.build_feature_matrix())
    rng = np.random.default_rng(3)
    sample = rng.choice(engine.n_tracks, size=5, replace=False)

    for row in sample:
        track_id = engine._track_ids[row]
        results = engine.recommend(track_id, k=10)

        assert len(results) == 10
        assert results["track_id"].is_unique
        assert track_id not in set(results["track_id"])
        assert results["score"].is_monotonic_decreasing
        assert np.isfinite(results["score"]).all()
        assert results["score"].between(0.0, 1.0).all()

        # the engine's actual guarantee: no two results share the approved
        # duplicate-recording signature (title + artist + audio + duration)
        keys = [engine._duplicate_keys[engine.track_row(t)] for t in results["track_id"]]
        assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- #
# "More by artist" list
# --------------------------------------------------------------------------- #


def test_more_by_artist_returns_only_that_artist(engine):
    result = engine.more_by_artist("id01", "Var Artist", k=5)
    assert len(result) == 2
    assert set(result["artists"]) == {"Var Artist"}
    assert list(result["rank"]) == [1, 2]
    assert result["score"].is_monotonic_decreasing


def test_more_by_artist_is_case_insensitive_and_excludes_reference(engine):
    result = engine.more_by_artist("id01", "ref artist", k=5)
    assert result.empty  # the only "Ref Artist" track is the reference itself


def test_more_by_artist_unknown_or_blank_artist_is_empty(engine):
    assert engine.more_by_artist("id01", "Nobody", k=5).empty
    assert engine.more_by_artist("id01", "   ", k=5).empty


def test_more_by_artist_does_not_change_recommend(engine):
    before = engine.recommend("id01", k=5)
    engine.more_by_artist("id01", "Var Artist", k=5)
    after = engine.recommend("id01", k=5)
    pd.testing.assert_frame_equal(before, after)
