"""Phase 7 hardening: integration between preprocessing and the recommender,
plus error handling and edge cases that the unit tests do not isolate.
"""

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
    row.update(track_id=track_id, track_name=title, artists=artist, track_genre=genre)
    row.update(overrides)
    return row


@pytest.fixture()
def df():
    return pd.DataFrame(
        [
            _row("ref", "Reference Song", "Ref Artist", "rock"),
            # zero-coded row: must be excluded from the catalogue
            _row("zero", "Silence", "Quiet Artist", "sleep",
                 tempo=0.0, danceability=0.0, speechiness=0.0, valence=0.0),
            # junk row: no metadata at all
            _row("junk", None, None, "k-pop", album_name=None, duration_ms=0),
            # same recording under two ids -> one result after suppression
            _row("dup1", "Same Recording", "Same Artist", "pop"),
            _row("dup2", "Same Recording", "Same Artist", "pop"),
            _row("other", "Different Song", "Other Artist", "jazz", energy=0.2),
        ]
    )


@pytest.fixture()
def fm(df):
    return pp.build_feature_matrix(df)


# --------------------------------------------------------------------------- #
# Preprocessing -> recommender integration
# --------------------------------------------------------------------------- #


def test_zero_coded_track_is_not_recommendable(fm):
    engine = rec.Recommender(fm)
    assert "zero" not in set(fm.catalogue["track_id"])
    with pytest.raises(rec.UnknownTrackError):
        engine.recommend("zero")


def test_junk_row_is_not_recommendable(fm):
    engine = rec.Recommender(fm)
    assert "junk" not in set(fm.catalogue["track_id"])
    with pytest.raises(rec.UnknownTrackError):
        engine.recommend("junk")


def test_cross_id_duplicate_recordings_are_suppressed_end_to_end(fm):
    engine = rec.Recommender(fm)
    ids = set(engine.recommend("ref", k=10)["track_id"])
    assert len({"dup1", "dup2"} & ids) == 1


def test_matrix_rows_align_with_catalogue(fm):
    engine = rec.Recommender(fm)
    for track_id in fm.catalogue["track_id"].tolist():
        assert engine.track_row(track_id) == fm.row_index(track_id)


def test_all_returned_tracks_come_from_the_catalogue(fm):
    engine = rec.Recommender(fm)
    catalogue_ids = set(fm.catalogue["track_id"])
    results = engine.recommend("ref", k=10)
    assert set(results["track_id"]).issubset(catalogue_ids)


def test_reusing_scaler_and_vocabulary_reproduces_the_matrix(df, fm):
    reused = pp.build_feature_matrix(
        df, scaler=fm.scaler, genre_vocabulary=fm.genre_vocabulary
    )
    assert reused.matrix.shape == fm.matrix.shape
    assert (reused.matrix != fm.matrix).nnz == 0
    assert reused.catalogue.index.equals(fm.catalogue.index)


def test_one_track_catalogue_yields_empty_recommendations():
    single = pd.DataFrame([_row("solo", "Solo", "Solo Artist", "rock")])
    engine = rec.Recommender(pp.build_feature_matrix(single))
    results = engine.recommend("solo", k=5)
    assert results.empty
    assert list(results.columns) == rec.RESULT_COLUMNS


def test_k_one_returns_the_single_best_candidate(fm):
    engine = rec.Recommender(fm)
    results = engine.recommend("ref", k=1)
    assert len(results) == 1
    assert int(results.iloc[0]["rank"]) == 1


def test_unmatched_artist_preference_preserves_order(fm):
    engine = rec.Recommender(fm)
    baseline = engine.recommend("ref", k=10)
    adjusted = engine.recommend(
        "ref", k=10, preferences=rec.Preferences.build(artists=["Nobody At All"])
    )
    assert adjusted["track_id"].tolist() == baseline["track_id"].tolist()
    assert (adjusted["preference_score"] == 0).all()
    # scores scale by the 0.80 base weight, order is unchanged
    assert np.allclose(adjusted["score"], 0.80 * baseline["score"])


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_missing_required_column_raises_value_error(df):
    broken = df.drop(columns=["tempo"])
    with pytest.raises(ValueError) as excinfo:
        pp.build_feature_matrix(broken)
    assert "tempo" in str(excinfo.value)


def test_empty_dataframe_is_rejected():
    with pytest.raises(ValueError):
        pp.build_feature_matrix(pd.DataFrame())


def test_malformed_csv_file_raises_actionable_error(tmp_path):
    csv = tmp_path / "broken.csv"
    pd.DataFrame([{"track_id": "x", "artists": "y"}]).to_csv(csv)
    loaded = pp.load_raw_data(csv)
    with pytest.raises(ValueError) as excinfo:
        pp.build_feature_matrix(loaded)
    assert "missing required columns" in str(excinfo.value)


def test_engine_rejects_invalid_track_id_types(fm):
    engine = rec.Recommender(fm)
    for bad in (None, "", "   ", 7, ["ref"]):
        with pytest.raises(rec.InvalidRequestError):
            engine.recommend(bad)


def test_engine_rejects_out_of_range_k(fm):
    engine = rec.Recommender(fm)
    for bad in (0, -3, rec.MAX_TOP_K + 1, 3.5, True):
        with pytest.raises(rec.InvalidRequestError):
            engine.recommend("ref", k=bad)
