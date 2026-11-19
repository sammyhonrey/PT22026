"""Tests for the Phase 4 preprocessing pipeline.

Two layers:

* synthetic-frame tests — fast, no dataset needed, prove each rule in isolation;
* one integration test — skipped automatically when ``data/dataset.csv`` is
  absent, proves the rules against the real 114,000-row catalogue.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import preprocessing as pp

# --------------------------------------------------------------------------- #
# Synthetic fixture
# --------------------------------------------------------------------------- #

BASE = {
    "album_name": "Test Album",
    "explicit": False,
    "popularity": 50,
    "duration_ms": 200_000,
    "key": 5,
    "mode": 1,
    "loudness": -8.0,
    "time_signature": 4,
}


def _row(track_id, track_name, artists, genre, **overrides):
    row = dict(BASE)
    row.update(
        track_id=track_id,
        track_name=track_name,
        artists=artists,
        track_genre=genre,
        danceability=0.50,
        energy=0.60,
        speechiness=0.05,
        acousticness=0.20,
        instrumentalness=0.01,
        liveness=0.20,
        valence=0.70,
        tempo=120.0,
    )
    row.update(overrides)
    return row


@pytest.fixture()
def synthetic_df():
    """Six rows exercising every rule.

    A appears twice (duplicate, differing popularity and genre) -> first wins.
    C is zero-coded (tempo/danceability/speechiness/valence all 0) -> excluded.
    D is junk (no metadata at all) -> dropped.
    """
    return pd.DataFrame(
        [
            _row("A", "Song A", "Artist A", "rock", popularity=10, duration_ms=210_000),
            _row("A", "Song A", "Artist A", "pop", popularity=40, duration_ms=210_000),
            _row("B", "Song B", "Artist B", "pop", danceability=0.80, tempo=90.0, duration_ms=180_000),
            _row(
                "C", "Song C", "Artist C", "sleep",
                danceability=0.0, speechiness=0.0, valence=0.0, tempo=0.0,
                duration_ms=300_000,
            ),
            _row("D", None, None, "k-pop", duration_ms=0, album_name=None),
            _row("E", "Song E", "Artist E", "country", danceability=0.30, energy=0.20, tempo=140.0),
        ]
    )


# --------------------------------------------------------------------------- #
# Cleaning rules
# --------------------------------------------------------------------------- #


def test_junk_row_is_dropped(synthetic_df):
    clean, dropped = pp.drop_junk_rows(synthetic_df)
    assert dropped == 1
    assert "D" not in set(clean[pp.ID_COLUMN])


def test_zero_coded_mask_flags_only_the_coded_row(synthetic_df):
    mask = pp.zero_coded_mask(synthetic_df)
    assert set(synthetic_df.loc[mask, pp.ID_COLUMN]) == {"C"}
    assert int(mask.sum()) == 1


def test_raw_dataframe_is_not_modified(synthetic_df):
    """The pipeline must never mutate its input (raw data stays unchanged)."""
    before = synthetic_df.copy(deep=True)

    pp.build_feature_matrix(synthetic_df)

    pd.testing.assert_frame_equal(synthetic_df, before)


# --------------------------------------------------------------------------- #
# Catalogue: duplicates and genres
# --------------------------------------------------------------------------- #


def test_duplicate_keeps_first_occurrence(synthetic_df):
    catalogue, stats = pp.build_catalogue(synthetic_df)

    assert catalogue.index.is_unique
    assert stats.duplicate_rows_dropped == 1
    assert catalogue.loc["A", "popularity"] == 10          # first row wins
    assert catalogue.loc["A", pp.GENRE_COLUMN] == "rock"


def test_genre_associations_are_preserved(synthetic_df):
    catalogue, _ = pp.build_catalogue(synthetic_df)
    assert catalogue.loc["A", "genre_set"] == ("pop", "rock")   # both kept, sorted
    assert catalogue.loc["B", "genre_set"] == ("pop",)


def test_zero_coded_track_is_excluded_but_others_remain(synthetic_df):
    catalogue, stats = pp.build_catalogue(synthetic_df)

    assert "C" not in catalogue.index
    assert stats.zero_coded_tracks_excluded == 1
    assert set(catalogue.index) == {"A", "B", "E"}
    assert stats.tracks_in_catalogue == 3


def test_counts_are_reported_honestly(synthetic_df):
    _, stats = pp.build_catalogue(synthetic_df)
    assert stats.rows_loaded == 6
    assert stats.junk_rows_dropped == 1
    assert stats.duplicate_rows_dropped == 1
    assert stats.zero_coded_tracks_excluded == 1
    assert stats.tracks_in_catalogue == 3


# --------------------------------------------------------------------------- #
# Feature blocks: scaling, dimensions, weighting
# --------------------------------------------------------------------------- #


def test_scaler_is_fitted_on_the_catalogue(synthetic_df):
    fm = pp.build_feature_matrix(synthetic_df)
    expected = fm.catalogue[list(pp.NUMERIC_FEATURES)].to_numpy(dtype=float)
    std = expected.std(axis=0, ddof=0)

    assert np.allclose(fm.scaler.mean_, expected.mean(axis=0))

    # sklearn's rule: scale_ == std for varying features, and 1.0 for a feature
    # with zero variance (this fixture has 5 constant features with 3 tracks).
    # A tolerance is required because float arithmetic leaves ~1e-17 noise.
    varying = std > 1e-12
    assert varying.sum() > 0
    assert np.allclose(fm.scaler.scale_[varying], std[varying])
    assert np.allclose(fm.scaler.scale_[~varying], 1.0)


def test_constant_features_do_not_produce_nan(synthetic_df):
    """A zero-variance feature must be scaled, not divide by zero."""
    fm = pp.build_feature_matrix(synthetic_df)
    assert np.isfinite(fm.matrix.data).all()


def test_block_types_match_annotations(synthetic_df):
    """numeric_matrix is dense (annotated np.ndarray), genre_matrix is CSR."""
    fm = pp.build_feature_matrix(synthetic_df)

    assert isinstance(fm.numeric_matrix, np.ndarray)
    assert not sparse.issparse(fm.numeric_matrix)
    assert sparse.issparse(fm.genre_matrix)
    assert isinstance(fm.genre_matrix, sparse.csr_matrix)


def test_matrix_dimensions(synthetic_df):
    fm = pp.build_feature_matrix(synthetic_df)

    assert fm.numeric_matrix.shape == (3, len(pp.NUMERIC_FEATURES))
    assert fm.genre_matrix.shape == (3, len(fm.genre_vocabulary))
    assert fm.n_features == len(pp.NUMERIC_FEATURES) + len(fm.genre_vocabulary)
    assert fm.genre_vocabulary == ("country", "pop", "rock")


def test_duration_is_not_in_the_similarity_vector():
    assert pp.DURATION_COLUMN not in pp.NUMERIC_FEATURES
    assert len(pp.NUMERIC_FEATURES) == 9


def test_every_row_has_unit_norm(synthetic_df):
    fm = pp.build_feature_matrix(synthetic_df)
    norms = np.sqrt(fm.matrix.multiply(fm.matrix).sum(axis=1)).A1
    assert np.allclose(norms, 1.0)


def test_multi_genre_track_has_two_nonzero_genre_entries(synthetic_df):
    fm = pp.build_feature_matrix(synthetic_df)
    row = fm.row_index("A")
    assert fm.genre_matrix.getrow(row).nnz == 2
    assert np.isclose(fm.genre_matrix.getrow(row).sum(), 2 / np.sqrt(2))


def test_block_weights_change_the_matrix(synthetic_df):
    fm_70 = pp.build_feature_matrix(synthetic_df, numeric_weight=0.7, genre_weight=0.3)
    fm_50 = pp.build_feature_matrix(synthetic_df, numeric_weight=0.5, genre_weight=0.5)

    assert fm_70.matrix.shape == fm_50.matrix.shape
    assert (fm_70.matrix != fm_50.matrix).nnz > 0


def test_invalid_weights_are_rejected(synthetic_df):
    with pytest.raises(ValueError):
        pp.build_feature_matrix(synthetic_df, numeric_weight=0.7, genre_weight=0.7)
    with pytest.raises(ValueError):
        pp.build_feature_matrix(synthetic_df, numeric_weight=-0.1, genre_weight=1.1)


def test_pipeline_is_reproducible(synthetic_df):
    first = pp.build_feature_matrix(synthetic_df)
    second = pp.build_feature_matrix(synthetic_df)

    assert first.matrix.shape == second.matrix.shape
    assert (first.matrix != second.matrix).nnz == 0
    assert first.genre_vocabulary == second.genre_vocabulary
    assert np.allclose(first.scaler.mean_, second.scaler.mean_)


def test_unknown_track_lookup_raises(synthetic_df):
    fm = pp.build_feature_matrix(synthetic_df)
    with pytest.raises(KeyError):
        fm.row_index("does-not-exist")


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_loader_raises_actionable_error_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        pp.load_raw_data(tmp_path / "missing.csv")
    message = str(excinfo.value)
    assert "not found" in message.lower()
    assert "kaggle.com" in message


# --------------------------------------------------------------------------- #
# Integration test against the real dataset (skipped when absent)
# --------------------------------------------------------------------------- #


requires_dataset = pytest.mark.skipif(
    not pp.DEFAULT_DATA_PATH.is_file(),
    reason="data/dataset.csv not present; run from a checkout with the dataset",
)


@requires_dataset
def test_real_dataset_pipeline():
    fm = pp.build_feature_matrix()

    # documented cleaning counts (docs/dataset.md, docs/preprocessing-design.md)
    assert fm.stats.rows_loaded == 114_000
    assert fm.stats.junk_rows_dropped == 1
    assert fm.stats.zero_coded_tracks_excluded == 157
    assert fm.stats.sleep_tracks_before == 1_000
    assert fm.stats.sleep_tracks_after == 862

    # one row per track, nothing lost beyond the documented exclusions
    assert fm.catalogue.index.is_unique
    assert fm.n_tracks == fm.stats.tracks_in_catalogue
    assert fm.n_tracks == 114_000 - 1 - 24_259 - 157

    # dimensions: 9 numeric features + 114 genre indicators
    assert fm.numeric_matrix.shape[1] == 9
    assert fm.genre_matrix.shape[1] == 114
    assert fm.n_features == 9 + 114

    # quality of the matrix itself
    assert np.isfinite(fm.matrix.data).all()
    norms = np.sqrt(fm.matrix.multiply(fm.matrix).sum(axis=1)).A1
    assert np.allclose(norms, 1.0, atol=1e-9)
    assert fm.matrix.nnz > 0
