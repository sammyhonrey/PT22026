"""Data loading, validation, cleaning, and feature preparation.

Pipeline implemented here (Phase 4, tasks 4.1-4.6):

    load_raw_data()          -> raw DataFrame (CSV untouched on disk)
    drop_junk_rows()         -> removes rows with no usable metadata
    zero_coded_mask()        -> flags rows whose analysis values are absent
    build_genre_sets()       -> per-track multi-label genre associations
    build_catalogue()        -> one row per track_id, first occurrence wins
    build_numeric_block()    -> standardised, row-normalised audio features
    build_genre_matrix()     -> multi-label one-hot, row-normalised
    assemble_feature_matrix()-> weighted sparse matrix for cosine similarity
    build_feature_matrix()   -> the whole pipeline in one call

Design decisions and their evidence are documented in
`docs/preprocessing-design.md`; dataset facts in `docs/dataset.md`.

The raw CSV is never modified: every step returns a new object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, normalize

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATA_FILENAME = "dataset.csv"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / DATA_FILENAME

DOWNLOAD_HINT = (
    "Download the dataset first (see docs/dataset.md, section 3):\n"
    "  curl.exe -L -o data/spotify-tracks-dataset.zip "
    '"https://www.kaggle.com/api/v1/datasets/download/maharshipandya/-spotify-tracks-dataset"\n'
    "  Expand-Archive data/spotify-tracks-dataset.zip -DestinationPath data"
)

ID_COLUMN = "track_id"
GENRE_COLUMN = "track_genre"

#: Used for display only. Decision 4: duration is NOT part of the similarity vector.
DURATION_COLUMN = "duration_ms"

#: Columns needed to display a track; a row with none of them is unusable.
DISPLAY_COLUMNS = ("track_name", "artists", "album_name")

#: Approved similarity features (Decision 4 removed duration_ms).
NUMERIC_FEATURES: tuple[str, ...] = (
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "tempo",
)

#: Features whose simultaneous 0.0 was verified to be absent measurements.
ZERO_CODED_FEATURES: tuple[str, ...] = (
    "tempo",
    "danceability",
    "speechiness",
    "valence",
)

#: Approved initial block weights (2026-11-18, after the C3 sensitivity analysis).
#: 0.90 numeric / 0.10 genre is a documented design choice, not a proven optimum:
#: no relevance labels exist, so no weight can be shown to be objectively best.
DEFAULT_NUMERIC_WEIGHT = 0.9
DEFAULT_GENRE_WEIGHT = 0.1


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PipelineStats:
    """Counts recorded while building the catalogue. Reported, never hidden."""

    rows_loaded: int
    junk_rows_dropped: int
    duplicate_rows_dropped: int
    zero_coded_tracks_excluded: int
    sleep_tracks_before: int
    sleep_tracks_after: int
    tracks_in_catalogue: int


@dataclass(frozen=True)
class FeatureMatrix:
    """Everything the recommender needs, plus the evidence of how it was built."""

    catalogue: pd.DataFrame          # one row per track_id, index = track_id
    matrix: sparse.csr_matrix        # weighted, row-normalised feature matrix
    numeric_matrix: np.ndarray       # dense (n_tracks, 9): sklearn.normalize on dense input
    genre_matrix: sparse.csr_matrix  # sparse (n_tracks, 114)
    genre_vocabulary: tuple[str, ...]
    scaler: StandardScaler
    numeric_weight: float
    genre_weight: float
    stats: PipelineStats

    @property
    def n_tracks(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]

    def row_index(self, track_id: str) -> int:
        """Position of a track in the matrix (raises KeyError if unknown)."""
        positions = np.flatnonzero(self.catalogue.index == track_id)
        if positions.size == 0:
            raise KeyError(f"track_id not in catalogue: {track_id!r}")
        return int(positions[0])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Spotify Tracks CSV.

    Parameters
    ----------
    path:
        Optional CSV path; defaults to ``data/dataset.csv`` next to this module.

    Returns
    -------
    pandas.DataFrame
        The data as stored, except that the first CSV column — a redundant
        integer index (verified: exactly ``0..n-1``) — becomes the index.

    Raises
    ------
    FileNotFoundError
        If the CSV is absent, with the expected path and download instructions.

    Notes
    -----
    Loads only; it does not clean, de-duplicate or transform anything.
    """
    csv_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}\n{DOWNLOAD_HINT}")
    df = pd.read_csv(csv_path, index_col=0)
    df.index.name = None
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def _require_columns(df: pd.DataFrame) -> None:
    required = [ID_COLUMN, GENRE_COLUMN, DURATION_COLUMN, *DISPLAY_COLUMNS, *NUMERIC_FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def drop_junk_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows that cannot be used or displayed.

    A row is junk when it has no display metadata at all, or when a value the
    pipeline depends on (id, genre, any approved feature) is missing.

    Returns
    -------
    (clean_df, n_dropped)
        ``clean_df`` is a new DataFrame; the input is not modified.
    """
    _require_columns(df)

    no_metadata = df[list(DISPLAY_COLUMNS)].isna().all(axis=1)
    missing_critical = df[[ID_COLUMN, GENRE_COLUMN, *NUMERIC_FEATURES]].isna().any(axis=1)
    mask = no_metadata | missing_critical

    return df.loc[~mask].copy(), int(mask.sum())


def zero_coded_mask(df: pd.DataFrame) -> pd.Series:
    """Flag rows where every zero-coded feature is exactly ``0.0``.

    Verified in Phase 3 (docs/preprocessing-design.md §1b): in all 157 such rows
    ``tempo``, ``danceability``, ``speechiness`` and ``valence`` are simultaneously
    exactly zero, which is absent analysis data rather than quiet music.
    """
    mask = pd.Series(True, index=df.index)
    for column in ZERO_CODED_FEATURES:
        mask &= df[column] == 0
    return mask


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #


def build_genre_sets(df: pd.DataFrame) -> pd.Series:
    """Map ``track_id`` -> sorted tuple of every genre it appears under.

    Preserves multi-label associations instead of picking one genre.
    """
    grouped = df.groupby(ID_COLUMN)[GENRE_COLUMN].agg(
        lambda values: tuple(sorted(set(values)))
    )
    return grouped


def build_catalogue(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, PipelineStats]:
    """Build one row per ``track_id`` with multi-label genres attached.

    De-duplication keeps the **first occurrence in file order** (Decision 2).
    Rows flagged by :func:`zero_coded_mask` are removed from the catalogue
    (Decision 1); the raw DataFrame is untouched.

    Returns
    -------
    (catalogue, stats)
        ``catalogue`` is indexed by ``track_id`` and also keeps ``track_id`` as a
        column, so it can be used for both lookups and display.
    """
    _require_columns(df)

    rows_loaded = len(df)

    clean, junk_dropped = drop_junk_rows(df)

    genres = build_genre_sets(clean)

    deduped = clean.drop_duplicates(subset=[ID_COLUMN], keep="first").copy()
    duplicates_dropped = len(clean) - len(deduped)

    deduped = deduped.set_index(ID_COLUMN, drop=False)
    deduped["genre_set"] = genres.reindex(deduped.index)

    flagged = zero_coded_mask(deduped)
    zero_coded_excluded = int(flagged.sum())

    # Count sleep tracks by genre *membership*, not by the single genre label on
    # the surviving row: 3 sleep-labelled rows were the first-seen row of a track
    # that also appears under another genre, so label-counting would report 997.
    has_sleep = deduped["genre_set"].map(lambda genres: "sleep" in genres)
    sleep_before = int(has_sleep.sum())
    catalogue = deduped.loc[~flagged].copy()
    sleep_after = int(catalogue["genre_set"].map(lambda genres: "sleep" in genres).sum())

    if catalogue.index.has_duplicates:
        raise AssertionError("catalogue index must contain one row per track_id")
    if catalogue["genre_set"].isna().any():
        raise AssertionError("every catalogue row must have at least one genre")

    stats = PipelineStats(
        rows_loaded=rows_loaded,
        junk_rows_dropped=junk_dropped,
        duplicate_rows_dropped=duplicates_dropped,
        zero_coded_tracks_excluded=zero_coded_excluded,
        sleep_tracks_before=sleep_before,
        sleep_tracks_after=sleep_after,
        tracks_in_catalogue=len(catalogue),
    )
    return catalogue, stats


# --------------------------------------------------------------------------- #
# Feature blocks
# --------------------------------------------------------------------------- #


def _row_normalise(matrix) -> np.ndarray | sparse.csr_matrix:
    """L2-normalise every row; all-zero rows stay zero.

    The return type follows the input: ``sklearn.preprocessing.normalize``
    returns a dense ndarray for dense input and a CSR matrix for sparse input.
    """
    return normalize(matrix, norm="l2", axis=1, copy=True)


def build_numeric_block(
    catalogue: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Standardise the approved numeric features and give each row unit norm.

    Returns a **dense** ndarray of shape ``(n_tracks, 9)`` — the standardiser
    produces dense output and stays dense through row normalisation.

    The scaler is fitted once on the catalogue and returned so that the app and
    the evaluation path share exactly one fit.
    """
    values = catalogue[list(NUMERIC_FEATURES)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("numeric features contain NaN/inf after cleaning")

    if scaler is None:
        scaler = StandardScaler().fit(values)
    standardised = scaler.transform(values)

    return _row_normalise(standardised), scaler


def build_genre_matrix(
    catalogue: pd.DataFrame,
    vocabulary: tuple[str, ...] | None = None,
) -> tuple[sparse.csr_matrix, tuple[str, ...]]:
    """Multi-label one-hot genre block, row-normalised to unit length.

    A track tagged ``('country', 'rock')`` gets a 1 in both columns. After row
    normalisation each genre contributes ``1/sqrt(k)`` for a track with ``k``
    genres, so a multi-genre track is not automatically "similar to everything".
    """
    genre_lists = [tuple(g) for g in catalogue["genre_set"]]

    if vocabulary is None:
        vocabulary = tuple(sorted({genre for genres in genre_lists for genre in genres}))

    encoder = MultiLabelBinarizer(classes=list(vocabulary), sparse_output=True)
    indicators = encoder.fit_transform(genre_lists)

    return _row_normalise(sparse.csr_matrix(indicators)), tuple(vocabulary)


def assemble_feature_matrix(
    numeric_matrix: np.ndarray,
    genre_matrix: sparse.csr_matrix,
    numeric_weight: float = DEFAULT_NUMERIC_WEIGHT,
    genre_weight: float = DEFAULT_GENRE_WEIGHT,
) -> sparse.csr_matrix:
    """Concatenate the blocks with square-root weights.

    With both blocks row-normalised, ``cosine(X)[i, j]`` equals
    ``α * S_numeric + β * S_genre``. The square roots are applied inside the
    matrix so the weights do not silently change the row norms.
    """
    if numeric_weight < 0 or genre_weight < 0:
        raise ValueError("block weights must be non-negative")
    if not np.isclose(numeric_weight + genre_weight, 1.0):
        raise ValueError(
            f"block weights must sum to 1.0, got {numeric_weight} + {genre_weight}"
        )
    if numeric_matrix.shape[0] != genre_matrix.shape[0]:
        raise ValueError("numeric and genre blocks must have the same row count")

    weighted = sparse.hstack(
        [
            numeric_matrix * np.sqrt(numeric_weight),
            genre_matrix * np.sqrt(genre_weight),
        ],
        format="csr",
    )
    return weighted


def build_feature_matrix(
    df: pd.DataFrame | None = None,
    numeric_weight: float = DEFAULT_NUMERIC_WEIGHT,
    genre_weight: float = DEFAULT_GENRE_WEIGHT,
    scaler: StandardScaler | None = None,
    genre_vocabulary: tuple[str, ...] | None = None,
) -> FeatureMatrix:
    """Run the whole preprocessing pipeline.

    Parameters
    ----------
    df:
        Raw DataFrame; loaded from ``data/dataset.csv`` when omitted.
    numeric_weight, genre_weight:
        Block weights (must sum to 1). Decision 3: 0.7 / 0.3 initially, to be
        tested at Checkpoint C3.
    scaler, genre_vocabulary:
        Reuse a previous fit (used by the app and by tests); fitted if omitted.

    Returns
    -------
    FeatureMatrix
        Weighted matrix, catalogue, blocks, vocabulary, scaler and statistics.
    """
    if df is None:
        df = load_raw_data()

    catalogue, stats = build_catalogue(df)
    numeric_block, fitted_scaler = build_numeric_block(catalogue, scaler=scaler)
    genre_block, vocabulary = build_genre_matrix(catalogue, vocabulary=genre_vocabulary)
    matrix = assemble_feature_matrix(
        numeric_block, genre_block, numeric_weight, genre_weight
    )

    return FeatureMatrix(
        catalogue=catalogue,
        matrix=matrix,
        numeric_matrix=numeric_block,
        genre_matrix=genre_block,
        genre_vocabulary=vocabulary,
        scaler=fitted_scaler,
        numeric_weight=numeric_weight,
        genre_weight=genre_weight,
        stats=stats,
    )
