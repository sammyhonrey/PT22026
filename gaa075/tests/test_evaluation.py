"""Tests for the Phase 6 evaluation procedures.

Pure metric functions are tested on small arrays; the integration tests use the
real catalogue and are skipped when ``data/dataset.csv`` is absent.
"""

import numpy as np
import pandas as pd
import pytest

import evaluation as ev
import preprocessing as pp
import recommender as rec

requires_dataset = pytest.mark.skipif(
    not pp.DEFAULT_DATA_PATH.is_file(),
    reason="data/dataset.csv not present",
)


# --------------------------------------------------------------------------- #
# Pure metrics
# --------------------------------------------------------------------------- #


def test_intra_list_similarity_of_identical_rows_is_one():
    block = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert ev.intra_list_similarity(block) == pytest.approx(1.0)


def test_intra_list_similarity_of_orthogonal_rows_is_zero():
    block = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ev.intra_list_similarity(block) == pytest.approx(0.0)


def test_intra_list_similarity_with_fewer_than_two_rows_is_zero():
    assert ev.intra_list_similarity(np.array([[1.0, 0.0]])) == 0.0
    assert ev.intra_list_similarity(np.zeros((0, 2))) == 0.0


def test_genre_diversity_counts_distinct_genres():
    assert ev.genre_diversity([("rock",), ("rock", "pop")]) == 2
    assert ev.genre_diversity([]) == 0


def test_genre_consistency_full_partial_and_none():
    reference = ("rock",)
    assert ev.genre_consistency(reference, [("rock",), ("rock", "pop")]) == 1.0
    assert ev.genre_consistency(reference, [("rock",), ("jazz",)]) == 0.5
    assert ev.genre_consistency(reference, [("jazz",), ("pop",)]) == 0.0
    assert ev.genre_consistency(reference, []) == 0.0


def test_artist_diversity_counts_distinct_artists():
    assert ev.artist_diversity(["A", "A", "B", "C"]) == pytest.approx(0.75)
    assert ev.artist_diversity([]) == 0.0


def test_lookalike_repeats_counts_extra_rows_case_insensitively():
    results = pd.DataFrame(
        {
            "track_name": ["Song", "song", "Other"],
            "artists": ["Artist", "ARTIST", "Artist"],
        }
    )
    assert ev.lookalike_repeats(results) == 1
    unique = pd.DataFrame({"track_name": ["A", "B"], "artists": ["X", "Y"]})
    assert ev.lookalike_repeats(unique) == 0


# --------------------------------------------------------------------------- #
# Integration with the real catalogue
# --------------------------------------------------------------------------- #


@requires_dataset
def test_reference_selection_is_deterministic_and_spans_profiles():
    fm = pp.build_feature_matrix()
    first = ev.select_reference_tracks(fm)
    second = ev.select_reference_tracks(fm)

    assert [r.track_id for r in first] == [r.track_id for r in second]
    assert len(first) >= 10
    labels = [r.label for r in first]
    assert any(label.startswith("genre:rock") for label in labels)
    assert "multi-genre" in labels
    # multi-genre references really are multi-genre
    for reference in first:
        if reference.label == "multi-genre":
            assert len(reference.genres) >= 3


@requires_dataset
def test_evaluate_all_reports_structural_invariants():
    fm = pp.build_feature_matrix()
    engine = rec.Recommender(fm)
    references = ev.select_reference_tracks(fm)
    table = ev.evaluate_all(engine, references, k=5)

    assert len(table) == len(references)
    assert table["self_present"].sum() == 0          # reference never returned
    assert table["full_signature_repeats"].sum() == 0  # approved rule holds
    assert (table["n_results"] == 5).all()
    assert table["genre_consistency"].between(0, 1).all()
    assert table["artist_diversity"].between(0, 1).all()
    assert table["ils_numeric"].between(-1, 1).all()
    assert table["ils_genre"].between(-1, 1).all()

    summary = ev.summarise(table, engine.n_tracks)
    assert 0 < summary["catalogue_coverage"] <= 1
    assert summary["self_match_violations"] == 0


@requires_dataset
def test_preference_comparison_measures_change():
    fm = pp.build_feature_matrix()
    engine = rec.Recommender(fm)
    references = ev.select_reference_tracks(fm)[:2]
    table = ev.compare_preferences(engine, references, k=5)

    assert set(table["scenario"]) == {"aligned genre", "cross genre", "artist"}
    assert (table["changed_items"] >= 0).all()
    assert (table["shared_items"] >= 0).all()
    assert table["changed_items"].max() > 0, "some preference must move the list"


# --------------------------------------------------------------------------- #
# Honesty guard: no invented relevance metrics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["precision", "recall", "accuracy", "ndcg", "f1_score"])
def test_no_relevance_metrics_are_defined(name):
    """The dataset has no labels, so these must not exist to be misused later."""
    assert not hasattr(ev, name)
