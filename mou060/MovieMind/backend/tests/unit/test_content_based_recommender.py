"""Content-based recommender pure helpers (T8.5 acceptance)."""

from __future__ import annotations

from app.ml.recommenders.content_based_recommender import (
    compute_genre_preferences,
    score_by_preferences,
)


def test_genre_preferences_centered_mean():
    rated = [
        (["Action", "Thriller"], 5.0),
        (["Action"], 3.0),
        (["Drama"], 1.0),
    ]
    prefs = compute_genre_preferences(rated)
    # Action: (5.0 + 3.0) / 2 - 3.0 = 1.0
    assert prefs["Action"] == 1.0
    # Thriller: 5.0 - 3.0 = 2.0
    assert prefs["Thriller"] == 2.0
    # Drama: 1.0 - 3.0 = -2.0
    assert prefs["Drama"] == -2.0


def test_genre_preferences_empty():
    assert compute_genre_preferences([]) == {}


def test_score_by_preferences_sums_genres():
    prefs = {"Action": 1.0, "Thriller": 2.0, "Drama": -2.0}
    assert score_by_preferences(["Action", "Thriller"], prefs) == 3.0
    assert score_by_preferences(["Drama"], prefs) == -2.0
    # Unknown genre contributes 0; no genres contributes 0.
    assert score_by_preferences(["Comedy"], prefs) == 0.0
    assert score_by_preferences([], prefs) == 0.0
