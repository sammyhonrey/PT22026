"""Recommendation engine: similarity, ranking, preferences and explanations.

Design (approved 2026-11-18):

    base   = alpha * S_num + beta * S_genre        alpha = 0.90, beta = 0.10
    final  = 0.80 * base + 0.20 * preference_score  (only when preferences are set)
    final  = base                                   (no preferences -> ranking unchanged)

* ``S_num``   cosine similarity of the 9 standardised audio features
* ``S_genre`` cosine similarity of the 114 multi-label genre indicators
* ``preference_score`` — the mean of the available components below, each 0..1:
    * genres: share of the *requested* genres the track belongs to
    * artists: share of the *requested* artists the track is credited to
* the reference track is always excluded, and its duplicate-recording signature is
  pre-seeded so the same recording cannot reappear under another ``track_id``
* duplicate *recordings* are suppressed per the approved signature; catalogue
  rows are never removed
* ordering is deterministic: score descending, then ``track_id`` ascending

Weights are documented design decisions, **not** proven optima: the dataset has
no relevance labels, so no weighting can be shown to be objectively better.
Similarity and preference scores are not probabilities and not percentages of a
listener liking a track — only the ordering is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from preprocessing import (
    DURATION_COLUMN,
    FeatureMatrix,
    GENRE_COLUMN,
    ID_COLUMN,
    NUMERIC_FEATURES,
)

DEFAULT_TOP_K = 10
MIN_TOP_K = 1
MAX_TOP_K = 50

NUMERIC_WEIGHT = 0.90
GENRE_WEIGHT = 0.10

#: Share of the final score that comes from content similarity once preferences
#: are supplied. The remainder (0.20) comes from the preference match.
BASE_WEIGHT_WITH_PREFERENCES = 0.80
PREFERENCE_WEIGHT = 0.20

#: Decimals used when comparing audio features for the duplicate signature.
SIGNATURE_DECIMALS = 9

#: Number of shared genres / audio traits quoted in an explanation.
EXPLANATION_GENRES = 3
EXPLANATION_TRAITS = 2

SCORE_INTERPRETATION_WARNING = (
    "Similarity scores compare track features (audio profile and genre). They are "
    "not probabilities, not percentages, and not a prediction that you will like a "
    "song; only the ranking order is meaningful."
)

RESULT_COLUMNS = [
    "rank",
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "genres",
    "duration_ms",
    "score",
    "base_score",
    "preference_score",
    "numeric_score",
    "genre_score",
    "explanation",
]


class UnknownTrackError(KeyError):
    """Raised when the requested reference track is not in the catalogue."""


class InvalidRequestError(ValueError):
    """Raised for invalid arguments (wrong type, out-of-range k, bad weights, unknown genre)."""


@dataclass(frozen=True)
class Preferences:
    """User preferences.

    Both fields are normalised (stripped, case-folded, de-duplicated, sorted) so
    that identical intent always produces identical output.

    Parameters
    ----------
    genres:
        Preferred genres, e.g. ``("rock", "pop")``.
    artists:
        Preferred artists, matched case-insensitively against the track's
        semicolon-separated artist list.
    """

    genres: tuple[str, ...] = ()
    artists: tuple[str, ...] = ()

    @classmethod
    def build(cls, genres=None, artists=None) -> "Preferences":
        """Normalise raw user input into a :class:`Preferences` value."""

        def normalise(values) -> tuple[str, ...]:
            if values is None:
                return ()
            if isinstance(values, str):
                values = [values]
            cleaned = {str(v).strip().casefold() for v in values if str(v).strip()}
            return tuple(sorted(cleaned))

        return cls(genres=normalise(genres), artists=normalise(artists))

    def is_empty(self) -> bool:
        return not self.genres and not self.artists


@dataclass(frozen=True)
class _DuplicateKey:
    """The approved duplicate-recording signature.

    Two rows are the same recording only when normalised title, normalised
    artist, the nine audio features and the exact duration all match.
    """

    title: str
    artist: str
    audio: tuple[float, ...]
    duration_ms: int


class Recommender:
    """Content-based recommender over a :class:`FeatureMatrix`.

    Parameters
    ----------
    feature_matrix:
        Output of :func:`preprocessing.build_feature_matrix`.
    numeric_weight, genre_weight:
        Content-block weights; must sum to 1. Defaults are the approved 0.90/0.10.
    suppress_duplicate_recordings:
        When True (default) only the highest-scoring row of an identical
        recording is returned. Catalogue rows are never deleted.
    """

    def __init__(
        self,
        feature_matrix: FeatureMatrix,
        numeric_weight: float = NUMERIC_WEIGHT,
        genre_weight: float = GENRE_WEIGHT,
        *,
        suppress_duplicate_recordings: bool = True,
    ) -> None:
        for name, value in (("numeric_weight", numeric_weight), ("genre_weight", genre_weight)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise InvalidRequestError(f"{name} must be a number, got {value!r}")
            if value < 0:
                raise InvalidRequestError(f"{name} must be non-negative, got {value}")
        if not np.isclose(numeric_weight + genre_weight, 1.0):
            raise InvalidRequestError(
                f"weights must sum to 1.0, got {numeric_weight} + {genre_weight}"
            )

        self._fm = feature_matrix
        self._catalogue = feature_matrix.catalogue
        self._numeric_weight = float(numeric_weight)
        self._genre_weight = float(genre_weight)
        self._suppress = bool(suppress_duplicate_recordings)

        # Blocks are row-normalised already, so a dot product is the cosine.
        self._numeric = sparse.csr_matrix(feature_matrix.numeric_matrix)
        self._genre = sparse.csr_matrix(feature_matrix.genre_matrix)

        # Dense copy used only to explain which audio traits drive a match.
        self._numeric_dense = np.asarray(feature_matrix.numeric_matrix, dtype=float)

        self._track_ids = self._catalogue[ID_COLUMN].to_numpy()
        self._row_positions = pd.Series(
            np.arange(len(self._catalogue)), index=self._catalogue.index
        )
        self._genre_sets = [tuple(g) for g in self._catalogue["genre_set"]]
        self._artist_sets = [
            frozenset(part.strip().casefold() for part in str(value).split(";") if part.strip())
            for value in self._catalogue["artists"].fillna("")
        ]

        self._duplicate_keys = self._build_duplicate_keys() if self._suppress else None

    # -- construction helpers ------------------------------------------------ #

    def _build_duplicate_keys(self) -> list[_DuplicateKey]:
        catalogue = self._catalogue
        titles = catalogue["track_name"].fillna("").astype(str).str.strip().str.casefold()
        artists = catalogue["artists"].fillna("").astype(str).str.strip().str.casefold()
        audio = np.round(
            catalogue[list(NUMERIC_FEATURES)].to_numpy(dtype=float), SIGNATURE_DECIMALS
        )
        durations = catalogue[DURATION_COLUMN].fillna(-1).to_numpy(dtype=np.int64)

        return [
            _DuplicateKey(
                title=titles.iloc[i],
                artist=artists.iloc[i],
                audio=tuple(audio[i].tolist()),
                duration_ms=int(durations[i]),
            )
            for i in range(len(catalogue))
        ]

    # -- public properties --------------------------------------------------- #

    @property
    def numeric_weight(self) -> float:
        return self._numeric_weight

    @property
    def genre_weight(self) -> float:
        return self._genre_weight

    @property
    def n_tracks(self) -> int:
        return len(self._catalogue)

    @property
    def genres(self) -> tuple[str, ...]:
        """Genre vocabulary accepted by :class:`Preferences`."""
        return tuple(self._fm.genre_vocabulary)

    def track_row(self, track_id: str) -> int:
        """Matrix row for a ``track_id``; raises :class:`UnknownTrackError`."""
        if not isinstance(track_id, str) or not track_id.strip():
            raise InvalidRequestError(
                f"track_id must be a non-empty string, got {track_id!r}"
            )
        try:
            return int(self._row_positions.loc[track_id])
        except KeyError as exc:
            raise UnknownTrackError(f"unknown track_id: {track_id!r}") from exc

    # -- scoring ------------------------------------------------------------- #

    def _validate_preferences(self, preferences: Preferences) -> Preferences:
        if not isinstance(preferences, Preferences):
            raise InvalidRequestError(
                f"preferences must be a Preferences instance, got {type(preferences).__name__}"
            )
        unknown = [g for g in preferences.genres if g not in set(self.genres)]
        if unknown:
            raise InvalidRequestError(
                f"unknown genre(s) {unknown}; call .genres for the {len(self.genres)} valid values"
            )
        return preferences

    def _preference_scores(self, preferences: Preferences) -> np.ndarray:
        """Preference match in [0, 1] per catalogue row.

        Genre and artist components are each the share of the *requested* values
        the track satisfies; when both are requested their mean is used.
        """
        components: list[np.ndarray] = []
        count = len(self._catalogue)

        if preferences.genres:
            wanted = set(preferences.genres)
            hits = np.fromiter(
                (len(wanted & set(genres)) for genres in self._genre_sets),
                dtype=float,
                count=count,
            )
            components.append(hits / len(wanted))

        if preferences.artists:
            wanted = set(preferences.artists)
            hits = np.fromiter(
                (len(wanted & artists) for artists in self._artist_sets),
                dtype=float,
                count=count,
            )
            components.append(hits / len(wanted))

        if not components:
            return np.zeros(count, dtype=float)
        return np.mean(components, axis=0)

    def scores(self, track_id: str) -> pd.DataFrame:
        """Component content scores of every catalogue track against one reference.

        Returns one row per catalogue track (the reference included, at score
        1.0). Preferences are not applied here; use :meth:`recommend`.
        """
        row = self.track_row(track_id)
        numeric = self._numeric @ self._numeric.getrow(row).T
        genre = self._genre @ self._genre.getrow(row).T
        numeric_scores = np.asarray(numeric.todense()).ravel()
        genre_scores = np.asarray(genre.todense()).ravel()
        combined = self._numeric_weight * numeric_scores + self._genre_weight * genre_scores

        # Cosine is mathematically within [-1, 1]; float noise can overshoot.
        numeric_scores = np.clip(numeric_scores, -1.0, 1.0)
        genre_scores = np.clip(genre_scores, -1.0, 1.0)
        combined = np.clip(combined, -1.0, 1.0)

        return pd.DataFrame(
            {
                "track_id": self._track_ids,
                "score": combined,
                "numeric_score": numeric_scores,
                "genre_score": genre_scores,
            }
        )

    def recommend(
        self,
        track_id: str,
        k: int = DEFAULT_TOP_K,
        preferences: Preferences | None = None,
    ) -> pd.DataFrame:
        """Top-``k`` tracks similar to ``track_id``, optionally preference-adjusted.

        With no preferences the ranking and scores are exactly the content-based
        result. With preferences the final score is
        ``0.80 * base + 0.20 * preference_score``.

        Returns an empty DataFrame with the documented columns when no candidate
        survives, and fewer than ``k`` rows when the catalogue has fewer distinct
        candidates than ``k``.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise InvalidRequestError(f"k must be an integer, got {k!r}")
        if k < MIN_TOP_K or k > MAX_TOP_K:
            raise InvalidRequestError(f"k must be between {MIN_TOP_K} and {MAX_TOP_K}, got {k}")

        reference_row = self.track_row(track_id)

        if preferences is None:
            preferences = Preferences()
        preferences = self._validate_preferences(preferences)

        scored = self.scores(track_id)
        base = scored["score"].to_numpy()

        if preferences.is_empty():
            final = base
            preference_scores = np.zeros_like(base)
        else:
            preference_scores = self._preference_scores(preferences)
            final = BASE_WEIGHT_WITH_PREFERENCES * base + PREFERENCE_WEIGHT * preference_scores

        order = np.lexsort((self._track_ids, -final))
        order = order[order != reference_row]

        chosen: list[int] = []
        seen: set[_DuplicateKey] = set()
        if self._suppress and self._duplicate_keys is not None:
            # Excluding the reference row is not enough: another row can hold the
            # same recording under a different track_id.
            seen.add(self._duplicate_keys[reference_row])
        for row in order:
            if self._suppress and self._duplicate_keys is not None:
                key = self._duplicate_keys[row]
                if key in seen:
                    continue
                seen.add(key)
            chosen.append(int(row))
            if len(chosen) == k:
                break

        return self._format_results(
            scored=scored,
            rows=chosen,
            final_scores=final,
            preference_scores=preference_scores,
            reference_row=reference_row,
            preferences=preferences,
        )

    def more_by_artist(self, track_id: str, artist: str, k: int = 5) -> pd.DataFrame:
        """Up to ``k`` tracks by ``artist``, ordered by content similarity to the reference.

        This is a separate, explicitly labelled list for users who want songs by a
        particular artist. It does not change :meth:`recommend` or its scores: the
        ``score`` column is the plain content similarity (no preference boost).
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < MIN_TOP_K or k > MAX_TOP_K:
            raise InvalidRequestError(f"k must be between {MIN_TOP_K} and {MAX_TOP_K}, got {k!r}")
        wanted = str(artist or "").strip().casefold()
        if not wanted:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        reference_row = self.track_row(track_id)
        preferences = Preferences.build(artists=[artist])
        scored = self.scores(track_id)
        base = scored["score"].to_numpy()

        candidates = np.array(
            [i for i, artists in enumerate(self._artist_sets) if wanted in artists and i != reference_row],
            dtype=int,
        )
        if candidates.size == 0:
            return self._format_results(scored, [], base, np.zeros_like(base), reference_row, preferences)

        order = candidates[np.lexsort((self._track_ids[candidates], -base[candidates]))]
        chosen: list[int] = []
        seen: set[_DuplicateKey] = set()
        if self._suppress and self._duplicate_keys is not None:
            seen.add(self._duplicate_keys[reference_row])
        for row in order:
            if self._suppress and self._duplicate_keys is not None:
                key = self._duplicate_keys[row]
                if key in seen:
                    continue
                seen.add(key)
            chosen.append(int(row))
            if len(chosen) == k:
                break

        preference_scores = np.zeros_like(base)
        preference_scores[chosen] = 1.0
        return self._format_results(scored, chosen, base, preference_scores, reference_row, preferences)

    # -- explanations -------------------------------------------------------- #

    def _explain(
        self,
        reference_row: int,
        candidate_row: int,
        preferences: Preferences,
    ) -> str:
        """Deterministic, fact-based explanation built from real values."""
        parts: list[str] = []

        reference_genres = set(self._genre_sets[reference_row])
        candidate_genres = self._genre_sets[candidate_row]
        shared = sorted(reference_genres & set(candidate_genres))
        if shared:
            listed = shared[:EXPLANATION_GENRES]
            label = "genre" if len(listed) == 1 else "genres"
            parts.append(f"shared {label}: {', '.join(listed)}")

        # Element-wise products of the unit-normalised numeric rows sum to S_num,
        # so the largest products are the audio traits that pull the tracks together.
        contributions = self._numeric_dense[reference_row] * self._numeric_dense[candidate_row]
        top = sorted(
            zip(NUMERIC_FEATURES, contributions, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
        positive = [name for name, value in top if value > 0][:EXPLANATION_TRAITS]
        if positive:
            parts.append(f"closest audio traits: {', '.join(positive)}")

        matched_genres = sorted(set(preferences.genres) & set(candidate_genres))
        if matched_genres:
            parts.append(f"matches preferred genre: {', '.join(matched_genres)}")

        matched_artists = sorted(set(preferences.artists) & self._artist_sets[candidate_row])
        if matched_artists:
            parts.append(f"matches preferred artist: {', '.join(matched_artists)}")

        if not parts:
            parts.append("no shared genre; the audio profile is the only link")
        return "; ".join(parts)

    # -- output -------------------------------------------------------------- #

    def _format_results(
        self,
        scored: pd.DataFrame,
        rows: list[int],
        final_scores: np.ndarray,
        preference_scores: np.ndarray,
        reference_row: int,
        preferences: Preferences,
    ) -> pd.DataFrame:
        if not rows:
            empty = pd.DataFrame(columns=RESULT_COLUMNS)
            empty.attrs["score_warning"] = SCORE_INTERPRETATION_WARNING
            empty.attrs["preferences_applied"] = not preferences.is_empty()
            return empty

        subset = self._catalogue.iloc[rows]
        results = pd.DataFrame(
            {
                "rank": np.arange(1, len(rows) + 1),
                "track_id": subset[ID_COLUMN].to_numpy(),
                "track_name": subset["track_name"].to_numpy(),
                "artists": subset["artists"].to_numpy(),
                "album_name": subset["album_name"].to_numpy(),
                "genres": [", ".join(g) for g in subset["genre_set"]],
                "duration_ms": subset[DURATION_COLUMN].to_numpy(),
                "score": final_scores[rows],
                "base_score": scored["score"].to_numpy()[rows],
                "preference_score": preference_scores[rows],
                "numeric_score": scored["numeric_score"].to_numpy()[rows],
                "genre_score": scored["genre_score"].to_numpy()[rows],
                "explanation": [
                    self._explain(reference_row, row, preferences) for row in rows
                ],
            }
        )
        results.attrs["score_warning"] = SCORE_INTERPRETATION_WARNING
        results.attrs["preferences_applied"] = not preferences.is_empty()
        return results
