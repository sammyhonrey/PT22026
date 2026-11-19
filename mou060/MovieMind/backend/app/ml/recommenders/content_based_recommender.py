"""Content-based recommender (Stage B) — genre-preference personalization.

Builds a genre preference vector from the user's own application ratings (each
score centered around the neutral 3.0), then ranks unrated movies by how well
their genres match those preferences.

The ranking is pushed down to SQL (a ``CASE``-weighted genre sum, ``ORDER BY`` +
``LIMIT``) so only the top ``limit`` movie ids cross the wire — a full-table
scan of the catalogue is avoided (important because the dev database is a
remote, high-latency Neon instance). Only the winning movies are then fetched
as full ORM objects for serialization.

The pure helpers (``compute_genre_preferences`` / ``score_by_preferences``) stay
free of ORM/DB so they can be unit-tested in isolation.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Iterable

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models.movie import Genre, Movie, MovieGenre
from app.db.models.rating import Rating
from app.services.recommendation_service import (
    SOURCE_CONTENT_BASED,
    RecommendationItem,
    RecommendationStrategy,
)

# Neutral score: ratings above this increase preference, below decrease it.
NEUTRAL_SCORE = 3.0


def compute_genre_preferences(
    rated: Iterable[tuple[Collection[str], float]],
) -> dict[str, float]:
    """Return ``{genre: centered_average_score}`` for a user's rated movies.

    ``rated`` is an iterable of ``(genres, score)`` tuples. Each genre's
    preference is ``mean(score) - NEUTRAL_SCORE``, so positive values mean the
    user liked that genre and negative values mean they disliked it.
    """
    by_genre: dict[str, list[float]] = {}
    for genres, score in rated:
        for genre in genres:
            by_genre.setdefault(genre, []).append(float(score))

    return {
        genre: (sum(scores) / len(scores)) - NEUTRAL_SCORE for genre, scores in by_genre.items()
    }


def score_by_preferences(genres: Collection[str], prefs: dict[str, float]) -> float:
    """Sum the preference weights of a movie's genres (0.0 if none known)."""
    if not genres:
        return 0.0
    return sum(prefs.get(genre, 0.0) for genre in genres)


class ContentBasedRecommender(RecommendationStrategy):
    """Recommend unrated movies by similarity to the user's genre preferences."""

    source = SOURCE_CONTENT_BASED

    def __init__(
        self,
        db: Session,
        user_id: uuid.UUID,
        *,
        watchlisted_movie_ids: Collection[uuid.UUID] | None = None,
        watchlist_weight: float = 0.5,
    ) -> None:
        self._db = db
        self._user_id = user_id
        self._watchlisted_movie_ids = set(watchlisted_movie_ids or ())
        self._watchlist_weight = watchlist_weight

    def recommend(self, *, limit: int) -> list[RecommendationItem]:
        ratings = list(
            self._db.scalars(
                select(Rating)
                .where(Rating.user_id == self._user_id)
                .options(
                    selectinload(Rating.movie)
                    .selectinload(Movie.genre_links)
                    .selectinload(MovieGenre.genre)
                )
            ).all()
        )

        watchlist_items: list[tuple[Collection[str], float]] = []
        if self._watchlisted_movie_ids:
            movies = {
                m.id: m
                for m in self._db.scalars(
                    select(Movie)
                    .where(Movie.id.in_(sorted(self._watchlisted_movie_ids)))
                    .options(selectinload(Movie.genre_links).selectinload(MovieGenre.genre))
                ).all()
            }
            for movie_id in sorted(self._watchlisted_movie_ids):
                movie = movies.get(movie_id)
                if movie is not None:
                    watchlist_items.append((movie.genres, 3.5 * self._watchlist_weight))

        prefs = compute_genre_preferences(
            [(r.movie.genres, float(r.score)) for r in ratings] + watchlist_items
        )
        rated_ids = {r.movie_id for r in ratings}
        if not prefs:
            return []

        # Rank in SQL: a movie's score is the sum of its genres' preference
        # weights. Only the top ``limit`` ids are returned.
        score_case = case(
            {name: weight for name, weight in prefs.items()},
            value=Genre.name,
            else_=0.0,
        )
        score_expr = func.coalesce(func.sum(score_case), 0.0)

        stmt = (
            select(Movie.id, score_expr.label("score"))
            .outerjoin(MovieGenre, MovieGenre.movie_id == Movie.id)
            .outerjoin(Genre, Genre.id == MovieGenre.genre_id)
        )
        if rated_ids:
            stmt = stmt.where(Movie.id.not_in(rated_ids))
        stmt = stmt.group_by(Movie.id).order_by(score_expr.desc(), Movie.id).limit(limit)

        top = self._db.execute(stmt).all()
        movie_ids = [movie_id for movie_id, _ in top]
        if not movie_ids:
            return []

        movies = {
            m.id: m
            for m in self._db.scalars(
                select(Movie)
                .where(Movie.id.in_(movie_ids))
                .options(selectinload(Movie.genre_links).selectinload(MovieGenre.genre))
            ).all()
        }

        items = [
            RecommendationItem(
                movie=movies[movie_id],
                score=float(score),
                source=self.source,
                reason="Matches genres you rated highly",
            )
            for movie_id, score in top
            if movie_id in movies
        ]
        # Deterministic tie-break on the (small) final result set.
        items.sort(key=lambda i: (-i.score, i.movie.title, str(i.movie.id)))
        return items
