"""Popularity recommender — the honest cold-start fallback.

Ranks movies by their *community* ratings (count, then average) computed live
from the application's ``ratings`` table, with deterministic tie-breaks. When
nobody has rated anything yet, the ordering degrades gracefully to title order.

This strategy never fabricates a score or an explanation: the ``reason`` always
describes the ranking method actually used.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models.movie import Movie, MovieGenre
from app.db.models.rating import Rating
from app.services.recommendation_service import (
    SOURCE_POPULARITY,
    RecommendationItem,
    RecommendationStrategy,
)


class PopularityRecommender(RecommendationStrategy):
    """Rank movies by community rating count and average score."""

    source = SOURCE_POPULARITY

    def __init__(self, db: Session) -> None:
        self._db = db

    def recommend(self, *, limit: int) -> list[RecommendationItem]:
        stats = (
            select(
                Rating.movie_id,
                func.count().label("rating_count"),
                func.avg(Rating.score).label("avg_score"),
            )
            .group_by(Rating.movie_id)
            .subquery()
        )
        stmt = (
            select(Movie, func.coalesce(stats.c.avg_score, 0.0))
            .options(selectinload(Movie.genre_links).selectinload(MovieGenre.genre))
            .outerjoin(stats, Movie.id == stats.c.movie_id)
            .order_by(
                func.coalesce(stats.c.rating_count, 0).desc(),
                func.coalesce(stats.c.avg_score, 0).desc(),
                Movie.title.asc(),
                Movie.id,
            )
            .limit(limit)
        )
        items: list[RecommendationItem] = []
        for movie, avg_score in self._db.execute(stmt).all():
            items.append(
                RecommendationItem(
                    movie=movie,
                    score=float(avg_score),
                    source=self.source,
                    reason="Popular movies ranked by community ratings",
                )
            )
        return items
