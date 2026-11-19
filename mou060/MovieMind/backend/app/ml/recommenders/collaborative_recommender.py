from __future__ import annotations

import math
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models.movie import Movie, MovieGenre
from app.db.models.rating import Rating
from app.services.recommendation_service import (
    SOURCE_COLLABORATIVE,
    RecommendationItem,
    RecommendationStrategy,
)


class CollaborativeRecommender(RecommendationStrategy):
    """Recommend movies based on similarity to users who rated similarly."""

    source = SOURCE_COLLABORATIVE

    def __init__(
        self,
        db: Session,
        user_id: uuid.UUID,
        *,
        watchlisted_movie_ids: set[uuid.UUID] | None = None,
        min_overlap: int = 2,
        min_similarity: float = 0.05,
        max_neighbors: int = 30,
    ) -> None:
        self._db = db
        self._user_id = user_id
        self._watchlisted_movie_ids = set(watchlisted_movie_ids or ())
        self._min_overlap = min_overlap
        self._min_similarity = min_similarity
        self._max_neighbors = max_neighbors

    def recommend(self, *, limit: int) -> list[RecommendationItem]:
        target_ratings = list(
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
        if len(target_ratings) < self._min_overlap:
            return []

        target_scores = {rating.movie_id: float(rating.score) for rating in target_ratings}
        if self._watchlisted_movie_ids:
            for movie_id in sorted(self._watchlisted_movie_ids):
                if movie_id not in target_scores:
                    target_scores[movie_id] = 3.5 * 0.5
        target_mean = sum(target_scores.values()) / len(target_scores)
        overlapping_movie_ids = list(target_scores)

        similar_users: list[tuple[uuid.UUID, float, dict[uuid.UUID, float], float]] = []

        # Rebuild the neighbor map in a single pass so the similarity calculation
        # uses only the exact overlapping ratings between the target user and a
        # peer user.
        other_user_ratings: dict[uuid.UUID, dict[uuid.UUID, float]] = {}
        rows = self._db.execute(
            select(Rating.user_id, Rating.movie_id, Rating.score)
            .where(Rating.user_id != self._user_id, Rating.movie_id.in_(overlapping_movie_ids))
        ).all()
        for user_id, movie_id, score in rows:
            other_user_ratings.setdefault(user_id, {})[movie_id] = float(score)

        for other_user, other_scores in other_user_ratings.items():
            overlap = sorted(set(target_scores) & set(other_scores))
            if len(overlap) < self._min_overlap:
                continue
            target_values = [target_scores[movie_id] for movie_id in overlap]
            peer_values = [other_scores[movie_id] for movie_id in overlap]
            target_centered = [value - (sum(target_values) / len(target_values)) for value in target_values]
            peer_centered = [value - (sum(peer_values) / len(peer_values)) for value in peer_values]
            numerator = sum(
                target_value * peer_value
                for target_value, peer_value in zip(target_centered, peer_centered, strict=True)
            )
            denominator = (
                math.sqrt(sum(value * value for value in target_centered))
                * math.sqrt(sum(value * value for value in peer_centered))
            )
            if denominator == 0:
                continue
            similarity = numerator / denominator
            if similarity > self._min_similarity:
                similar_users.append((other_user, similarity, other_scores, sum(peer_values) / len(peer_values)))

        if not similar_users:
            return []

        similar_users.sort(key=lambda item: item[1], reverse=True)
        similar_users = similar_users[: self._max_neighbors]

        candidate_scores: dict[uuid.UUID, float] = {}
        candidate_weights: dict[uuid.UUID, float] = {}
        for other_user, similarity, other_scores, other_mean in similar_users:
            for movie_id, score in other_scores.items():
                if movie_id in target_scores:
                    continue
                candidate_scores[movie_id] = candidate_scores.get(movie_id, 0.0) + similarity * (
                    score - other_mean
                )
                candidate_weights[movie_id] = candidate_weights.get(movie_id, 0.0) + abs(similarity)

        if not candidate_scores:
            return []

        candidate_ids = list(candidate_scores)
        movies = {
            movie.id: movie
            for movie in self._db.scalars(
                select(Movie)
                .where(Movie.id.in_(candidate_ids))
                .options(selectinload(Movie.genre_links).selectinload(MovieGenre.genre))
            ).all()
        }

        ranked: list[RecommendationItem] = []
        for movie_id, weighted_sum in candidate_scores.items():
            movie = movies.get(movie_id)
            if movie is None:
                continue
            total_weight = candidate_weights.get(movie_id, 0.0)
            if total_weight == 0:
                continue
            predicted_score = target_mean + (weighted_sum / total_weight)
            predicted_score = min(max(float(predicted_score), 0.5), 5.0)
            ranked.append(
                RecommendationItem(
                    movie=movie,
                    score=predicted_score,
                    source=self.source,
                    reason="Similar users rated this movie positively",
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.movie.title, str(item.movie.id)))
        return ranked[:limit]
