from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ml.recommenders.collaborative_recommender import CollaborativeRecommender
from app.ml.recommenders.content_based_recommender import ContentBasedRecommender
from app.services.recommendation_service import (
    SOURCE_HYBRID,
    RecommendationItem,
    RecommendationStrategy,
    combine_score_components,
    normalize_score,
)


class HybridRecommender(RecommendationStrategy):
    """Blend content-based and collaborative signals for a user."""

    source = SOURCE_HYBRID

    def __init__(
        self,
        db: Session,
        user_id: uuid.UUID,
        *,
        watchlisted_movie_ids: set[uuid.UUID] | None = None,
        content_weight: float | None = None,
        collaborative_weight: float | None = None,
    ) -> None:
        settings = get_settings()
        self._db = db
        self._user_id = user_id
        self._content_weight = (
            settings.RECOMMENDATION_CONTENT_WEIGHT if content_weight is None else content_weight
        )
        self._collaborative_weight = (
            settings.RECOMMENDATION_COLLABORATIVE_WEIGHT
            if collaborative_weight is None
            else collaborative_weight
        )
        self._content_recommender = ContentBasedRecommender(
            db,
            user_id,
            watchlisted_movie_ids=watchlisted_movie_ids,
        )
        self._collaborative_recommender = CollaborativeRecommender(
            db,
            user_id,
            watchlisted_movie_ids=watchlisted_movie_ids,
        )

    def recommend(self, *, limit: int) -> list[RecommendationItem]:
        content_items = self._content_recommender.recommend(limit=max(limit * 3, 10))
        collaborative_items = self._collaborative_recommender.recommend(limit=max(limit * 3, 10))

        if not content_items and not collaborative_items:
            return []
        if not collaborative_items:
            return content_items[:limit]
        if not content_items:
            return collaborative_items[:limit]

        content_lookup: dict[uuid.UUID, RecommendationItem] = {
            item.movie.id: item for item in content_items
        }
        collaborative_lookup: dict[uuid.UUID, RecommendationItem] = {
            item.movie.id: item for item in collaborative_items
        }

        combined: dict[uuid.UUID, dict[str, RecommendationItem | None]] = {}
        for movie_id, item in content_lookup.items():
            combined.setdefault(movie_id, {"content": None, "collaborative": None})["content"] = item
        for movie_id, item in collaborative_lookup.items():
            combined.setdefault(movie_id, {"content": None, "collaborative": None})[
                "collaborative"
            ] = item

        ranked: list[RecommendationItem] = []
        for movie_id, signals in combined.items():
            content_item = signals.get("content")
            collab_item = signals.get("collaborative")
            if content_item is None and collab_item is None:
                continue
            if content_item is None:
                ranked.append(
                    RecommendationItem(
                        movie=collab_item.movie,
                        score=float(collab_item.score),
                        source=collab_item.source,
                        reason=collab_item.reason,
                    )
                )
                continue
            if collab_item is None:
                ranked.append(
                    RecommendationItem(
                        movie=content_item.movie,
                        score=float(content_item.score),
                        source=content_item.source,
                        reason=content_item.reason,
                    )
                )
                continue

            content_component = normalize_score(content_item.score)
            collaborative_component = normalize_score(collab_item.score)
            score = combine_score_components(
                content_component,
                collaborative_component,
                content_weight=self._content_weight,
                collaborative_weight=self._collaborative_weight,
            )
            ranked.append(
                RecommendationItem(
                    movie=content_item.movie,
                    score=float(score),
                    source=self.source,
                    reason="Combined your genre preferences with similar-user signals",
                )
            )

        if not ranked:
            return []

        ranked.sort(key=lambda item: (-item.score, item.movie.title, str(item.movie.id)))
        return ranked[:limit]
