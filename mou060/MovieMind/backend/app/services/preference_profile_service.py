from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.db.models.movie import Movie
from app.db.models.user_preference import UserPreference
from app.ml.recommenders.preference_based_recommender import PreferenceBasedRecommender
from app.repositories import movies as movie_repo


class PreferenceProfileService:
    """Build and rank recommendations from a saved user preference profile."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def build_profile(self, preference: UserPreference | None) -> tuple[list[str], list[uuid.UUID], str | None]:
        if preference is None:
            return [], [], None
        favorite_genres = [str(genre).strip() for genre in preference.favorite_genres or [] if str(genre).strip()]
        favorite_movie_ids = [uuid.UUID(str(movie_id)) for movie_id in (preference.favorite_movie_ids or []) if str(movie_id).strip()]
        description = preference.description.strip() if preference.description and preference.description.strip() else None
        return favorite_genres, favorite_movie_ids, description

    def generate(self, preference: UserPreference | None, *, limit: int = 10) -> list[tuple[Movie, float, str]]:
        favorite_genres, favorite_movie_ids, description = self.build_profile(preference)
        if not favorite_genres and not favorite_movie_ids and not description:
            return []

        recommender = PreferenceBasedRecommender(
            self._db,
            favorite_genres=favorite_genres,
            favorite_movie_ids=favorite_movie_ids,
            description=description,
        )
        ranked = recommender.recommend(limit=limit)
        return [(item.movie, item.score, item.reason) for item in ranked]

    def has_valid_profile(self, preference: UserPreference | None) -> bool:
        if preference is None:
            return False
        favorite_genres = [str(genre).strip() for genre in (preference.favorite_genres or []) if str(genre).strip()]
        favorite_movie_ids = [str(movie_id).strip() for movie_id in (preference.favorite_movie_ids or []) if str(movie_id).strip()]
        description = preference.description.strip() if preference.description and preference.description.strip() else ""
        return bool(favorite_genres or favorite_movie_ids or description)
