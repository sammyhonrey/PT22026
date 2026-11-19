"""NCF demo recommender — model validation for known MovieLens users only.

Wraps :class:`NCFRecommender` as a :class:`RecommendationStrategy`. It scores
every movie id the model knows (taken from the in-memory mapping — no
full-table catalogue scan), sorts by predicted rating, then fetches only the
winning movies from the database for serialization.

It must never receive a MovieMind account id — callers pass a raw MovieLens id
from the demo route.

Concurrency note (T9.4): inference is synchronous but runs *off the event loop*
because the demo routes are plain ``def`` endpoints, which FastAPI executes in
its worker threadpool. Measured on this machine: a full 9724-movie batch scores
in ~0.05 s on CPU and a single-pair prediction is sub-millisecond, so inference
cannot starve the liveness/readiness checks.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db.models.movie import Movie, MovieGenre
from app.ml.recommenders.ncf_recommender import NCFRecommender
from app.services.recommendation_service import (
    SOURCE_NCF,
    RecommendationItem,
    RecommendationStrategy,
)


class NCFDemoRecommender(RecommendationStrategy):
    """Score mapped catalogue movies for one known raw MovieLens user."""

    source = SOURCE_NCF

    def __init__(
        self,
        db: Session,
        recommender: NCFRecommender,
        movielens_user_id: str,
        exclude_movie_ids: set[str] | None = None,
    ) -> None:
        self._db = db
        self._recommender = recommender
        self._user_id = movielens_user_id
        self._exclude = exclude_movie_ids or set()

    def recommend(self, *, limit: int) -> list[RecommendationItem]:
        raw_ids = [r for r in self._recommender.known_movie_ids() if r not in self._exclude]
        if not raw_ids:
            return []

        # One batched forward pass over every known movie (in-memory, fast).
        scored = self._recommender.score_candidates(self._user_id, raw_ids)
        scored.sort(key=lambda pair: -pair[1])
        top = scored[:limit]

        # Fetch only the winning movies (avoid a full catalogue scan on the
        # remote database).
        int_ids = [int(raw_id) for raw_id, _ in top]
        movies = {
            m.movielens_id: m
            for m in self._db.scalars(
                select(Movie)
                .where(Movie.movielens_id.in_(int_ids))
                .options(selectinload(Movie.genre_links).selectinload(MovieGenre.genre))
            ).all()
        }

        items = [
            RecommendationItem(
                movie=movies[int(raw_id)],
                score=score,
                source=self.source,
                reason="Predicted rating from the NCF model",
            )
            for raw_id, score in top
            if int(raw_id) in movies
        ]
        # Deterministic tie-break on the (small) final result set.
        items.sort(key=lambda i: (-i.score, i.movie.title, str(i.movie.id)))
        return items
