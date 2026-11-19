"""Recommendation endpoints.

- ``GET /recommendations/me`` — personalized recommendations for the current
  user (popularity fallback for cold-start users, content-based once they have
  enough ratings). Never maps a MovieMind account id into the NCF model.
- ``GET /recommendations/demo/movielens/{user_id}`` — NCF demo for a known raw
  MovieLens user id (gated by ``ENABLE_DEMO_ENDPOINTS``).
- ``GET /predictions/demo/movielens`` — single NCF demo prediction.

HTTP handlers here contain no tensor or raw-SQL logic; that lives in the
strategies under ``app/ml/recommenders/``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import CurrentUser, DbSession, rate_limit_demo
from app.core.config import get_settings
from app.core.errors import NotFoundApiError
from app.ml.recommenders.collaborative_recommender import CollaborativeRecommender
from app.ml.recommenders.content_based_recommender import ContentBasedRecommender
from app.ml.recommenders.hybrid_recommender import HybridRecommender
from app.ml.recommenders.ncf_demo_recommender import NCFDemoRecommender
from app.ml.recommenders.popularity_recommender import PopularityRecommender
from app.repositories import ratings as rating_repo
from app.repositories import watchlist as watchlist_repo
from app.schemas.movie import MovieOut
from app.schemas.recommendation import PredictionOut, RecommendationOut, RecommendationsOut
from app.services import model_service
from app.services.recommendation_service import RecommendationItem, RecommendationService

router = APIRouter(tags=["recommendations"])


def _to_out(item: RecommendationItem) -> RecommendationOut:
    return RecommendationOut(
        movie=MovieOut.model_validate(item.movie),
        score=item.score,
        source=item.source,
        reason=item.reason,
    )


def _require_demo_enabled() -> None:
    if not get_settings().ENABLE_DEMO_ENDPOINTS:
        raise NotFoundApiError("Demo endpoints are disabled")


def _parse_exclude(raw: str | None) -> set[str]:
    if not raw:
        return set()
    result: set[str] = set()
    for part in raw.split(","):
        part = part.strip()
        if part:
            result.add(part)
    return result


@router.get("/recommendations/me", response_model=RecommendationsOut, summary="My recommendations")
def my_recommendations(
    user: CurrentUser,
    db: DbSession,
    limit: int = Query(20, ge=1, le=50),
) -> RecommendationsOut:
    settings = get_settings()
    rated_ids = rating_repo.list_movie_ids(db, user_id=user.id)
    watchlisted_ids = watchlist_repo.list_movie_ids(db, user_id=user.id)

    if len(rated_ids) < settings.MIN_RATINGS_FOR_PERSONALIZATION:
        strategy = PopularityRecommender(db)
        is_fallback = True
    else:
        collaborative = CollaborativeRecommender(
            db,
            user.id,
            watchlisted_movie_ids=watchlisted_ids,
        )
        if collaborative.recommend(limit=1):
            strategy = HybridRecommender(
                db,
                user.id,
                content_weight=settings.RECOMMENDATION_CONTENT_WEIGHT,
                collaborative_weight=settings.RECOMMENDATION_COLLABORATIVE_WEIGHT,
                watchlisted_movie_ids=watchlisted_ids,
            )
        else:
            strategy = ContentBasedRecommender(
                db,
                user.id,
                watchlisted_movie_ids=watchlisted_ids,
            )
        is_fallback = False

    service = RecommendationService()
    items = service.recommend(
        strategy,
        limit=limit,
        rated_movie_ids=rated_ids,
        watchlisted_movie_ids=watchlisted_ids,
    )
    return RecommendationsOut(
        items=[_to_out(i) for i in items],
        source=strategy.source,
        is_fallback=is_fallback,
        limit=limit,
    )


@router.get(
    "/recommendations/demo/movielens/{user_id}",
    response_model=list[RecommendationOut],
    dependencies=[Depends(rate_limit_demo)],
    summary="NCF demo recommendations for a MovieLens user",
)
def demo_movielens_recommendations(
    user_id: str,
    db: DbSession,
    limit: int = Query(20, ge=1, le=100),
    exclude: str | None = Query(None, description="Comma-separated raw MovieLens movie ids"),
) -> list[RecommendationOut]:
    _require_demo_enabled()
    recommender = model_service.get_ncf_recommender()
    if not recommender.has_user(user_id):
        raise NotFoundApiError("Unknown MovieLens user id")

    strategy = NCFDemoRecommender(
        db, recommender, user_id, exclude_movie_ids=_parse_exclude(exclude)
    )
    return [_to_out(i) for i in strategy.recommend(limit=limit)]


@router.get(
    "/predictions/demo/movielens",
    response_model=PredictionOut,
    dependencies=[Depends(rate_limit_demo)],
    summary="NCF demo prediction for one (user, movie) pair",
)
def demo_movielens_prediction(
    user_id: str = Query(..., description="Raw MovieLens user id"),
    movie_id: str = Query(..., description="Raw MovieLens movie id"),
) -> PredictionOut:
    _require_demo_enabled()
    recommender = model_service.get_ncf_recommender()
    if not recommender.has_user(user_id):
        raise NotFoundApiError("Unknown MovieLens user id")
    if not recommender.has_movie(movie_id):
        raise NotFoundApiError("Unknown MovieLens movie id")

    return PredictionOut(
        user_id=user_id,
        movie_id=movie_id,
        score=recommender.predict(user_id, movie_id),
    )
