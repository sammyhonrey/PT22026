from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, DbSession
from app.core.errors import NotFoundApiError, ValidationApiError
from app.ml.recommenders.popularity_recommender import PopularityRecommender
from app.repositories import user_preferences as user_pref_repo
from app.schemas.movie import MovieOut
from app.schemas.recommendation import RecommendationOut, RecommendationsOut
from app.schemas.user_preference import UserPreferenceInput, UserPreferenceOut
from app.services.preference_profile_service import PreferenceProfileService

router = APIRouter()


def _to_response(preference) -> UserPreferenceOut:
    return UserPreferenceOut(
        user_id=str(preference.user_id),
        favorite_genres=list(preference.favorite_genres or []),
        favorite_movie_ids=list(preference.favorite_movie_ids or []),
        description=preference.description,
        created_at=preference.created_at,
        updated_at=preference.updated_at,
    )


@router.get("/preferences/me", response_model=UserPreferenceOut | None, summary="Get my movie preferences")
def get_preferences(user: CurrentUser, db: DbSession) -> UserPreferenceOut | None:
    preference = user_pref_repo.get_for_user(db, user_id=user.id)
    if preference is None:
        return None
    return _to_response(preference)


@router.put("/preferences/me", response_model=UserPreferenceOut, summary="Save my movie preferences")
def save_preferences(
    user: CurrentUser,
    payload: UserPreferenceInput,
    db: DbSession,
) -> UserPreferenceOut:
    normalized_genres = payload.normalized_genres
    normalized_movie_ids = payload.normalized_movie_ids
    description = payload.normalized_description

    if not normalized_genres and not normalized_movie_ids and not description:
        raise ValidationApiError("Provide at least one genre, movie, or description to save your taste profile.")

    preference = user_pref_repo.upsert(
        db,
        user_id=user.id,
        favorite_genres=normalized_genres,
        favorite_movie_ids=normalized_movie_ids,
        description=description,
    )
    return _to_response(preference)


@router.post("/preferences/me/recommend", response_model=RecommendationsOut, summary="Recommend movies from my taste profile")
def recommend_preferences(
    user: CurrentUser,
    db: DbSession,
    payload: UserPreferenceInput | None = None,
    limit: int = Query(20, ge=1, le=50),
) -> RecommendationsOut:
    if payload is None:
        payload = UserPreferenceInput()
    normalized = UserPreferenceInput(
        favorite_genres=payload.favorite_genres,
        favorite_movie_ids=payload.favorite_movie_ids,
        description=payload.description,
    )
    saved = user_pref_repo.get_for_user(db, user_id=user.id)
    has_input = bool(normalized.normalized_genres or normalized.normalized_movie_ids or normalized.normalized_description)
    if has_input:
        preference = user_pref_repo.upsert(
            db,
            user_id=user.id,
            favorite_genres=normalized.normalized_genres,
            favorite_movie_ids=normalized.normalized_movie_ids,
            description=normalized.normalized_description,
        )
    else:
        preference = saved

    if preference is None or not has_input:
        fallback = PopularityRecommender(db).recommend(limit=limit)
        return RecommendationsOut(
            items=[RecommendationOut(movie=MovieOut.model_validate(item.movie), score=item.score, source=item.source, reason=item.reason) for item in fallback],
            source="popularity",
            is_fallback=True,
            limit=limit,
        )

    service = PreferenceProfileService(db)
    ranked = service.generate(preference, limit=limit)
    if not ranked:
        fallback = PopularityRecommender(db).recommend(limit=limit)
        return RecommendationsOut(
            items=[RecommendationOut(movie=MovieOut.model_validate(item.movie), score=item.score, source=item.source, reason=item.reason) for item in fallback],
            source="popularity",
            is_fallback=True,
            limit=limit,
        )

    return RecommendationsOut(
        items=[
            RecommendationOut(
                movie=MovieOut.model_validate(movie),
                score=score,
                source="preference_based",
                reason=reason,
            )
            for movie, score, reason in ranked
        ],
        source="preference_based",
        is_fallback=False,
        limit=limit,
    )
