from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.movie import Genre, Movie, MovieGenre
from app.ml.recommenders.preference_based_recommender import PreferenceBasedRecommender


def test_preference_recommender_excludes_favorites_and_ranks_description_matches():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        science_fiction = Genre(name="Sci-Fi")
        romance = Genre(name="Romance")
        favorite = Movie(title="Favorite Space Romance")
        description_match = Movie(title="A Space Story")
        genre_match = Movie(title="Alien")
        db.add_all([science_fiction, romance, favorite, description_match, genre_match])
        db.flush()
        db.add_all(
            [
                MovieGenre(movie_id=favorite.id, genre_id=romance.id),
                MovieGenre(movie_id=description_match.id, genre_id=science_fiction.id),
                MovieGenre(movie_id=genre_match.id, genre_id=science_fiction.id),
            ]
        )
        db.flush()

        recommender = PreferenceBasedRecommender(
            db,
            favorite_genres=["Science Fiction"],
            favorite_movie_ids=[favorite.id],
            description="I love space stories",
        )
        results = recommender.recommend(limit=1)

        assert len(results) == 1
        assert results[0].movie.id == description_match.id
        assert results[0].score > 2.5
        assert all(result.movie.id != favorite.id for result in results)
