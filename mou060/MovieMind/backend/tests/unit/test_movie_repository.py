from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models.movie import Genre, Movie, MovieGenre
from app.repositories.movies import list_movies


def test_list_movies_searches_genre_names():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        genre = Genre(name="Science Fiction")
        movie = Movie(title="A Space Story")
        db.add_all([genre, movie])
        db.flush()
        db.add(MovieGenre(movie_id=movie.id, genre_id=genre.id))
        db.flush()

        results, total = list_movies(
            db,
            limit=8,
            offset=0,
            q="science",
            sort="title",
        )

        assert total == 1
        assert [result.id for result in results] == [movie.id]
