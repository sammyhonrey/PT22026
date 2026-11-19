

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Movie(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "movies"

    # External MovieLens ID — only set for movies that can be scored by the NCF model.
    movielens_id: Mapped[int | None] = mapped_column(
        Integer, unique=True, index=True, nullable=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    overview: Mapped[str | None] = mapped_column(Text)
    release_year: Mapped[int | None] = mapped_column(Integer)
    # Sourced only from an approved provider; never fabricated (open decision #4).
    poster_url: Mapped[str | None] = mapped_column(String(2048))
    # Cached/derived fields, updated by import or background refresh.
    average_rating: Mapped[float | None] = mapped_column(Numeric(3, 2))
    ratings_count: Mapped[int | None] = mapped_column(Integer, default=0)

    genre_links: Mapped[list[MovieGenre]] = relationship(
        back_populates="movie", cascade="all, delete-orphan"
    )

    @property
    def genres(self) -> list[str]:
        """Convenience: ordered genre names for serialization."""
        return [link.genre.name for link in self.genre_links]


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    movie_links: Mapped[list[MovieGenre]] = relationship(back_populates="genre")


class MovieGenre(Base):
    __tablename__ = "movie_genres"
    __table_args__ = (UniqueConstraint("movie_id", "genre_id", name="uq_movie_genre"),)

    movie_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    genre_id: Mapped[int] = mapped_column(
        ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True
    )

    movie: Mapped[Movie] = relationship(back_populates="genre_links")
    genre: Mapped[Genre] = relationship(back_populates="movie_links")
