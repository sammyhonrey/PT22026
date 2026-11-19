

from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, Numeric, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.models.movie import Movie


class Rating(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_user_movie_rating"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    movie_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # 0.5–5.0 in 0.5 increments -> one decimal place, max value 5.0 (2 digits).
    score: Mapped[float] = mapped_column(Numeric(2, 1), nullable=False)

    # Read-only navigation for serialization; no schema change. Load eagerly via
    # selectinload in the repository to avoid N+1 queries.
    movie: Mapped[Movie] = relationship(Movie)
