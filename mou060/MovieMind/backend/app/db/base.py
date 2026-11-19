

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, MetaData
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utcnow() -> datetime:
    """Naive-UTC timestamp (consistent across DB drivers)."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """All MovieMind models inherit from this base."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class UUIDPrimaryKeyMixin:
    """UUID primary key, generated in Python so ids exist before flush."""

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4, index=False)


class TimestampMixin:
    """created_at / updated_at in UTC, maintained by SQLAlchemy."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), default=utcnow, onupdate=utcnow, nullable=False
    )
