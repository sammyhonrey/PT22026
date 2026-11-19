"""User account model."""

from __future__ import annotations

import enum

from sqlalchemy import Boolean, CheckConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class UserRole(enum.StrEnum):
    USER = "user"
    ADMIN = "admin"


_ROLE_VALUES = ", ".join(f"'{r.value}'" for r in UserRole)


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"
    __table_args__ = (
        # DB-level guard; app-level validation lives in the Pydantic schema.
        CheckConstraint(f"role IN ({_ROLE_VALUES})", name="role"),
    )

    # Normalized (lowercased) on write; uniqueness enforced here and in Pydantic.
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    # Optional for now; unique when present.
    username: Mapped[str | None] = mapped_column(String(50), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(100))
    # Stored as lowercase string ('user'/'admin'); UserRole.USER is itself a str
    # so `role=UserRole.USER` binds correctly.
    role: Mapped[str] = mapped_column(
        String(20),
        default=UserRole.USER.value,
        server_default=UserRole.USER.value,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
