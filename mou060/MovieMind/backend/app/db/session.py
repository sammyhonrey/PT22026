

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """Return the cached SQLAlchemy engine.

    - Pre-ping so stale pooled connections are replaced transparently.
    - Pool limits kept small; this is a modular monolith, not a worker farm.
    """
    settings = get_settings()
    return create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=1800,
        future=True,
        # Critical on Windows: a firewalled/dead DB port times out silently
        # instead of refusing, so every connection needs a bounded timeout.
        connect_args={"connect_timeout": 5},
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return the cached session factory."""
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Session:
    """FastAPI dependency: one session per request, always closed on exit."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
