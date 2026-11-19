"""Alembic environment.

Wires migrations to the application's settings (``DATABASE_URL`` from the
environment/``.env``) and the SQLAlchemy metadata of all MovieMind models.
"""

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context
from app.core.config import get_settings
from app.db.base import Base
from app.db.models import (  # noqa: F401 — import to populate Base.metadata
    Genre,
    Movie,
    MovieGenre,
    Rating,
    RefreshSession,
    User,
    UserRole,
    WatchlistItem,
)

# Alembic Config object (values in alembic.ini).
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate.
target_metadata = Base.metadata

# Database URL comes from app settings (never hardcoded in alembic.ini).
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)


def _build_url():
    # Escape '%' in the URL so configparser interpolation does not mangle it.
    return config.get_main_option("sqlalchemy.url").replace("%", "%%")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    context.configure(
        url=_build_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live engine."""
    connectable = create_engine(
        settings.DATABASE_URL,
        poolclass=pool.NullPool,
        # Bounded timeout (see app/db/session.py).
        connect_args={"connect_timeout": 5},
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
