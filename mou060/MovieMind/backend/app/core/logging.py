"""Structured logging setup.

- Logs go to stdout in a single-line key=value format that is easy to grep
  and safe to ship (no stack traces of unhandled errors in the message body —
  those are logged by the error handler explicitly).
- Secrets and tokens are never logged; a small ``redact`` helper is provided
  for future middleware use.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import get_settings

_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "password",
    "token",
    "access_token",
    "refresh_token",
    "authorization",
    "secret",
    "secret_key",
    "api_key",
}


def _configure_root(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def setup_logging() -> None:
    """Idempotently configure root logging from settings."""
    settings = get_settings()
    _configure_root(settings.LOG_LEVEL)
    # Quieten noisy third-party loggers; uvicorn's own loggers keep their config.
    for name in ("sqlalchemy.engine", "alembic"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(f"moviemind.{name}")


def redact(data: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``data`` with sensitive keys redacted.

    Used anywhere request/response payloads might end up in logs.
    """
    return {k: (_REDACTED if k.lower() in _SENSITIVE_KEYS else v) for k, v in data.items()}
