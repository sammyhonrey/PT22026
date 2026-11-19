"""Shared pytest fixtures.

Uses an in-memory-backed test settings profile so tests never touch the
configured production/development database.
"""

from __future__ import annotations

import pytest

from app.api.dependencies import reset_rate_limiters


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_rate_limiters() -> None:
    """Reset in-process rate-limit state before every test for isolation."""
    reset_rate_limiters()
