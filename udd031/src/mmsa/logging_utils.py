"""Structured logging, request correlation and stage timing.

Two rules the rest of the codebase depends on:

* nothing in :mod:`mmsa` prints -- everything goes through :func:`get_logger`;
* every request carries a ``request_id`` so a UI error message can be matched to
  exactly one traceback in the log file.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar("mmsa_request_id", default="-")

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | req=%(request_id)s | %(message)s"


class _RequestIdFilter(logging.Filter):
    """Injects the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102 - logging API
        record.request_id = _REQUEST_ID.get()
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger; idempotent across repeated imports."""
    logger = logging.getLogger(name)
    if getattr(logger, "_mmsa_configured", False):
        return logger

    level = os.environ.get("MMSA_LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_RequestIdFilter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = False
    logger._mmsa_configured = True  # type: ignore[attr-defined]
    return logger


def new_request_id() -> str:
    """Short, human-quotable identifier (8 hex chars)."""
    return uuid.uuid4().hex[:8]


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind a request id for the duration of the block."""
    rid = request_id or new_request_id()
    token = _REQUEST_ID.set(rid)
    try:
        yield rid
    finally:
        _REQUEST_ID.reset(token)


class Timer:
    """Wall-clock stage timer.

    ``with timer.stage("asr"): ...`` appends ``{"asr": milliseconds}``. Stages may
    repeat; the last measurement for a name wins, which is what a per-request
    breakdown wants.
    """

    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure one stage of the pipeline."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = round((time.perf_counter() - start) * 1000.0, 2)

    def add(self, name: str, milliseconds: float) -> None:
        """Record an externally measured duration."""
        self.timings_ms[name] = round(float(milliseconds), 2)

    @property
    def total_ms(self) -> float:
        """Sum of all recorded stages (excludes unmeasured overhead)."""
        return round(sum(self.timings_ms.values()), 2)


def log_summary(logger: logging.Logger, payload: dict[str, Any]) -> None:
    """Emit a compact one-line summary of a completed analysis."""
    logger.info(
        "analysis | fused=%s p=%.3f stress=%.3f(%s) conflict=%s agreement=%s ms=%.0f",
        payload.get("fused_emotion"),
        float(payload.get("fused_confidence", 0.0)),
        float(payload.get("stress_score", 0.0)),
        payload.get("stress_band"),
        payload.get("conflict_index"),
        payload.get("agreement"),
        float(payload.get("total_ms", 0.0)),
    )
