"""Sliding-window rate limiter unit tests (T9.2)."""

from __future__ import annotations

import time

from app.core.rate_limit import SlidingWindowRateLimiter


def test_allows_up_to_max_requests():
    limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is True
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False  # 4th blocked


def test_keys_are_independent():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("a") is True
    assert limiter.allow("b") is True  # different key unaffected
    assert limiter.allow("a") is False


def test_reset_clears_state():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    limiter.reset()
    assert limiter.allow("a") is True


def test_window_expires_over_time():
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=0.05)
    assert limiter.allow("a") is True
    assert limiter.allow("a") is False
    time.sleep(0.1)
    assert limiter.allow("a") is True  # window has slid
