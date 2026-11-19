"""Thread-safe, in-process sliding-window rate limiter.

Used to throttle login/register (and demo endpoints) before public deployment.
In-process storage is sufficient for a single-instance deployment; a shared
store (e.g. Redis) would be required for horizontal scaling — noted here so the
limitation is explicit, not silently assumed.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowRateLimiter:
    """Allow at most ``max_requests`` per ``window_seconds`` per key."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._hits: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """Return True if the request is allowed, and record it.

        Evicts timestamps older than the window before deciding, so memory is
        bounded by the number of active keys times ``max_requests``.
        """
        now = time.monotonic()
        with self._lock:
            window = self._hits[key]
            while window and now - window[0] >= self._window:
                window.popleft()
            if len(window) >= self._max:
                return False
            window.append(now)
            return True

    def reset(self) -> None:
        """Clear all recorded hits (used by tests)."""
        with self._lock:
            self._hits.clear()
