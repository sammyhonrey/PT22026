"""
Safe cleanup for the MP3 conversion cache.

Only *.mp3 files inside the cache directory are ever considered. Original
WAV files, the Audio Library database and job folders are never touched,
so cleanup cannot remove audio the application still references.
"""

from __future__ import annotations

import time
from pathlib import Path

MAX_CACHE_AGE_DAYS = 7
MAX_CACHE_TOTAL_MB = 256


def cleanup_mp3_cache(
    cache_dir: str | Path,
    max_age_days: int = MAX_CACHE_AGE_DAYS,
    max_total_mb: int = MAX_CACHE_TOTAL_MB,
    now: float | None = None,
) -> dict:
    """
    Delete stale or excess cached MP3 files.

    Policy:
    - only regular ``*.mp3`` files directly inside ``cache_dir`` are eligible;
    - files older than ``max_age_days`` are removed;
    - if the remaining size exceeds ``max_total_mb``, the oldest files are
      removed until it fits.

    Returns a summary dict for logging/tests.
    """

    directory = Path(cache_dir)
    current = time.time() if now is None else now
    cutoff = current - max_age_days * 86400
    max_bytes = max(0, int(float(max_total_mb) * 1024 * 1024))

    removed: list[str] = []
    kept = 0
    freed = 0

    if not directory.is_dir():
        return {
            "removed": removed,
            "kept": kept,
            "freed_bytes": freed,
            "cache_dir": str(directory),
        }

    candidates: list[tuple[float, Path, int]] = []

    for path in directory.iterdir():
        try:
            if not path.is_file() or path.suffix.lower() != ".mp3":
                continue

            stat = path.stat()
        except OSError:
            continue

        if stat.st_mtime < cutoff:
            try:
                path.unlink()
                removed.append(path.name)
                freed += stat.st_size
                continue
            except OSError:
                kept += 1
                candidates.append((stat.st_mtime, path, stat.st_size))
                continue

        kept += 1
        candidates.append((stat.st_mtime, path, stat.st_size))

    total = sum(size for _, _, size in candidates)

    if total > max_bytes:
        for _, path, size in sorted(candidates, key=lambda item: item[0]):
            if total <= max_bytes:
                break

            try:
                path.unlink()
                removed.append(path.name)
                total -= size
                kept -= 1
                freed += size
            except OSError:
                continue

    return {
        "removed": removed,
        "kept": kept,
        "freed_bytes": freed,
        "cache_dir": str(directory),
    }
