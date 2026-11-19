"""Fetch the trained files the app needs when the repository does not carry them.

The course rule forbids large binaries in the repository, so `models/` and the
drift baselines are not tracked by git. A deployment therefore starts with the
code only and has to pull those files once, before the engine can be loaded:

    models/<tag>/{isolation_forest,logistic_regression,stacker,xgboost}.joblib
    results/<tag>/baseline_scores.npz

Two hosts are tried in order -- Hugging Face first, the GitHub release second --
so losing either one does not take the app down. A downloaded file is accepted
only when its SHA-256 matches the value recorded below: a truncated or replaced
response is rejected and the next source is tried, which is what makes the
fallback worth having.

Nothing here runs when the files are already on disk, so a local checkout and a
warm container pay only a few `stat` calls.
"""
from __future__ import annotations

import hashlib
import http.client
import os
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

CHUNK = 1 << 16                 # 64 KiB: small enough to show progress, big enough to be fast
TIMEOUT = 30.0                  # seconds per attempt; the largest file is ~2 MB
ATTEMPTS = 2                    # per source, before moving on to the next one
PAUSE = 1.0                     # seconds between attempts
RESUMES = 8                     # streams a slow CDN may drop before a file counts as lost
SUFFIX = ".part"                # a download in progress, never mistaken for a finished file
USER_AGENT = "FRAUDINTEL/1.0"   # Hugging Face answers 403 to the bare python-urllib agent


@dataclass(frozen=True)
class Artifact:
    """One file the app needs: where it belongs, how big it is, what it must hash to."""
    path: str                   # relative to the project root, forward slashes
    size: int
    sha256: str


@dataclass(frozen=True)
class Source:
    """A place to fetch from. `keep_dirs` says whether the URL repeats the folders."""
    name: str
    base: str
    keep_dirs: bool = True

    def url_for(self, artifact: Artifact) -> str:
        name = artifact.path.rsplit("/", 1)[-1]
        tail = artifact.path if self.keep_dirs else name
        return f"{self.base.rstrip('/')}/{tail}"


# The one run the app can use: `discover_runs` needs all three detectors and the
# stacker, and only this tag has them. Sizes and hashes come from the local files.
ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("models/leakage_safe_no_amount_to_balance_oldbalanceOrg/isolation_forest.joblib",
             2288653, "3b6c18ef6470db8b4ffb6789f47630a725b14104c200a39c4a1490b096c5ac41"),
    Artifact("models/leakage_safe_no_amount_to_balance_oldbalanceOrg/logistic_regression.joblib",
             3722, "b5ad4235ce042119f0e7a70b01cd81b274ce5726622e9fd86ddca74d4d26dca5"),
    Artifact("models/leakage_safe_no_amount_to_balance_oldbalanceOrg/stacker.joblib",
             1745, "019073f5ad1cd948115ceb615bc326c34476215e2415f4bcc68e2aeac5eaa466"),
    Artifact("models/leakage_safe_no_amount_to_balance_oldbalanceOrg/xgboost.joblib",
             1211096, "0242bdcaad5ce80cc74ada855778e7591810763c0fe70c3cca0f00fefe96d727"),
    Artifact("results/leakage_safe_no_amount_to_balance_oldbalanceOrg/baseline_scores.npz",
             1305780, "bbe72301a9d306a8f9473c58199c33b8fdc668b682cbb1e1fb074ecc1cbf85c8"),
)

# Hugging Face keeps the folder structure; a GitHub release asset is a flat file
# name, so the two need different URLs for the same artifact.
SOURCES: tuple[Source, ...] = (
    Source("Hugging Face", "https://huggingface.co/spaces/thecrack243/fraudintel/resolve/main"),
    Source("GitHub release",
           "https://github.com/thecrack243/PT22026/releases/download/artifacts-v1",
           keep_dirs=False),
)


class ArtifactsUnavailable(RuntimeError):
    """Every source failed: the app cannot score anything without these files."""


def missing(root: str | Path, artifacts: Iterable[Artifact] = ARTIFACTS) -> list[Artifact]:
    """The artifacts that are absent or the wrong size (so, almost certainly broken)."""
    root = Path(root)
    return [a for a in artifacts if _size(root / a.path) != a.size]


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


class _Restart(Exception):
    """The server ignored a Range request, so the file has to start again from zero."""


def _tolerate_windows_cert_store() -> None:
    """Ignore one unreadable certificate in the Windows store (same fix as run_local.py).

    Without it every HTTPS call on this machine dies with `SSLError: NOT_ENOUGH_DATA`
    while the store is being loaded. The hook only exists on Windows, so on the
    Linux container a deployment runs on this is a no-op.
    """
    original = getattr(ssl.SSLContext, "_load_windows_store_certs", None)
    if original is None or getattr(original, "_fraudintel_safe", False):
        return

    def safe_load(context, storename, purpose):
        try:
            return original(context, storename, purpose)
        except ssl.SSLError:
            return bytearray()

    safe_load._fraudintel_safe = True
    ssl.SSLContext._load_windows_store_certs = safe_load


def _stream(url: str, handle, timeout: float) -> None:
    """Append everything from the current file position to the end of the response.

    The offset becomes a Range header, which is what lets a dropped connection
    pick up where it stopped instead of downloading the file again.
    """
    offset = handle.tell()
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if offset and response.status != 206:      # 200 means it sent the whole file again
            raise _Restart(f"no range support (status {response.status})")
        for block in iter(lambda: response.read(CHUNK), b""):
            handle.write(block)


def _fetch(url: str, target: Path, artifact: Artifact, timeout: float,
           resumes: int = RESUMES) -> None:
    """Download one file, resuming dropped streams, and keep it only if it is exact."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + SUFFIX)
    try:
        with temporary.open("wb") as handle:
            restarted = False
            problem: Exception | None = None
            for _ in range(resumes + 1):
                if handle.tell() >= artifact.size:
                    break
                before = handle.tell()
                try:
                    _stream(url, handle, timeout)
                except _Restart:
                    if restarted:                  # a second refusal is a real failure
                        raise
                    restarted = True
                    handle.seek(0)
                    handle.truncate()
                    continue
                except (urllib.error.URLError, socket.timeout, OSError,
                        http.client.HTTPException) as exc:
                    if handle.tell() == before:    # nothing arrived: retrying here cannot help
                        raise
                    problem = exc                  # the stream dropped late; resume it
        received = temporary.stat().st_size
        if received != artifact.size:
            raise ValueError(f"expected {artifact.size} bytes, received {received}"
                             + (f" (last error: {problem})" if problem else ""))
        digest = _sha256(temporary)
        if digest != artifact.sha256:
            raise ValueError(f"sha256 mismatch ({digest[:12]}... != {artifact.sha256[:12]}...)")
        os.replace(temporary, target)        # atomic: a half file never gets the real name
    finally:
        temporary.unlink(missing_ok=True)


def ensure(root: str | Path,
           artifacts: Sequence[Artifact] = ARTIFACTS,
           sources: Sequence[Source] = SOURCES,
           timeout: float = TIMEOUT,
           attempts: int = ATTEMPTS,
           pause: float = PAUSE,
           log: Callable[[str], None] | None = None) -> list[str]:
    """Make sure every artifact is on disk, fetching the missing ones.

    Returns one short line per fetch that succeeded, for the caller to show.
    Raises ArtifactsUnavailable when a file could not be obtained from any
    source -- the app then has nothing to score with and must say so loudly.
    """
    root = Path(root)
    pending = missing(root, artifacts)
    if not pending:
        return []

    _tolerate_windows_cert_store()

    override = os.environ.get("FRAUDINTEL_ARTIFACTS_URL")
    ordered = ([Source("FRAUDINTEL_ARTIFACTS_URL", override)] if override else []) + list(sources)

    notes: list[str] = []
    for artifact in pending:
        target = root / artifact.path
        failures: dict[str, str] = {}            # last reason per source, in the order tried
        for source in ordered:
            url = source.url_for(artifact)
            for attempt in range(1, attempts + 1):
                try:
                    _fetch(url, target, artifact, timeout)
                except (urllib.error.URLError, socket.timeout, OSError, ValueError) as exc:
                    reason = getattr(exc, "reason", exc)
                    failures[source.name] = f"{type(exc).__name__}: {reason}"
                    if log is not None:
                        log(f"[bootstrap] {artifact.path} from {source.name} "
                            f"(attempt {attempt}/{attempts}) failed: {reason}")
                    if attempt < attempts:
                        time.sleep(pause)
                    continue
                notes.append(f"{artifact.path.rsplit('/', 1)[-1]} from {source.name}")
                break
            else:
                continue                         # this source is out, try the next one
            break                                # file is in place
        else:
            raise ArtifactsUnavailable(
                f"Could not fetch {artifact.path} from any source:\n  "
                + "\n  ".join(f"{name}: {reason}" for name, reason in failures.items())
                + "\n\nRun the training command to rebuild the file locally, or check the "
                  "links in app/bootstrap.py."
            )
    return notes


def summary(notes: Sequence[str]) -> str:
    """One line for the UI: how many files, from where, how large."""
    if not notes:
        return ""
    hosts = {note.rsplit(" from ", 1)[-1] for note in notes}
    megabytes = sum(a.size for a in ARTIFACTS) / (1024 * 1024)
    return (f"Fetched {len(notes)} trained file{'s' if len(notes) != 1 else ''} "
            f"({megabytes:.1f} MB) from {' + '.join(sorted(hosts))}")
