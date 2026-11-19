"""Content-addressed experiment releases and verified reads of trusted local files.

Hashes detect accidental replacement and stale caches; they are not signatures
or permission to load untrusted pickle/joblib content.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

# The four files the app needs for one trained experiment.
RUNTIME_FILES = ("model_bundle.joblib", "test_metrics.csv", "permutation_importance.csv", "training_metadata.json")
# The manifest lists the SHA-256 hash (a file fingerprint) of each file above.
MANIFEST_NAME = "release_manifest.json"


class ArtifactReleaseError(ValueError):
    """The configured experiment does not match its declared release."""


@dataclass(frozen=True)
class ArtifactRelease:
    """The result of a release check.

    release_id is one fingerprint for the whole experiment. managed is True
    when a valid manifest file was found, and False for local files without one.
    """

    dataset_id: str
    release_id: str
    digests: dict[str, str | None]
    managed: bool


def file_digest(path: Path) -> str | None:
    """Return the SHA-256 hash of a file, or None if the file does not exist."""
    try:
        # Read in small pieces so this also works on Python 3.10 (file_digest needs 3.11).
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha.update(chunk)
        digest = sha.hexdigest()
    except FileNotFoundError:
        return None
    return digest


def capture_digests(folder: Path) -> dict[str, str | None]:
    """Hash all four runtime files in a folder. A missing file gets None."""
    return {name: file_digest(Path(folder) / name) for name in RUNTIME_FILES}


def release_identity(dataset_id: str, digests: dict) -> str:
    """Make one hash from the dataset id and all file hashes.

    If any file changes, this release ID changes too.
    """
    payload = {"schema_version": 1, "dataset_id": dataset_id, "files": digests}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_release(folder: Path, dataset_id: str, *, require_manifest: bool = False) -> ArtifactRelease:
    """Verify a declared release, or fingerprint legacy local preparation files."""
    folder = Path(folder)
    digests = capture_digests(folder)
    identity = release_identity(dataset_id, digests)
    manifest_path = folder / MANIFEST_NAME
    if not manifest_path.exists():
        if require_manifest:
            raise ArtifactReleaseError("Release manifest missing. Run scripts/verify_release.py --dataset " + dataset_id + " --write-manifests.")
        return ArtifactRelease(dataset_id, identity, digests, False)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The manifest is valid only if its structure is correct and every
        # hash is a 64-character hexadecimal text.
        valid = (
            isinstance(manifest, dict) and manifest.get("schema_version") == 1
            and manifest.get("dataset_id") == dataset_id
            and isinstance(manifest.get("files"), dict)
            and set(manifest["files"]) == set(RUNTIME_FILES)
            and all(isinstance(value, str) and len(value) == 64
                    and all(c in "0123456789abcdef" for c in value) for value in manifest["files"].values())
            and manifest.get("release_id") == release_identity(dataset_id, manifest["files"])
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ArtifactReleaseError("Release manifest is unreadable or invalid. Verify and republish the selected experiment.") from exc
    if not valid:
        raise ArtifactReleaseError("Release manifest schema, dataset identity or release ID is invalid.")
    # Compare the hash of each real file with the hash written in the manifest.
    changed = [name for name in RUNTIME_FILES if digests[name] != manifest["files"][name]]
    if changed:
        raise ArtifactReleaseError("Release files do not match their manifest: " + ", ".join(changed)
                                   + ". Verify the experiment before publishing a replacement manifest.")
    return ArtifactRelease(dataset_id, identity, digests, True)


def verified_bytes(path: Path, expected_sha256: str | None) -> bytes:
    """Read once and validate the exact bytes that the caller will deserialize."""
    content = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ArtifactReleaseError("Artifact changed while loading: " + Path(path).name + ". Retry with a consistent release.")
    return content


def publish_manifest(folder: Path, dataset_id: str, expected_digests: dict) -> ArtifactRelease:
    """Atomically publish the hash set already checked by the preflight command."""
    folder = Path(folder)
    digests = capture_digests(folder)
    if digests != expected_digests or any(value is None for value in digests.values()):
        raise ArtifactReleaseError("Runtime files changed or are missing. Repeat experiment validation before publishing.")
    metadata = json.loads(verified_bytes(folder / "training_metadata.json", digests["training_metadata.json"]))
    if not isinstance(metadata, dict) or metadata.get("dataset_id", "uci_household") != dataset_id:
        raise ArtifactReleaseError("Training metadata belongs to a different dataset.")
    manifest = {"schema_version": 1, "dataset_id": dataset_id, "release_id": release_identity(dataset_id, digests), "files": digests}
    # Write to a temporary file first, then rename it. This way the manifest
    # is never left half-written if the program stops in the middle.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, folder / MANIFEST_NAME)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return read_release(folder, dataset_id, require_manifest=True)
