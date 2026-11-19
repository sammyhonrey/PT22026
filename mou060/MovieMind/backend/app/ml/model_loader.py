"""Model artifact loader.

Loads the real NCF artifact bundle once and validates it before any inference
is allowed. The bundle is documented in ``ARTIFACTS.md``; the actual filenames
and shapes there are the source of truth (``model_config.json`` — not
``config.json``).

Security/compatibility rules enforced here:
- ``torch.load(..., map_location="cpu", weights_only=True)`` — never load an
  untrusted pickle.
- The architecture is reconstructed from ``model_config.json``, never hardcoded.
- Mapping indices are validated against the embedding sizes before use.
- The checkpoint's ``state_dict`` keys must match the model exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.ml.model_architecture import MovieMindNCF

# Files required for the NCF bundle (movies.json is optional here; the import
# script and demo recommender consume it separately).
REQUIRED_FILES = (
    "model_config.json",
    "moviemind_ncf.pt",
    "user_mapping.json",
    "movie_mapping.json",
)

VALIDATION_KEYS = {
    "num_users": int,
    "num_movies": int,
    "embedding_dim": int,
    "mlp_hidden_dims": list,
    "dropout": (int, float),
}


class ModelArtifactError(RuntimeError):
    """Raised when the artifact bundle is missing, malformed, or incompatible.

    The message is safe to surface in API diagnostics (no secrets or paths).
    """


@dataclass(frozen=True)
class LoadedModel:
    """A validated, ready-for-inference NCF model and its metadata."""

    model: MovieMindNCF
    config: dict[str, Any]
    user_mapping: dict[str, int]
    movie_mapping: dict[str, int]
    version: str
    source_dir: str
    device: str = "cpu"

    def __post_init__(self) -> None:
        # Frozen dataclass: switch the module to eval mode once, here.
        self.model.eval()


def _read_json(path: Path, what: str) -> dict[str, Any] | list[Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise ModelArtifactError(f"Missing {what} file: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ModelArtifactError(f"Invalid JSON in {what} file: {path.name}") from exc


def _validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ModelArtifactError("model_config.json must be a JSON object")
    for key, expected in VALIDATION_KEYS.items():
        if key not in config:
            raise ModelArtifactError(f"model_config.json is missing required key: {key}")
        value = config[key]
        if not isinstance(value, expected):
            raise ModelArtifactError(f"model_config.json key {key!r} has wrong type")
        if isinstance(value, bool) or (isinstance(value, (int, float)) and value < 0):
            raise ModelArtifactError(f"model_config.json key {key!r} has invalid value")
    if not config["mlp_hidden_dims"] or not all(
        isinstance(d, int) and d > 0 for d in config["mlp_hidden_dims"]
    ):
        raise ModelArtifactError("model_config.json mlp_hidden_dims must be non-empty ints")
    if not 0.0 <= float(config["dropout"]) <= 1.0:
        raise ModelArtifactError("model_config.json dropout must be in [0, 1]")


def _validate_mapping(
    mapping: dict[str, int], size: int, what: str
) -> dict[str, int]:
    if not isinstance(mapping, dict):
        raise ModelArtifactError(f"{what} must be a JSON object")
    normalized: dict[str, int] = {}
    for raw_id, idx in mapping.items():
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ModelArtifactError(f"{what} has non-integer index for id {raw_id!r}")
        if idx < 0 or idx >= size:
            raise ModelArtifactError(
                f"{what} index {idx} for id {raw_id!r} is out of range [0, {size})"
            )
        normalized[str(raw_id)] = idx
    return normalized


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except FileNotFoundError as exc:
        raise ModelArtifactError(f"Missing checkpoint file: {path.name}") from exc
    except Exception as exc:  # noqa: BLE001 — surface any load failure clearly
        raise ModelArtifactError(
            f"Failed to load checkpoint {path.name}: {type(exc).__name__}"
        ) from exc
    if not isinstance(state, dict):
        raise ModelArtifactError("Checkpoint is not a state_dict (expected a dict of tensors)")
    return state


def load_model(artifact_dir: str | Path, device: str = "cpu") -> LoadedModel:
    """Load and validate the NCF bundle from ``artifact_dir``.

    Raises :class:`ModelArtifactError` on any missing/incompatible artifact.
    """
    base = Path(artifact_dir)
    if not base.is_dir():
        raise ModelArtifactError(f"Model artifact directory not found: {base.name}")

    config = _read_json(base / "model_config.json", "config")
    if not isinstance(config, dict):
        raise ModelArtifactError("model_config.json must be a JSON object")
    _validate_config(config)

    user_mapping = _read_json(base / "user_mapping.json", "user mapping")
    movie_mapping = _read_json(base / "movie_mapping.json", "movie mapping")
    if not isinstance(user_mapping, dict) or not isinstance(movie_mapping, dict):
        raise ModelArtifactError("Mapping files must be JSON objects")

    num_users = config["num_users"]
    num_movies = config["num_movies"]
    user_mapping = _validate_mapping(user_mapping, num_users, "user_mapping")
    movie_mapping = _validate_mapping(movie_mapping, num_movies, "movie_mapping")

    model = MovieMindNCF(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=config["embedding_dim"],
        mlp_hidden_dims=tuple(config["mlp_hidden_dims"]),
        dropout=float(config["dropout"]),
    )

    state = _load_checkpoint(base / "moviemind_ncf.pt")
    expected = set(model.state_dict().keys())
    actual = set(state.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append(f"missing keys: {missing}")
        if extra:
            detail.append(f"unexpected keys: {extra}")
        raise ModelArtifactError(
            "Checkpoint does not match model architecture: " + "; ".join(detail)
        )

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    version = str(config.get("model_version", "unknown"))

    return LoadedModel(
        model=model,
        config=config,
        user_mapping=user_mapping,
        movie_mapping=movie_mapping,
        version=version,
        source_dir=str(base),
        device=device,
    )
