"""Typed configuration objects.

All tunable behaviour lives in ``config/*.yaml``; this module turns those files
into frozen dataclasses so that no module reaches for a literal constant mid
function. Environment overrides (``MMSA_DEVICE``, ``MMSA_CACHE_DIR``,
``MMSA_LOG_LEVEL``, ``MMSA_DIMENSIONAL``) are applied at load time, which is what
makes the same code run on a laptop, a lab machine with a GPU, and Hugging Face
Spaces without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""``.../mej031`` -- resolved from this file, never from the shell's cwd."""

CONFIG_DIR: Path = PROJECT_ROOT / "config"

DEFAULT_ASR_ID = "openai/whisper-small.en"
DEFAULT_ACOUSTIC_ID = "superb/wav2vec2-base-superb-er"
DEFAULT_TEXT_ID = "bhadresh-savani/distilbert-base-uncased-emotion"
DEFAULT_DIMENSIONAL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


# Sections


@dataclass(frozen=True)
class ModelConfig:
    """Hugging Face Hub identifiers. Weights are never stored in this repo."""

    asr_id: str = DEFAULT_ASR_ID
    acoustic_id: str = DEFAULT_ACOUSTIC_ID
    text_id: str = DEFAULT_TEXT_ID
    dimensional_id: str = DEFAULT_DIMENSIONAL_ID
    # Repo-relative path to the small fitted head (`scripts/train_acoustic_probe.py`).
    # An empty string disables it and forces the zero-shot four-class fallback.
    acoustic_probe: str = "models/acoustic_probe.npz"

    def all_ids(self) -> tuple[str, ...]:
        """Every Hub id the application may touch."""
        return (self.asr_id, self.acoustic_id, self.text_id, self.dimensional_id)


@dataclass(frozen=True)
class ASRConfig:
    """Speech-recognition options."""

    backend: str = "transformers"  # "transformers" | "openai_whisper"
    language: str = "en"
    chunk_length_s: float = 30.0
    max_new_tokens: int = 200
    low_confidence_threshold: float = 0.35


@dataclass(frozen=True)
class FusionConfig:
    """Weights and thresholds for the fusion engine (see ``fusion.py``)."""

    mode: str = "credal"
    gamma: float = 0.75
    delta: float = 0.50
    epsilon: float = 1.0e-6
    alpha: float = 0.55
    ignorance_threshold: float = 0.50
    conflict_aligned: float = 0.15
    conflict_mild: float = 0.40
    text_evidence_floor: float = 0.02
    text_evidence_withhold: float = 0.15
    priors_acoustic: Mapping[str, float] = field(default_factory=dict)
    priors_text: Mapping[str, float] = field(default_factory=dict)

    def prior(self, modality: str) -> Mapping[str, float]:
        """Per-class modality prior for ``"acoustic"`` or ``"text"``."""
        return self.priors_acoustic if modality == "acoustic" else self.priors_text


@dataclass(frozen=True)
class StressConfig:
    """Stress-index coefficients, thresholds and feature wiring."""

    bias: float = -0.15
    weights: Mapping[str, float] = field(default_factory=dict)
    threshold_low: float = 0.35
    threshold_high: float = 0.65
    borderline_band: float = 0.05
    dimensional_enabled: bool = True
    proxy_valence: Mapping[str, float] = field(default_factory=dict)
    proxy_arousal: Mapping[str, float] = field(default_factory=dict)
    instability_features: tuple[str, ...] = ("f0_iqr", "rms_std", "pause_ratio", "zcr_std")
    prosody_reference_file: str = "config/prosody_ref.json"
    lexicon_file: str = "config/lexicon.yaml"
    lexicon_max_hits: int = 6


@dataclass(frozen=True)
class AppConfig:
    """Everything the pipeline and the interface need, in one object."""

    models: ModelConfig = field(default_factory=ModelConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    device: str = "cpu"
    cache_dir: str | None = None
    log_level: str = "INFO"
    seed: int = 1337
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0
    max_file_mb: float = 25.0
    project_root: Path = PROJECT_ROOT

    @property
    def allowed_extensions(self) -> tuple[str, ...]:
        """Accepted audio container extensions."""
        return (".wav", ".flac", ".mp3", ".ogg", ".m4a")

    def resolve(self, relative: str | Path) -> Path:
        """Resolve a repo-relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (self.project_root / p)


# Loading


def detect_device(preference: str = "auto") -> tuple[str, str]:
    """Return ``(device, reason)`` without importing torch when unnecessary."""
    preference = (preference or "auto").lower()
    if preference in {"cpu", "cuda", "mps"}:
        return preference, "explicitly configured"

    try:
        import torch  # local import: keeps the pure-maths modules torch-free
    except ImportError:  # pragma: no cover - exercised only on lean installs
        return "cpu", "torch not installed"

    if torch.cuda.is_available():
        return "cuda", "CUDA device available"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple Metal device available"
    return "cpu", "no accelerator detected"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_config(
    config_dir: str | Path | None = None,
    *,
    device: str | None = None,
) -> AppConfig:
    """Build an :class:`AppConfig` from ``config/*.yaml`` plus env overrides."""
    cfg_dir = Path(config_dir) if config_dir else CONFIG_DIR

    fusion_raw = _read_yaml(cfg_dir / "fusion.yaml")
    stress_raw = _read_yaml(cfg_dir / "stress.yaml")

    bands = fusion_raw.get("conflict_bands", {}) or {}
    priors = fusion_raw.get("priors", {}) or {}
    evidence = fusion_raw.get("text_evidence", {}) or {}
    fusion = FusionConfig(
        mode=str(fusion_raw.get("mode", "credal")),
        gamma=float(fusion_raw.get("gamma", 0.75)),
        delta=float(fusion_raw.get("delta", 0.50)),
        epsilon=float(fusion_raw.get("epsilon", 1e-6)),
        alpha=float(fusion_raw.get("alpha", 0.55)),
        ignorance_threshold=float(fusion_raw.get("ignorance_threshold", 0.50)),
        conflict_aligned=float(bands.get("aligned", 0.15)),
        conflict_mild=float(bands.get("mild", 0.40)),
        text_evidence_floor=float(evidence.get("floor", 0.02)),
        text_evidence_withhold=float(evidence.get("withhold_below", 0.15)),
        priors_acoustic={k: float(v) for k, v in (priors.get("acoustic") or {}).items()},
        priors_text={k: float(v) for k, v in (priors.get("text") or {}).items()},
    )

    thresholds = stress_raw.get("thresholds", {}) or {}
    dimensional = stress_raw.get("dimensional", {}) or {}
    proxy = stress_raw.get("proxy", {}) or {}
    prosody = stress_raw.get("prosody", {}) or {}
    lexicon = stress_raw.get("lexicon", {}) or {}

    dimensional_enabled = bool(dimensional.get("enabled", True))
    env_dimensional = os.environ.get("MMSA_DIMENSIONAL")
    if env_dimensional is not None:
        dimensional_enabled = env_dimensional.strip().lower() in {"1", "true", "yes", "on"}

    stress = StressConfig(
        bias=float(stress_raw.get("bias", -0.15)),
        weights={k: float(v) for k, v in (stress_raw.get("weights") or {}).items()},
        threshold_low=float(thresholds.get("low", 0.35)),
        threshold_high=float(thresholds.get("high", 0.65)),
        borderline_band=float(stress_raw.get("borderline_band", 0.05)),
        dimensional_enabled=dimensional_enabled,
        proxy_valence={k: float(v) for k, v in (proxy.get("valence") or {}).items()},
        proxy_arousal={k: float(v) for k, v in (proxy.get("arousal") or {}).items()},
        instability_features=tuple(prosody.get("instability_features", ())),
        prosody_reference_file=str(prosody.get("reference_file", "config/prosody_ref.json")),
        lexicon_file=str(lexicon.get("file", "config/lexicon.yaml")),
        lexicon_max_hits=int(lexicon.get("max_hits", 6)),
    )

    requested = device if device is not None else os.environ.get("MMSA_DEVICE", "auto")
    resolved_device, _reason = detect_device(requested)

    probe = os.environ.get("MMSA_ACOUSTIC_PROBE")
    if probe is None:
        probe = str(_read_yaml(cfg_dir / "models.yaml").get("acoustic_probe", "models/acoustic_probe.npz"))

    return AppConfig(
        models=ModelConfig(
            asr_id=str(_read_yaml(cfg_dir / "fusion.yaml").get("asr_id", DEFAULT_ASR_ID)),
            acoustic_id=DEFAULT_ACOUSTIC_ID,
            text_id=DEFAULT_TEXT_ID,
            dimensional_id=str(dimensional.get("model_id", DEFAULT_DIMENSIONAL_ID)),
            acoustic_probe=probe,
        ),
        fusion=fusion,
        stress=stress,
        device=resolved_device,
        cache_dir=os.environ.get("MMSA_CACHE_DIR") or None,
        log_level=os.environ.get("MMSA_LOG_LEVEL", "INFO").upper(),
        seed=int(os.environ.get("MMSA_SEED", "1337")),
    )
