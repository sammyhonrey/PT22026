"""Frozen result types shared by the pipeline, the interface and the report.

These dataclasses are the single source of truth for every number the system
displays. The UI renders them; ``scripts/evaluate_*.py`` serialises them; the
report's figures are built from those JSON files. Nothing is recomputed twice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

CANONICAL_LABELS: tuple[str, ...] = ("happy", "sad", "angry", "fear", "disgust", "neutral")
"""Target emotion ontology, in fixed order. Bars, radars, confusion matrices and
JSON keys all follow this order so that figures are comparable across runs."""

Modality = Literal["acoustic", "text", "fused"]
Agreement = Literal["aligned", "mild_conflict", "strong_conflict", "indeterminate"]


@dataclass(frozen=True)
class AudioInfo:
    """Descriptive statistics of the decoded waveform."""

    path: str
    duration_s: float
    sample_rate: int
    peak_dbfs: float
    rms_dbfs: float
    snr_db: float | None
    clipping_ratio: float
    is_speech_suspected: bool


@dataclass(frozen=True)
class ASRResult:
    """Transcript plus the decoder's own uncertainty about it."""

    text: str
    confidence: float
    n_tokens: int
    duration_s: float
    backend: str


@dataclass(frozen=True)
class DimensionalResult:
    """Arousal/dominance/valence from the optional dimensional head, in 0-1."""

    arousal: float
    dominance: float
    valence: float
    model_id: str


@dataclass(frozen=True)
class BranchResult:
    """One modality's opinion, in both its native and the canonical vocabulary.

    ``mapped_probs`` sums to ``coverage``, which is at most 1.0. It is *not*
    renormalised: the residual is the mass this branch structurally cannot
    express, and the fusion engine consumes it as ignorance rather than letting it
    silently inflate the remaining classes.
    """

    modality: Literal["acoustic", "text"]
    raw_labels: tuple[str, ...]
    raw_probs: tuple[float, ...]
    mapped_probs: Mapping[str, float]
    coverage: float
    confidence: float
    weight: float
    per_class_weight: Mapping[str, float]
    top_label: str
    latency_ms: float


@dataclass(frozen=True)
class StressResult:
    """Composite stress index with a full contributor decomposition."""

    score: float
    band: Literal["Low", "Medium", "High"]
    band_label: str
    contributors: Mapping[str, float]
    features: Mapping[str, float]


@dataclass(frozen=True)
class AnalysisResult:
    """Everything one request produced, immutable, JSON-serialisable."""

    request_id: str
    created_at: str
    audio: AudioInfo
    asr: ASRResult
    acoustic: BranchResult
    text: BranchResult
    fused_probs: Mapping[str, float]
    fused_emotion: str
    fused_confidence: float
    entropy: float
    ignorance: float
    conflict_index: float | None
    agreement: Agreement
    stress: StressResult
    narrative: str
    narrative_short: str
    timings_ms: Mapping[str, float]
    dimensional: DimensionalResult | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    text_evidence: float = 1.0
    """Lexical evidence the transcript carried (``mmsa.evidence``), in ``[0, 1]``.

    Fusion multiplies the text branch's reliability by this value. The interface and
    the report both read it, because "the text channel had nothing to say" is a fact
    about the request that a user deserves to see.
    """

    # -- serialisation ------------------------------------------------------- #

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for JSON output (tuples become lists)."""
        return asdict(self)

    @property
    def top_two(self) -> tuple[str, str]:
        """The two most probable emotions, most probable first."""
        ordered = sorted(self.fused_probs.items(), key=lambda kv: kv[1], reverse=True)
        first = ordered[0][0]
        second = ordered[1][0] if len(ordered) > 1 else first
        return first, second

    @property
    def dominant_modality(self) -> Literal["acoustic", "text"]:
        """Which branch carried more fused weight overall."""
        return "acoustic" if self.acoustic.weight >= self.text.weight else "text"
