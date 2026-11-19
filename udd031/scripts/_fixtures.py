"""Deterministic fixtures shared by the analysis scripts.

Branch input vectors here are illustrative, not measured; everything after them --
projection, reliability weighting, fusion, conflict, stress -- is the real code.
Figures built from this fixture say "illustrative" in their captions.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from mmsa.ontology import Ontology
from mmsa.pipeline import AnalysisPipeline
from mmsa.schemas import ASRResult, BranchResult, DimensionalResult

__all__ = [
    "TEXT_STUB",
    "write_tone",
    "FixturePipeline",
    "build_fixture_pipeline",
    "fixture_emotions",
]

TEXT_STUB = "I am fine, but the deadline has been difficult this week."

# Six illustrative input profiles, one per canonical emotion, used to exercise the
# fusion maths across the whole target space rather than only its middle.
_ACOUSTIC_PROFILE: dict[str, tuple[float, float, float, float]] = {
    # superb/wav2vec2-base-superb-er order: ang, hap, neu, sad
    "angry": (0.62, 0.10, 0.16, 0.12),
    "happy": (0.08, 0.64, 0.18, 0.10),
    "neutral": (0.12, 0.14, 0.58, 0.16),
    "sad": (0.10, 0.08, 0.20, 0.62),
    "fear": (0.24, 0.16, 0.34, 0.26),
    "disgust": (0.30, 0.12, 0.38, 0.20),
}

_TEXT_PROFILE: dict[str, tuple[float, ...]] = {
    # bhadresh-savani order: sadness, joy, love, anger, fear, surprise
    "angry": (0.10, 0.06, 0.05, 0.62, 0.12, 0.05),
    "happy": (0.06, 0.58, 0.20, 0.05, 0.05, 0.06),
    "neutral": (0.18, 0.16, 0.14, 0.16, 0.18, 0.18),
    "sad": (0.56, 0.06, 0.12, 0.08, 0.12, 0.06),
    "fear": (0.16, 0.05, 0.06, 0.10, 0.56, 0.07),
    "disgust": (0.34, 0.05, 0.06, 0.34, 0.14, 0.07),
}

_ACOUSTIC_LABELS = ("ang", "hap", "neu", "sad")
_TEXT_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")


def fixture_emotions() -> tuple[str, ...]:
    """Canonical emotions the fixture can represent."""
    return tuple(_ACOUSTIC_PROFILE.keys())


def write_tone(
    path: Path,
    *,
    seconds: float = 3.0,
    sample_rate: int = 16_000,
    f0: float = 118.0,
    vibrato_hz: float = 0.7,
    syllable_hz: float = 3.1,
    noise: float = 0.004,
    seed: int = 7,
) -> Path:
    """Write a deterministic speech-like test signal as 16-bit mono PCM.

    A harmonic stack with gentle vibrato, gated into syllables, plus a noise floor.
    It is not speech and is never presented as speech -- it exists so that decode,
    prosody and figure generation can run on any machine without a dataset.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(seconds * sample_rate)
    t = np.arange(n) / sample_rate
    freq = f0 + 12.0 * np.sin(2 * np.pi * vibrato_hz * t)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    carrier = sum(np.sin(k * phase) / k for k in range(1, 7))
    gate = 0.5 * (1.0 + np.sin(2 * np.pi * syllable_hz * t))
    signal = 0.35 * carrier * np.clip(gate, 0.0, 1.0)
    signal = signal + noise * np.random.default_rng(seed).standard_normal(n)

    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


class FixturePipeline(AnalysisPipeline):
    """Pipeline whose branches return the illustrative profiles above.

    Subclassing keeps the real ``analyse`` flow -- validation, prosody, fusion,
    stress, narration, timing -- so figures reflect production behaviour.
    """

    def __init__(self, cfg, ontology: Ontology, emotion: str = "neutral") -> None:
        super().__init__(cfg, ontology=ontology, asr=_StubASR(), acoustic=_StubAcoustic(ontology, emotion),
                         text=_StubText(ontology, emotion))
        self._emotion = emotion


class _StubASR:
    backend = "fixture"

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        return ASRResult(
            text=TEXT_STUB,
            confidence=0.86,
            n_tokens=len(TEXT_STUB.split()),
            duration_s=float(len(audio) / max(1, sample_rate)),
            backend=self.backend,
        )

    def warmup(self) -> None:
        return None


class _StubAcoustic:
    def __init__(self, ontology: Ontology, emotion: str) -> None:
        self._ontology = ontology
        self._emotion = emotion

    def predict(self, audio: np.ndarray, sample_rate: int) -> BranchResult:
        probs = _ACOUSTIC_PROFILE[self._emotion]
        mapped, coverage = self._ontology.map_probs("acoustic", _ACOUSTIC_LABELS, probs)
        return BranchResult(
            modality="acoustic",
            raw_labels=_ACOUSTIC_LABELS,
            raw_probs=probs,
            mapped_probs=mapped,
            coverage=coverage,
            confidence=max(probs),
            weight=0.5,
            per_class_weight={},
            top_label=max(mapped, key=mapped.get),
            latency_ms=0.0,
        )

    def warmup(self) -> None:
        return None

    def predict_dimensional(self, audio: np.ndarray, sample_rate: int) -> DimensionalResult:
        return DimensionalResult(arousal=0.55, dominance=0.50, valence=0.45, model_id="fixture")


class _StubText:
    def __init__(self, ontology: Ontology, emotion: str) -> None:
        self._ontology = ontology
        self._emotion = emotion

    def predict(self, text: str) -> BranchResult:
        probs = _TEXT_PROFILE[self._emotion]
        mapped, coverage = self._ontology.map_probs("text", _TEXT_LABELS, probs)
        return BranchResult(
            modality="text",
            raw_labels=_TEXT_LABELS,
            raw_probs=probs,
            mapped_probs=mapped,
            coverage=coverage,
            confidence=max(probs),
            weight=0.5,
            per_class_weight={},
            top_label=max(mapped, key=mapped.get),
            latency_ms=0.0,
        )

    def warmup(self) -> None:
        return None


def build_fixture_pipeline(cfg, emotion: str = "neutral") -> FixturePipeline:
    """Construct a fixture pipeline for ``emotion``."""
    ontology = Ontology.from_yaml(cfg.resolve("config/ontology.yaml"))
    return FixturePipeline(cfg, ontology, emotion)
