"""End-to-end orchestration tests using injected fakes.

These run offline, in under a second, with no model downloads -- which is the
point: the ordering, error translation, fusion wiring and narrative assembly are
all exercised here, so when the real models are loaded the only remaining risk is
the models themselves.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mmsa.errors import AudioTooShortError, InferenceError
from mmsa.pipeline import AnalysisPipeline
from mmsa.schemas import ASRResult, BranchResult, DimensionalResult

ACOUSTIC = ["ang", "hap", "neu", "sad"]
TEXT = ["joy", "love", "anger", "sadness", "fear", "surprise"]

PROSODY_REFERENCE = {
    "f0_iqr": {"median": 40.0, "iqr": 25.0},
    "rms_std": {"median": 6.0, "iqr": 3.0},
    "pause_ratio": {"median": 0.25, "iqr": 0.15},
    "zcr_std": {"median": 0.04, "iqr": 0.02},
}

LEXICON = {
    "negative": ["stressed", "tired", "deadline"],
    "positive": ["calm", "fine"],
    "intensifiers": ["very"],
    "negators": ["not"],
}


class FakeASR:
    name = "fake-asr"

    def __init__(self, text: str = "I am very tired and stressed about the deadline", confidence: float = 0.91):
        self.text = text
        self.confidence = confidence
        self.called = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        self.called += 1
        return ASRResult(
            text=self.text,
            confidence=self.confidence,
            n_tokens=len(self.text.split()),
            duration_s=len(audio) / sample_rate,
            backend=self.name,
        )


class FakeAcoustic:
    def __init__(self, ontology, probs=(0.10, 0.05, 0.10, 0.75), dimensional=True):
        self.ontology = ontology
        self.probs = list(probs)
        self.dimensional = dimensional
        self.called = 0

    def predict(self, audio: np.ndarray, sample_rate: int) -> BranchResult:
        self.called += 1
        mapped, coverage = self.ontology.map_probs("acoustic", ACOUSTIC, self.probs)
        return BranchResult(
            modality="acoustic",
            raw_labels=tuple(ACOUSTIC),
            raw_probs=tuple(self.probs),
            mapped_probs=mapped,
            coverage=coverage,
            confidence=0.42,
            weight=0.0,
            per_class_weight={},
            top_label=max(mapped.items(), key=lambda kv: kv[1])[0],
            latency_ms=12.0,
        )

    def predict_dimensional(self, audio: np.ndarray, sample_rate: int):
        if not self.dimensional:
            return None
        return DimensionalResult(arousal=0.7, dominance=0.3, valence=0.25, model_id="fake-dim")


class FakeText:
    def __init__(self, ontology, probs=(0.02, 0.03, 0.05, 0.60, 0.25, 0.05)):
        self.ontology = ontology
        self.probs = list(probs)
        self.called = 0

    def predict(self, text: str) -> BranchResult:
        self.called += 1
        mapped, coverage = self.ontology.map_probs("text", TEXT, self.probs)
        return BranchResult(
            modality="text",
            raw_labels=tuple(TEXT),
            raw_probs=tuple(self.probs),
            mapped_probs=mapped,
            coverage=coverage,
            confidence=0.55,
            weight=0.0,
            per_class_weight={},
            top_label=max(mapped.items(), key=lambda kv: kv[1])[0],
            latency_ms=15.0,
        )


def build_fake_pipeline(project_root, ontology, app_config, **overrides):
    asr = overrides.pop("asr", FakeASR())
    acoustic = overrides.pop("acoustic", FakeAcoustic(ontology))
    text = overrides.pop("text", FakeText(ontology))
    return (
        AnalysisPipeline(
            app_config,
            ontology=ontology,
            asr=asr,
            acoustic=acoustic,
            text=text,
            prosody_reference=PROSODY_REFERENCE,
            lexicon=LEXICON,
        ),
        asr,
        acoustic,
        text,
    )


@pytest.fixture
def wav_path(tmp_path):
    sr = 16_000
    t = np.linspace(0.0, 2.0, sr * 2, endpoint=False)
    rng = np.random.default_rng(11)
    signal = (0.4 * np.sin(2 * np.pi * 200 * t) + 0.02 * rng.standard_normal(t.size)).astype(np.float32)
    path = tmp_path / "clip.wav"
    sf.write(str(path), signal, sr)
    return str(path)


# Happy path


def test_full_analysis_produces_a_coherent_result(project_root, ontology, app_config, wav_path):
    pipeline, asr, acoustic, text = build_fake_pipeline(project_root, ontology, app_config)
    result = pipeline.analyse(wav_path)

    # Every branch ran exactly once.
    assert (asr.called, acoustic.called, text.called) == (1, 1, 1)

    # The fused vector is a proper distribution over the canonical classes.
    assert set(result.fused_probs) == {"happy", "sad", "angry", "fear", "disgust", "neutral"}
    assert sum(result.fused_probs.values()) == pytest.approx(1.0)
    assert result.fused_emotion in result.fused_probs
    assert result.fused_confidence == pytest.approx(max(result.fused_probs.values()))

    # Uncertainty fields are present and bounded.
    assert 0.0 <= result.entropy <= 1.0
    assert 0.0 <= result.ignorance <= 1.0
    assert result.conflict_index is not None
    assert 0.0 <= result.conflict_index <= 1.0
    assert result.agreement in {"aligned", "mild_conflict", "strong_conflict", "indeterminate"}

    # Stress is decomposed and self-consistent.
    assert 0.0 <= result.stress.score <= 1.0
    assert result.stress.band in {"Low", "Medium", "High"}
    assert "negative_valence" in result.stress.contributors


def test_timings_cover_every_stage(project_root, ontology, app_config, wav_path):
    pipeline, *_ = build_fake_pipeline(project_root, ontology, app_config)
    result = pipeline.analyse(wav_path)

    for stage in ("audio", "asr", "acoustic", "text", "prosody", "fusion", "stress", "narrative", "total"):
        assert stage in result.timings_ms, f"missing timing for {stage}"
        assert result.timings_ms[stage] >= 0.0
    assert result.timings_ms["total"] >= result.timings_ms["audio"]


def test_narrative_mentions_the_verdict(project_root, ontology, app_config, wav_path):
    pipeline, *_ = build_fake_pipeline(project_root, ontology, app_config)
    result = pipeline.analyse(wav_path)

    assert result.fused_emotion in result.narrative.lower()
    assert len(result.narrative) > 60
    assert result.stress.band.lower() in result.narrative.lower() or "stress" in result.narrative.lower()
    assert result.narrative_short


def test_result_is_serialisable(project_root, ontology, app_config, wav_path):
    import json

    pipeline, *_ = build_fake_pipeline(project_root, ontology, app_config)
    result = pipeline.analyse(wav_path)
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["fused_emotion"] == result.fused_emotion
    assert payload["audio"]["sample_rate"] == 16_000


def test_analysis_is_deterministic(project_root, ontology, app_config, wav_path):
    pipeline, *_ = build_fake_pipeline(project_root, ontology, app_config)
    first = pipeline.analyse(wav_path)
    second = pipeline.analyse(wav_path)
    assert first.fused_probs == second.fused_probs
    assert first.stress.score == second.stress.score


def test_dominant_modality_is_reported(project_root, ontology, app_config, wav_path):
    pipeline, *_ = build_fake_pipeline(project_root, ontology, app_config)
    result = pipeline.analyse(wav_path)
    assert result.dominant_modality in {"acoustic", "text"}
    assert result.acoustic.weight >= 0.0
    assert result.text.weight >= 0.0


# Ordering and error behaviour


def test_short_clip_fails_before_any_model_is_called(project_root, ontology, app_config, tmp_path):
    sr = 16_000
    t = np.linspace(0.0, 0.4, int(0.4 * sr), endpoint=False)
    path = tmp_path / "short.wav"
    sf.write(str(path), (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)

    pipeline, asr, acoustic, text = build_fake_pipeline(project_root, ontology, app_config)
    with pytest.raises(AudioTooShortError):
        pipeline.analyse(str(path))

    assert (asr.called, acoustic.called, text.called) == (0, 0, 0)


def test_inference_failure_is_translated(project_root, ontology, app_config, wav_path):
    class BrokenAcoustic(FakeAcoustic):
        def predict(self, audio, sample_rate):
            raise InferenceError("acoustic")

    pipeline, *_ = build_fake_pipeline(
        project_root, ontology, app_config, acoustic=BrokenAcoustic(ontology)
    )
    with pytest.raises(InferenceError) as exc:
        pipeline.analyse(wav_path)
    assert exc.value.code == "MMSA-E109"


def test_missing_dimensional_head_still_works(project_root, ontology, app_config, wav_path):
    pipeline, *_ = build_fake_pipeline(
        project_root, ontology, app_config, acoustic=FakeAcoustic(ontology, dimensional=False)
    )
    result = pipeline.analyse(wav_path)
    assert result.dimensional is None
    assert 0.0 <= result.stress.score <= 1.0


def test_low_transcript_confidence_produces_a_warning(project_root, ontology, app_config, wav_path):
    pipeline, *_ = build_fake_pipeline(
        project_root, ontology, app_config, asr=FakeASR(confidence=0.12)
    )
    result = pipeline.analyse(wav_path)
    assert any("confidence" in w.lower() for w in result.warnings)


# Transcript-only rerun


def test_text_rerun_updates_the_fused_result(project_root, ontology, app_config, wav_path):
    pipeline, _asr, _acoustic, text = build_fake_pipeline(project_root, ontology, app_config)
    original = pipeline.analyse(wav_path)

    # Swap the text branch's opinion, then re-run only that branch.
    text.probs = [0.85, 0.05, 0.03, 0.03, 0.02, 0.02]
    updated = pipeline.reanalyse_text("I am so happy and calm today", original)

    assert updated.asr.text == "I am so happy and calm today"
    assert updated.fused_probs != original.fused_probs
    assert updated.acoustic.raw_probs == original.acoustic.raw_probs  # audio untouched
    assert sum(updated.fused_probs.values()) == pytest.approx(1.0)
