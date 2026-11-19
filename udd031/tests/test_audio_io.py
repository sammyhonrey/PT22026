"""Tests for audio ingestion, built on synthetic files written to tmp_path."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from mmsa.audio_io import estimate_snr_db, is_clipped, lexical_negativity, load_audio
from mmsa.errors import (
    AudioDecodeError,
    AudioTooShortError,
    AudioTruncatedWarning,
    FileTooLargeError,
    SilentAudioError,
    UnsupportedFormatError,
)


def _write(path, data: np.ndarray, sr: int = 16_000) -> str:
    sf.write(str(path), data, sr)
    return str(path)


def test_loads_a_valid_clip(tmp_path, sine_wave):
    path = _write(tmp_path / "ok.wav", sine_wave(2.0))
    data, info = load_audio(path)

    assert data.ndim == 1
    assert info.sample_rate == 16_000
    assert 1.9 <= info.duration_s <= 2.1
    assert np.max(np.abs(data)) == pytest.approx(0.7079, abs=1e-3)
    assert info.is_speech_suspected is True
    assert info.snr_db is None or info.snr_db > 0


def test_resamples_to_target_rate(tmp_path, sine_wave):
    path = _write(tmp_path / "44k.wav", sine_wave(2.0, sr=44_100), sr=44_100)
    data, info = load_audio(path, target_sr=16_000)
    assert info.sample_rate == 16_000
    assert 1.9 <= data.size / 16_000 <= 2.1


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("not audio")
    with pytest.raises(UnsupportedFormatError) as exc:
        load_audio(path)
    assert exc.value.code == "MMSA-E101"


def test_oversized_file_is_rejected(tmp_path, sine_wave):
    path = _write(tmp_path / "big.wav", sine_wave(2.0))
    with pytest.raises(FileTooLargeError) as exc:
        load_audio(path, max_file_mb=0.0001)
    assert exc.value.code == "MMSA-E102"


def test_too_short_clip_is_rejected(tmp_path, sine_wave):
    path = _write(tmp_path / "short.wav", sine_wave(0.4))
    with pytest.raises(AudioTooShortError) as exc:
        load_audio(path, min_duration_s=1.0)
    assert exc.value.code == "MMSA-E104"


def test_silent_clip_is_rejected(tmp_path):
    path = _write(tmp_path / "silence.wav", np.zeros(32_000, dtype=np.float32))
    with pytest.raises(SilentAudioError) as exc:
        load_audio(path)
    assert exc.value.code == "MMSA-E106"


def test_corrupt_payload_is_rejected(tmp_path):
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk" * 40)
    with pytest.raises(AudioDecodeError) as exc:
        load_audio(path)
    assert exc.value.code == "MMSA-E103"


def test_renamed_text_file_is_rejected(tmp_path):
    path = tmp_path / "fake.wav"
    path.write_text("this is definitely not a wav file, just prose" * 20)
    with pytest.raises(AudioDecodeError):
        load_audio(path)


def test_overlength_clip_is_trimmed_with_a_warning(tmp_path, sine_wave):
    path = _write(tmp_path / "long.wav", sine_wave(45.0))
    warnings: list = []
    data, info = load_audio(path, max_duration_s=30.0, warnings=warnings)

    assert info.duration_s <= 30.05
    assert data.size <= int(30.05 * 16_000)
    assert any(isinstance(w, AudioTruncatedWarning) for w in warnings)
    assert warnings[0].code == "MMSA-E105"


def test_clipping_is_detected(tmp_path):
    t = np.linspace(0, 2.0, 32_000, endpoint=False)
    square = (np.sign(np.sin(2 * np.pi * 200 * t)) * 1.0).astype(np.float32)
    path = _write(tmp_path / "clipped.wav", square)
    _data, info = load_audio(path)
    assert info.clipping_ratio > 0.9


def test_is_clipped_helper():
    assert is_clipped(np.array([0.0, 0.5, -0.5], dtype=np.float32)) == 0.0
    assert is_clipped(np.array([1.0, -1.0, 0.0], dtype=np.float32)) == pytest.approx(2 / 3)


def test_snr_estimate_prefers_clean_signal():
    """The estimator measures the gap between the loud and quiet deciles.

    A pure tone has an almost constant frame energy, so it is the wrong probe:
    the signal needs a loud part and a quiet part for a ratio to exist at all.
    """
    sr = 16_000
    rng = np.random.default_rng(7)
    speech = np.zeros(sr * 2, dtype=np.float32)
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    speech[:sr] = (0.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    clean = speech + 1e-6 * rng.standard_normal(speech.size).astype(np.float32)
    noisy = speech + 0.2 * rng.standard_normal(speech.size).astype(np.float32)

    clean_snr = estimate_snr_db(clean, sr)
    noisy_snr = estimate_snr_db(noisy, sr)
    assert clean_snr is not None and noisy_snr is not None
    assert clean_snr > noisy_snr + 10.0


def test_stereo_is_downmixed(tmp_path):
    t = np.linspace(0, 2.0, 32_000, endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 220 * t)
    right = 0.5 * np.sin(2 * np.pi * 330 * t)
    stereo = np.stack([left, right], axis=1).astype(np.float32)
    path = _write(tmp_path / "stereo.wav", stereo)
    data, info = load_audio(path)
    assert data.ndim == 1
    assert info.duration_s == pytest.approx(2.0, abs=0.1)


# Lexicon


LEXICON = {
    "negative": ["stressed", "tired", "deadline"],
    "positive": ["calm", "fine"],
    "intensifiers": ["very", "really"],
    "negators": ["not", "never"],
}


def test_lexicon_detects_negative_terms():
    score, hits = lexical_negativity("I am so tired and stressed", LEXICON)
    assert hits == 2
    assert score > 0


def test_lexicon_handles_negation():
    positive, hits = lexical_negativity("I am not stressed at all", LEXICON)
    assert hits == 1
    assert positive == 0.0


def test_lexicon_intensifier_increases_magnitude():
    plain, _ = lexical_negativity("I am stressed", LEXICON)
    strong, _ = lexical_negativity("I am very stressed", LEXICON)
    assert strong > plain


def test_lexicon_neutral_text_scores_zero():
    score, hits = lexical_negativity("The meeting is at four o'clock", LEXICON)
    assert (score, hits) == (0.0, 0)
