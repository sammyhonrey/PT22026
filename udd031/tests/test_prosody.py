"""Tests for prosodic feature extraction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mmsa.prosody import FEATURE_NAMES, PROSODY_FEATURES, extract_prosody


def test_all_features_present_and_finite(sine_wave):
    features = extract_prosody(sine_wave(2.0))
    assert set(features) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())
    assert all(math.isfinite(v) for v in features.values())


def test_every_feature_is_documented():
    assert set(PROSODY_FEATURES) == set(FEATURE_NAMES)


def test_pitch_tracks_a_pure_tone(sine_wave):
    features = extract_prosody(sine_wave(2.0, freq=220.0))
    assert features["f0_median"] == pytest.approx(220.0, abs=8.0)


def test_pitch_rises_with_frequency(sine_wave):
    low = extract_prosody(sine_wave(2.0, freq=140.0))
    high = extract_prosody(sine_wave(2.0, freq=320.0))
    assert high["f0_median"] > low["f0_median"]


def test_pause_ratio_separates_speech_from_gated_signal():
    sr = 16_000
    rng = np.random.default_rng(3)
    continuous = (0.3 * rng.standard_normal(sr * 2)).astype(np.float32)

    half = continuous.copy()
    half[sr:] = 0.0  # second half silent

    assert extract_prosody(continuous, sr)["pause_ratio"] < 0.35
    assert extract_prosody(half, sr)["pause_ratio"] > 0.35


def test_empty_audio_returns_zeros():
    features = extract_prosody(np.zeros(0, dtype=np.float32))
    assert features["f0_median"] == 0.0
    assert set(features) == set(FEATURE_NAMES)


def test_silence_does_not_produce_nan():
    features = extract_prosody(np.zeros(16_000, dtype=np.float32))
    assert all(math.isfinite(v) for v in features.values())


def test_energy_features_are_in_decibels(sine_wave):
    features = extract_prosody(sine_wave(2.0))
    # A 0.5-amplitude tone sits near -9 dBFS; the mean frame energy should be
    # comfortably in the negative dB range rather than a linear 0-1 value.
    assert -40.0 < features["rms_mean"] < 0.0
    assert features["rms_std"] >= 0.0
