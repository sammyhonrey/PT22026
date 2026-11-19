"""Prosodic feature extraction.

Thirteen features, chosen because they are the paralinguistic correlates of
arousal and vocal tension that the literature replicates most consistently (pitch
variability, energy dynamics, timing, spectral balance). They are computed here
rather than delegated to a library wrapper so that each parameter is explicit and
reproducible across machines.

Everything is derived from a 16 kHz mono ``float32`` array. ``librosa`` is imported
lazily so that the pure-maths test suite runs on a machine without it.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

__all__ = ["extract_prosody", "FEATURE_NAMES", "PROSODY_FEATURES"]

FEATURE_NAMES: tuple[str, ...] = (
    "f0_median",
    "f0_iqr",
    "f0_range_st",
    "voiced_ratio",
    "rms_mean",
    "rms_std",
    "rms_range_db",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean",
    "spectral_flatness_mean",
    "pause_ratio",
    "syllable_rate",
)
"""The complete feature set, in the order reported in the technical report."""

_HOP = 160      # 10 ms at 16 kHz
_N_FFT = 512    # 32 ms window
_FRAME = 512
_F0_MIN = 60.0
_F0_MAX = 400.0
_PAUSE_DBFS = -38.0


def _rms_envelope(y: np.ndarray) -> np.ndarray:
    """Frame-wise RMS using explicit parameters (stable across librosa versions)."""
    import librosa

    return librosa.feature.rms(y=y, frame_length=_FRAME, hop_length=_HOP)[0]


def _pitch(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Fundamental frequency track and voicing mask via probabilistic YIN."""
    import librosa

    f0, voiced_flag, _voiced_prob = librosa.pyin(
        y=y,
        fmin=_F0_MIN,
        fmax=_F0_MAX,
        sr=sr,
        frame_length=2048,
        hop_length=_HOP,
    )
    return np.asarray(f0, dtype=np.float64), np.asarray(voiced_flag, dtype=bool)


def _pause_ratio(rms: np.ndarray) -> float:
    """Fraction of frames below the pause gate."""
    if rms.size == 0:
        return 0.0
    peak = float(np.max(rms))
    if peak <= 0:
        return 1.0
    gate = peak * (10.0 ** (_PAUSE_DBFS / 20.0))
    return float(np.mean(rms < gate))


def _syllable_rate(y: np.ndarray, sr: int, rms: np.ndarray) -> float:
    """Peaks per second in the smoothed energy envelope.

    An approximation of syllable nuclei: smooth the envelope, find local maxima
    with a minimum separation of 120 ms and a height above the noise floor. It
    tracks speaking rate well enough to be a *relative* feature and is documented
    as an approximation in the report rather than presented as a true syllable
    counter.
    """
    from scipy.signal import find_peaks

    if rms.size < 5:
        return 0.0

    kernel = np.ones(5, dtype=np.float64) / 5.0
    smoothed = np.convolve(rms, kernel, mode="same")
    peak_val = float(np.max(smoothed))
    if peak_val <= 0:
        return 0.0

    min_distance = max(1, int(0.12 * sr / _HOP))
    peaks, _ = find_peaks(smoothed, height=0.25 * peak_val, distance=min_distance)
    duration_s = y.size / float(sr)
    if duration_s <= 0:
        return 0.0
    return float(peaks.size / duration_s)


def extract_prosody(y: np.ndarray, sr: int = 16_000) -> dict[str, float]:
    """Return the thirteen prosodic features as a JSON-safe dictionary.

    Every value is finite: missing or degenerate frames yield ``0.0`` rather than
    ``NaN``, because a ``NaN`` that reaches the chart layer or the stress index is
    much harder to diagnose than a zero.
    """
    import librosa

    audio = np.asarray(y, dtype=np.float32)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if audio.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    rms = _rms_envelope(audio)
    f0, voiced = _pitch(audio, sr)

    voiced_f0 = f0[np.isfinite(f0) & (f0 > 0)]
    if voiced_f0.size >= 2:
        f0_median = float(np.median(voiced_f0))
        f0_iqr = float(np.percentile(voiced_f0, 75) - np.percentile(voiced_f0, 25))
        lo = float(np.percentile(voiced_f0, 5))
        hi = float(np.percentile(voiced_f0, 95))
        f0_range_st = float(12.0 * np.log2(hi / lo)) if lo > 0 and hi > 0 else 0.0
    else:
        f0_median = f0_iqr = f0_range_st = 0.0

    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0

    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    rms_mean = float(np.mean(rms_db)) if rms_db.size else 0.0
    rms_std = float(np.std(rms_db)) if rms_db.size else 0.0
    rms_range_db = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5)) if rms_db.size else 0.0

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=_FRAME, hop_length=_HOP)[0]
    centroid = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=_N_FFT, hop_length=_HOP
    )[0]
    flatness = librosa.feature.spectral_flatness(
        y=audio, n_fft=_N_FFT, hop_length=_HOP
    )[0]

    features: dict[str, float] = {
        "f0_median": f0_median,
        "f0_iqr": f0_iqr,
        "f0_range_st": f0_range_st,
        "voiced_ratio": voiced_ratio,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_range_db": rms_range_db,
        "zcr_mean": float(np.mean(zcr)) if zcr.size else 0.0,
        "zcr_std": float(np.std(zcr)) if zcr.size else 0.0,
        "spectral_centroid_mean": float(np.mean(centroid)) if centroid.size else 0.0,
        "spectral_flatness_mean": float(np.mean(flatness)) if flatness.size else 0.0,
        "pause_ratio": _pause_ratio(rms),
        "syllable_rate": _syllable_rate(audio, sr, rms),
    }
    return {name: _safe_float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def _safe_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0
    return f if np.isfinite(f) else 0.0


PROSODY_FEATURES: Mapping[str, str] = {
    "f0_median": "median fundamental frequency of voiced frames (Hz)",
    "f0_iqr": "interquartile range of F0 (Hz) - pitch variability",
    "f0_range_st": "5th-95th percentile F0 span (semitones)",
    "voiced_ratio": "fraction of frames judged voiced",
    "rms_mean": "mean frame energy (dBFS)",
    "rms_std": "standard deviation of frame energy (dB) - loudness instability",
    "rms_range_db": "5th-95th percentile energy span (dB)",
    "zcr_mean": "mean zero-crossing rate",
    "zcr_std": "standard deviation of zero-crossing rate",
    "spectral_centroid_mean": "mean spectral centroid (Hz) - brightness",
    "spectral_flatness_mean": "mean spectral flatness - noisiness vs tonality",
    "pause_ratio": "fraction of frames below the pause gate",
    "syllable_rate": "approximate syllable nuclei per second",
}
"""Human-readable description of each feature, used in the report appendix."""
