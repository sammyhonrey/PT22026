"""Audio ingestion: decode, validate, resample, trim, and describe.

The app must never show a stack trace, so every failure mode here is translated
into a typed :class:`mmsa.errors.MMSAError`. Decoding is deliberately layered:
``soundfile`` first (fast, handles WAV/FLAC/OGG), then ``librosa`` (adds MP3/M4A
via audioread), then the standard-library ``wave`` module, which keeps the unit
tests runnable on a machine with no audio stack installed at all.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Sequence

import numpy as np

from .errors import (
    AudioDecodeError,
    AudioTooShortError,
    AudioTruncatedWarning,
    FileTooLargeError,
    SilentAudioError,
    UnsupportedFormatError,
    MMSAWarning,
)
from .logging_utils import get_logger
from .schemas import AudioInfo

__all__ = [
    "load_audio",
    "estimate_snr_db",
    "is_clipped",
    "audition_preview",
    "lexical_negativity",
]

logger = get_logger(__name__)

_TARGET_SR = 16_000
_SILENCE_DBFS = -50.0
_TRIM_FLOOR = 0.02  # 2 % of peak counts as speech for the trim gate


# Decoding


def _decode_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf  # local import: optional at module level

    data, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32), int(samplerate)


def _decode_with_librosa(path: Path) -> tuple[np.ndarray, int]:
    import librosa  # local import: heavy, and optional for tests

    data, samplerate = librosa.load(str(path), sr=None, mono=False)
    return np.asarray(data, dtype=np.float32), int(samplerate)


def _decode_with_wave(path: Path) -> tuple[np.ndarray, int]:
    """Standard-library fallback for uncompressed PCM WAV only."""
    with wave.open(str(path), "rb") as handle:
        n_channels = handle.getnchannels()
        width = handle.getsampwidth()
        samplerate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)

    if width == 1:  # 8-bit PCM is unsigned
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise AudioDecodeError(f"{width * 8}-bit PCM is not supported.")

    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    return np.asarray(data, dtype=np.float32), int(samplerate)


def _decode(path: Path) -> tuple[np.ndarray, int]:
    errors: list[str] = []
    for name, decoder in (
        ("soundfile", _decode_with_soundfile),
        ("librosa", _decode_with_librosa),
        ("wave", _decode_with_wave),
    ):
        try:
            data, sr = decoder(path)
            if data.size == 0:
                raise ValueError("decoder returned no samples")
            logger.debug("decoded with %s: %d samples @ %d Hz", name, data.size, sr)
            return data, sr
        except Exception as exc:  # noqa: BLE001 - each backend fails differently
            errors.append(f"{name}: {type(exc).__name__}")
    logger.warning("all decoders failed for %s (%s)", path.name, "; ".join(errors))
    raise AudioDecodeError()


# Signal helpers


def _to_mono(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (frames, channels) from soundfile, (channels, frames) from librosa
        axis = 1 if arr.shape[0] >= arr.shape[1] else 0
        return arr.mean(axis=axis).astype(np.float32)
    raise AudioDecodeError("Unexpected audio dimensionality.")


def _resample(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return data
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(int(sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(sr) // divisor
    return np.asarray(resample_poly(data, up, down), dtype=np.float32)


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-9)))


def is_clipped(data: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at or beyond the full-scale threshold."""
    if data.size == 0:
        return 0.0
    return float(np.mean(np.abs(data) >= threshold))


def estimate_snr_db(data: np.ndarray, sr: int, frame_ms: float = 20.0) -> float | None:
    """Rough SNR: energy ratio between the loudest and quietest deciles of frames.

    Not a substitute for a proper noise-floor estimate, but it is monotone in
    broadband noise and cheap enough to run on every request. Capped at 60 dB.
    """
    frame = max(1, int(sr * frame_ms / 1000.0))
    n_frames = data.size // frame
    if n_frames < 4:
        return None
    reshaped = data[: n_frames * frame].reshape(n_frames, frame)
    energies = np.mean(reshaped.astype(np.float64) ** 2, axis=1)
    energies = energies[energies > 0]
    if energies.size < 4:
        return None
    noise = float(np.percentile(energies, 10))
    speech = float(np.percentile(energies, 90))
    if noise <= 0:
        return 60.0
    return float(min(60.0, 10.0 * np.log10(speech / noise)))


def _trim_silence(data: np.ndarray, sr: int) -> np.ndarray:
    """Crop leading and trailing near-silence, keeping 50 ms of context."""
    frame = max(1, int(0.02 * sr))
    n_frames = data.size // frame
    if n_frames < 3:
        return data

    reshaped = data[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(reshaped.astype(np.float64) ** 2, axis=1))
    peak = float(np.max(np.abs(data)))
    gate = max(1e-4, _TRIM_FLOOR * peak)

    above = np.flatnonzero(rms > gate)
    if above.size == 0:
        return data

    pad = int(0.05 * sr)
    start = max(0, int(above[0]) * frame - pad)
    stop = min(data.size, (int(above[-1]) + 1) * frame + pad)
    return data[start:stop]


def audition_preview(data: np.ndarray, sr: int = _TARGET_SR) -> np.ndarray:
    """8 kHz mono copy for the browser player (smaller payload, no artefacts)."""
    return _resample(np.asarray(data, dtype=np.float32), sr, 8000)


# Public API


def load_audio(
    path: str | Path,
    target_sr: int = _TARGET_SR,
    min_duration_s: float = 1.0,
    max_duration_s: float = 30.0,
    max_file_mb: float = 25.0,
    allowed_extensions: Sequence[str] = (".wav", ".flac", ".mp3", ".ogg", ".m4a"),
    warnings: list[MMSAWarning] | None = None,
) -> tuple[np.ndarray, AudioInfo]:
    """Validate and load a speech clip.

    Returns a mono ``float32`` waveform at ``target_sr`` plus descriptive
    statistics. Recoverable conditions (currently only over-length clips) are
    appended to ``warnings`` rather than raised, so the request still completes.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix not in allowed_extensions:
        raise UnsupportedFormatError(suffix)

    if not p.exists():
        raise AudioDecodeError("The uploaded file could not be found.")

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_file_mb:
        raise FileTooLargeError(size_mb, max_file_mb)

    data, sr = _decode(p)
    data = _to_mono(data)

    if data.size == 0:
        raise AudioDecodeError()

    # Statistics are taken on the signal as supplied, before any gain change, so
    # that "your recording is very quiet" remains visible to the user.
    peak_raw = float(np.max(np.abs(data)))
    clipping_ratio = is_clipped(data, threshold=0.999)
    peak_dbfs = _dbfs(peak_raw)

    rms_raw = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    if peak_raw <= 1e-7 or rms_raw <= 1e-7:
        raise SilentAudioError()

    data = _resample(data, sr, target_sr)
    data = data - float(np.mean(data))  # remove DC offset

    snr = estimate_snr_db(data, target_sr)
    trimmed = _trim_silence(data, target_sr)
    if trimmed.size >= int(0.05 * target_sr):
        data = trimmed

    duration = data.size / float(target_sr)
    if duration < min_duration_s:
        raise AudioTooShortError(duration, min_duration_s)

    if duration > max_duration_s:
        keep = int(max_duration_s * target_sr)
        data = data[:keep]
        duration = data.size / float(target_sr)
        if warnings is not None:
            warnings.append(AudioTruncatedWarning(duration, max_duration_s))

    # Normalise to -3 dBFS so downstream models see a consistent level.
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = (data / peak) * 0.7079

    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    info = AudioInfo(
        path=str(p),
        duration_s=round(float(duration), 4),
        sample_rate=int(target_sr),
        peak_dbfs=round(peak_dbfs, 2),
        rms_dbfs=round(_dbfs(rms), 2),
        snr_db=None if snr is None else round(float(snr), 2),
        clipping_ratio=round(float(clipping_ratio), 4),
        is_speech_suspected=bool(peak_dbfs > _SILENCE_DBFS and duration >= min_duration_s),
    )
    return np.ascontiguousarray(data, dtype=np.float32), info


def lexical_negativity(
    text: str,
    lexicon: dict[str, list[str]],
    max_hits: int = 6,
) -> tuple[float, int]:
    """Count negative affect words in a transcript, normalised by length.

    A crude proxy used only as one stress term; it is not a sentiment model.
    """
    tokens = [t.strip(".,!?;:\"'()").lower() for t in text.split()]
    tokens = [t for t in tokens if t]

    negatives = set(lexicon.get("negative", []))
    positives = set(lexicon.get("positive", []))
    intensifiers = set(lexicon.get("intensifiers", []))
    negators = set(lexicon.get("negators", []))

    score = 0.0
    hits = 0
    for i, token in enumerate(tokens):
        weight = 0.0
        if token in negatives:
            weight = 0.6
        elif token in positives:
            weight = -0.4
        if weight == 0.0:
            continue

        # Look back over a two-token window so that "stressed", "very stressed"
        # and "not very stressed" are all handled by the same rule.
        window = tokens[max(0, i - 2) : i]
        if any(token in negators for token in window):
            weight *= -1.0
        if any(token in intensifiers for token in window):
            weight *= 1.5

        score += weight
        hits += 1
        if hits >= max_hits:
            break

    if hits == 0:
        return 0.0, 0
    normalised = score / float(max(hits, 1))
    return float(max(0.0, min(1.0, normalised))), hits
