"""
Join generated WAV chunks into a single audio file.

Long text and batches synthesise several chunks; this module concatenates
them with a short silence gap and guarantees a consistent sample rate and
channel layout in the output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

#: Default pause inserted between chunks.
DEFAULT_GAP_SECONDS = 0.15

#: Peak the joined file is scaled down to if it would clip.
SAFE_PEAK = 0.95


class AudioJoinError(RuntimeError):
    """Raised when chunks cannot be combined safely."""


def _to_channels(data: np.ndarray, channels: int) -> np.ndarray:
    """Convert ``data`` (frames, c) to the requested channel count."""

    current = data.shape[1]

    if current == channels:
        return data

    # Mono -> stereo: duplicate the channel.
    if current == 1 and channels == 2:
        return np.repeat(data, 2, axis=1)

    # Stereo -> mono: average.
    if current == 2 and channels == 1:
        return data.mean(axis=1, keepdims=True)

    raise AudioJoinError(
        f"Cannot convert {current} channels to {channels} channels"
    )


def join_wav_files(
    paths: list[str | Path],
    output_path: str | Path,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> dict:
    """
    Concatenate WAV files into one 16-bit PCM file.

    All inputs must share a sample rate; a mismatch raises
    :class:`AudioJoinError` rather than producing sped-up or corrupted
    audio. Channel mismatches are converted (mono/stereo only).

    Returns metadata about the written file.
    """

    if not paths:
        raise AudioJoinError("No audio chunks were provided")

    if gap_seconds < 0:
        raise ValueError("Gap between chunks cannot be negative")

    sample_rate: int | None = None
    target_channels = 1
    loaded: list[np.ndarray] = []

    for path in paths:
        source = Path(path)

        try:
            data, rate = sf.read(
                str(source),
                dtype="float32",
                always_2d=True,
            )
        except Exception as error:  # noqa: BLE001 - wrapped for the caller
            raise AudioJoinError(
                f"Could not read audio chunk {source.name}: {error}"
            ) from error

        if data.shape[0] == 0:
            raise AudioJoinError(f"Audio chunk {source.name} is empty")

        if data.shape[1] not in (1, 2):
            raise AudioJoinError(
                f"Cannot combine audio with {data.shape[1]} channels: "
                "only mono and stereo are supported"
            )

        if sample_rate is None:
            sample_rate = rate

        elif rate != sample_rate:
            raise AudioJoinError(
                "Cannot combine audio with different sample rates: "
                f"{rate} Hz and {sample_rate} Hz"
            )

        target_channels = max(target_channels, int(data.shape[1]))
        loaded.append(data)

    channels = target_channels
    parts = [_to_channels(data, channels) for data in loaded]

    assert sample_rate is not None

    gap_frames = int(round(gap_seconds * sample_rate))
    silence = np.zeros((gap_frames, channels), dtype=np.float32)

    separated: list[np.ndarray] = []

    for index, part in enumerate(parts):
        if index > 0 and gap_frames > 0:
            separated.append(silence)
        separated.append(part)

    audio = np.concatenate(separated, axis=0)

    peak = float(np.abs(audio).max())

    if peak > SAFE_PEAK:
        audio = audio * (SAFE_PEAK / peak)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        str(destination),
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    return {
        "path": str(destination),
        "filename": destination.name,
        "duration_seconds": round(audio.shape[0] / sample_rate, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "chunks": len(parts),
        "gap_seconds": gap_seconds,
    }
