"""
MP3 export for generated WAV audio.

Primary encoder: ``lameenc`` (bundled LAME, no external binary, works on
Windows). If it is unavailable, FFmpeg is used when installed. If neither
exists, a clear error explains how to install one - nothing is downloaded
automatically and no shell string is ever built from user input.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from voiceforge.app_core import AppError

SUPPORTED_BITRATES = (128, 192, 320)
DEFAULT_BITRATE = 192
MAX_BITRATE_KBPS = 320

FFMPEG_INSTALL_HINT = (
    "MP3 export needs an encoder. Install the Python package 'lameenc' "
    "(pip install lameenc), or install FFmpeg and add it to PATH "
    "(winget install Gyan.FFmpeg, then reopen the terminal)."
)


def _lameenc():
    try:
        import lameenc

        return lameenc
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def _ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def encoder_available() -> bool:
    return _lameenc() is not None or _ffmpeg_path() is not None


def validate_bitrate(bitrate: object) -> int:
    try:
        value = int(bitrate)
    except (TypeError, ValueError) as error:
        raise AppError(
            "The MP3 bitrate must be 128, 192 or 320 kbps.",
            code="invalid_bitrate",
        ) from error

    if value not in SUPPORTED_BITRATES:
        raise AppError(
            f"Unsupported bitrate {value} kbps. Choose 128, 192 or 320.",
            code="invalid_bitrate",
        )

    return value


def _validate_mp3(path: Path) -> None:
    """Reject empty or clearly corrupt encoder output."""

    if not path.is_file() or path.stat().st_size < 256:
        raise AppError(
            "MP3 conversion produced an empty file.",
            code="mp3_failed",
            status=500,
        )

    header = path.read_bytes()[:4]

    looks_mp3 = header[:3] == b"ID3" or (
        len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0
    )

    if not looks_mp3:
        path.unlink(missing_ok=True)
        raise AppError(
            "MP3 conversion produced an invalid file.",
            code="mp3_failed",
            status=500,
        )


def export_mp3(
    wav_path: str | Path,
    output_path: str | Path,
    bitrate: int = DEFAULT_BITRATE,
) -> dict:
    """Convert one WAV file to MP3 and return metadata."""

    bitrate = validate_bitrate(bitrate)

    source = Path(wav_path)

    if not source.is_file():
        raise AppError(
            "The audio file for this recording is missing.",
            code="audio_not_found",
            status=404,
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        data, sample_rate = sf.read(
            str(source),
            dtype="int16",
            always_2d=True,
        )
    except Exception as error:  # noqa: BLE001
        raise AppError(
            "This audio file could not be read for MP3 conversion.",
            code="invalid_audio",
        ) from error

    channels = int(data.shape[1])

    if data.shape[0] == 0:
        raise AppError(
            "This audio file is empty.",
            code="invalid_audio",
        )

    if channels not in (1, 2):
        raise AppError(
            "Only mono and stereo audio can be exported to MP3.",
            code="invalid_audio",
        )

    encoder_name = None
    lame = _lameenc()

    if lame is not None:
        try:
            encoder = lame.Encoder()
            encoder.set_bit_rate(bitrate)
            encoder.set_in_sample_rate(int(sample_rate))
            encoder.set_channels(channels)
            encoder.set_quality(2)

            mp3_bytes = encoder.encode(np.ascontiguousarray(data).tobytes())
            mp3_bytes += encoder.flush()

            destination.write_bytes(mp3_bytes)
            encoder_name = "lameenc"
        except Exception as error:  # noqa: BLE001
            raise AppError(
                f"MP3 conversion failed: {error}",
                code="mp3_failed",
                status=500,
            ) from error
    else:
        ffmpeg = _ffmpeg_path()

        if ffmpeg is None:
            raise AppError(
                FFMPEG_INSTALL_HINT,
                code="mp3_unavailable",
                status=503,
            )

        try:
            subprocess.run(  # noqa: S603 - fixed argument list, no shell
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-codec:a",
                    "libmp3lame",
                    "-b:a",
                    f"{bitrate}k",
                    str(destination),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
            encoder_name = "ffmpeg"
        except subprocess.CalledProcessError as error:
            raise AppError(
                "FFmpeg could not convert this file to MP3.",
                code="mp3_failed",
                status=500,
            ) from error
        except subprocess.TimeoutExpired as error:
            raise AppError(
                "MP3 conversion timed out.",
                code="mp3_failed",
                status=500,
            ) from error

    _validate_mp3(destination)

    return {
        "path": str(destination),
        "filename": destination.name,
        "bitrate": bitrate,
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": round(data.shape[0] / float(sample_rate), 2),
        "file_size": destination.stat().st_size,
        "encoder": encoder_name,
    }


def cached_mp3(
    wav_path: str | Path,
    cache_dir: str | Path,
    bitrate: int = DEFAULT_BITRATE,
) -> tuple[Path, dict]:
    """
    Return an MP3 for ``wav_path``, reusing a cached conversion.

    The cache key includes the WAV modification time, so regenerating a file
    produces a new MP3; the original WAV is never modified.
    """

    bitrate = validate_bitrate(bitrate)
    source = Path(wav_path)

    if not source.is_file():
        raise AppError(
            "The audio file for this recording is missing.",
            code="audio_not_found",
            status=404,
        )

    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha1(
        f"{source.resolve()}:{source.stat().st_mtime_ns}:{bitrate}".encode()
    ).hexdigest()[:16]

    target = directory / f"{key}-{bitrate}.mp3"

    if target.is_file() and target.stat().st_size > 0:
        data, sample_rate = sf.read(str(source), dtype="int16", always_2d=True)

        return target, {
            "path": str(target),
            "filename": target.name,
            "bitrate": bitrate,
            "sample_rate": int(sample_rate),
            "channels": int(data.shape[1]),
            "duration_seconds": round(data.shape[0] / float(sample_rate), 2),
            "file_size": target.stat().st_size,
            "encoder": "cache",
        }

    return target, export_mp3(source, target, bitrate)
