"""Check generated WAV files for common audio problems.

Sounds can fail in ways that look fine on a waveform plot but are obvious
defects: NaN samples, digital silence, clipping, a constant DC offset, or a
flat buzz with no speech-like dynamics.

This script generates speech from a checkpoint, analyses every file, and
returns a non-zero exit code if a hard problem is found.

Usage::

    python -m scripts.check_audio \
        --checkpoint checkpoints/voiceforge_attn.pt \
        --output-dir outputs/audio_checks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voiceforge.audio import SAMPLE_RATE
from voiceforge.tts import VoiceForgeTTS

DEFAULT_CHECKPOINT = "checkpoints/voiceforge_attn.pt"
DEFAULT_OUTPUT_DIR = "outputs/audio_checks"

DEFAULT_SENTENCES = [
    "Hello and welcome to VoiceForge.",
    "This is a test of the speech synthesis system.",
    "The quick brown fox jumps over the lazy dog.",
    "Speech should rise and fall in volume.",
]

# Thresholds used to decide whether a file is acceptable.
MIN_RMS = 1e-3            # below this the file is effectively silent
MIN_PEAK = 0.05           # peak amplitude
MAX_CLIPPING_RATIO = 0.01 # fraction of samples at full scale
MAX_DC_OFFSET = 0.02
MIN_ENERGY_VARIATION = 0.15  # coefficient of variation of short-term RMS
ENVIRONMENT_WINDOW = 1024


def _short_term_energy(waveform: np.ndarray) -> np.ndarray:
    """Root-mean-square energy in consecutive window blocks."""

    usable = len(waveform) // ENVIRONMENT_WINDOW * ENVIRONMENT_WINDOW

    if usable == 0:
        return np.array([])

    blocks = waveform[:usable].reshape(-1, ENVIRONMENT_WINDOW)
    return np.sqrt((blocks**2).mean(axis=1))


def analyze_waveform(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    """
    Analyse one waveform and list every problem found.

    Parameters
    ----------
    waveform:
        One-dimensional float array in the range roughly [-1, 1].

    Returns
    -------
    dict with the measurements, a list of ``problems`` and a boolean
    ``passed`` flag.
    """

    waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)

    if waveform.size == 0:
        return {
            "samples": 0,
            "duration_seconds": 0.0,
            "problems": ["empty"],
            "passed": False,
        }

    finite = np.isfinite(waveform)
    non_finite_count = int((~finite).sum())

    safe = np.where(finite, waveform, 0.0)

    peak = float(np.abs(safe).max())
    rms = float(np.sqrt((safe**2).mean()))
    dc_offset = float(safe.mean())
    clipping_ratio = float((np.abs(safe) >= 0.999).mean())

    energy = _short_term_energy(safe)

    if energy.size >= 2 and energy.mean() > 0:
        energy_variation = float(energy.std() / energy.mean())
    else:
        energy_variation = 0.0

    problems: list[str] = []

    if non_finite_count:
        problems.append("non_finite_samples")

    if rms < MIN_RMS:
        problems.append("silent")

    if peak < MIN_PEAK:
        problems.append("low_peak")

    if clipping_ratio > MAX_CLIPPING_RATIO:
        problems.append("clipping")

    if abs(dc_offset) > MAX_DC_OFFSET:
        problems.append("dc_offset")

    if energy_variation < MIN_ENERGY_VARIATION:
        problems.append("flat_buzz")

    return {
        "samples": int(waveform.size),
        "duration_seconds": round(waveform.size / sample_rate, 3),
        "peak": round(peak, 5),
        "rms": round(rms, 6),
        "dc_offset": round(dc_offset, 6),
        "clipping_ratio": round(clipping_ratio, 6),
        "non_finite_samples": non_finite_count,
        "energy_variation": round(energy_variation, 4),
        "problems": problems,
        "passed": not problems,
    }


def analyze_file(path: str | Path) -> dict:
    """Read a WAV file and analyse it."""

    audio, sample_rate = sf.read(str(path), dtype="float64")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    result = analyze_waveform(audio, sample_rate)
    result["path"] = str(path)

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate speech and check the audio for defects.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--n-iter", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--text",
        nargs="*",
        default=None,
        help="Sentences to synthesise. Defaults to a built-in list.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional path for the JSON report.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint not found: {checkpoint_path}")
        return 1

    sentences = args.text or DEFAULT_SENTENCES
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    pipeline = VoiceForgeTTS.from_pretrained(
        checkpoint_path,
        device=args.device,
        griffin_lim_iterations=args.n_iter,
    )

    reports = []

    for index, sentence in enumerate(sentences, start=1):
        path = output_dir / f"check_{index:02d}.wav"

        print(f"[{index}/{len(sentences)}] {sentence!r}")

        pipeline.save(sentence, path)

        report = analyze_file(path)
        report["text"] = sentence
        report["truncated"] = bool(pipeline.was_truncated)
        report["padded"] = bool(pipeline.was_padded)

        reports.append(report)

        status = "PASS" if report["passed"] else "FAIL"

        print(
            f"    {status} peak={report['peak']} rms={report['rms']} "
            f"variation={report['energy_variation']} "
            f"problems={report['problems']}"
        )

    failures = [report for report in reports if not report["passed"]]

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(pipeline.device),
        "files": reports,
        "passed": len(reports) - len(failures),
        "failed": len(failures),
    }

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\n{summary['passed']}/{len(reports)} files passed, "
        f"{summary['failed']} failed."
    )

    print(json.dumps(summary, indent=2))

    return 0 if not failures else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
