"""Generate speech with a trained VoiceForge checkpoint.

Usage examples
--------------
Generate one sentence and save it as a WAV file::

    python -m scripts.generate "Hello, welcome to VoiceForge!" \
        --output outputs/hello.wav

Generate several sentences into one folder::

    python -m scripts.generate "First sentence." "Second sentence." \
        --output-dir outputs/samples

Read the text from a file (one sentence per line)::

    python -m scripts.generate --text-file sentences.txt \
        --output-dir outputs/samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

from voiceforge.audio import SAMPLE_RATE
from voiceforge.tts import VoiceForgeTTS

DEFAULT_CHECKPOINT = "checkpoints/voiceforge_attn.pt"


def build_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""

    parser = argparse.ArgumentParser(
        description="Generate speech from text using VoiceForge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "text",
        nargs="*",
        help="One or more sentences to speak.",
    )

    parser.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="Read sentences from a UTF-8 text file, one per line.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained VoiceForge checkpoint.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output WAV path. Only valid with a single sentence.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Folder used when several sentences are generated.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device. Defaults to CUDA when available.",
    )

    parser.add_argument(
        "--n-iter",
        type=int,
        default=32,
        help="Griffin-Lim iterations (more is slower but cleaner).",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=2200,
        help="Hard limit on generated Mel frames (about 25 seconds).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by Griffin-Lim phase initialisation.",
    )

    return parser


def collect_sentences(args: argparse.Namespace) -> list[str]:
    """Combine sentences from the command line and an optional text file."""

    sentences = [text for text in args.text if text.strip()]

    if args.text_file is not None:
        text_path = Path(args.text_file)

        if not text_path.is_file():
            raise FileNotFoundError(
                f"Text file not found: {text_path}"
            )

        for line in text_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sentences.append(line.strip())

    if not sentences:
        raise ValueError(
            "No text was provided. Pass a sentence as an argument "
            "or use --text-file."
        )

    if args.output is not None and len(sentences) > 1:
        raise ValueError(
            "--output can only be used with a single sentence. "
            "Use --output-dir for multiple sentences."
        )

    return sentences


def _safe_filename(text: str, index: int) -> str:
    """Turn a sentence into a short, filesystem-safe file name."""

    cleaned = "".join(
        character if character.isalnum() else "_"
        for character in text.lower()
    )

    cleaned = "_".join(part for part in cleaned.split("_") if part)

    return f"{index:02d}_{cleaned[:40] or 'speech'}.wav"


def _describe_wav(path: Path) -> dict:
    """Read a WAV file back and describe its contents."""

    data, sample_rate = sf.read(str(path), dtype="float32")

    return {
        "sample_rate": int(sample_rate),
        "samples": int(data.shape[0]),
        "duration_seconds": round(float(data.shape[0]) / sample_rate, 3),
        "peak": round(float(abs(data).max()), 5),
    }


def generate(args: argparse.Namespace) -> list[dict]:
    """
    Run speech generation and return a summary for every output file.

    The function is separate from ``main`` so it can be tested directly.
    """

    sentences = collect_sentences(args)

    if args.n_iter < 1:
        raise ValueError("Griffin-Lim iterations must be at least 1")

    if args.max_frames < 1:
        raise ValueError("--max-frames must be at least 1")

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train a model first or pass --checkpoint."
        )

    # Griffin-Lim starts from random phase, so seed for repeatable output.
    torch.manual_seed(args.seed)

    print(f"Loading checkpoint: {checkpoint_path}")

    pipeline = VoiceForgeTTS.from_pretrained(
        checkpoint_path,
        device=args.device,
        griffin_lim_iterations=args.n_iter,
        max_mel_frames=args.max_frames,
    )

    print(f"Device: {pipeline.device}")

    output_dir = Path(args.output_dir)
    summaries: list[dict] = []

    for index, sentence in enumerate(sentences, start=1):
        if args.output is not None:
            output_path = Path(args.output)
        else:
            output_path = output_dir / _safe_filename(sentence, index)

        print(f"[{index}/{len(sentences)}] {sentence!r}")

        pipeline.save(sentence, output_path)

        if not output_path.is_file():
            raise RuntimeError(
                f"Expected output file was not created: {output_path}"
            )

        if output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Output file is empty: {output_path}"
            )

        details = _describe_wav(output_path)

        summary = {
            "text": sentence,
            "output": str(output_path),
            "truncated": bool(pipeline.was_truncated),
            **details,
        }

        summaries.append(summary)

        print(
            f"    saved {output_path} "
            f"({details['duration_seconds']} s, "
            f"{details['samples']} samples, "
            f"peak {details['peak']})"
        )

    return summaries


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.  Returns a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summaries = generate(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("\nSummary:")
    print(json.dumps(summaries, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
