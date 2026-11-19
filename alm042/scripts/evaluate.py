"""Objective evaluation of a VoiceForge checkpoint.

The script measures three separate things, because they fail for different
reasons and should not be confused with each other:

1. **Acoustic model quality** - how close the predicted log-Mel spectrogram
   is to the real one, and how accurate the duration predictor is.
2. **Vocoder quality** - how much detail Griffin-Lim loses when it converts
   a *real* Mel spectrogram back into audio.  This isolates the
   reconstruction algorithm from the acoustic model.
3. **Generated audio statistics** - how much the energy of generated speech
   varies over time compared with real speech.

The numbers are objective and reproducible.  They are **not** a substitute
for a human listening test, and they do not measure intelligibility
directly.

Usage::

    python -m scripts.evaluate --checkpoint checkpoints/voiceforge.pt \
        --samples 100 --output outputs/eval.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from voiceforge.audio import SAMPLE_RATE, mel_spectrogram, mel_to_audio, normalize_waveform
from voiceforge.checkpoint import load_model
from voiceforge.dataset import LJSpeechDataset

DEFAULT_METADATA = "data/LJSpeech-1.1/LJSpeech-1.1/metadata.csv"
DEFAULT_OUTPUT = "outputs/evaluation.json"

# Griffin-Lim needs at least n_fft // hop_length frames.
MIN_FRAMES = 4

# Window (in samples) used to measure short-term energy variation.
ENERGY_WINDOW = 1024


def build_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate a VoiceForge checkpoint objectively.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/voiceforge_attn.pt",
        help="Checkpoint to evaluate.",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA,
        help="LJSpeech metadata.csv used as the reference dataset.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of held-out samples used for the evaluation.",
    )
    parser.add_argument(
        "--vocoder-samples",
        type=int,
        default=4,
        help="Real samples used for the Griffin-Lim round-trip test.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=32,
        help="Griffin-Lim iterations used for the audio tests.",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device. Defaults to CUDA when available.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSON file the results are written to.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the final JSON, not per-sample progress.",
    )

    return parser


def _as_mel(sample: dict) -> torch.Tensor:
    """Return a sample Mel spectrogram as a 2-D ``(80, frames)`` tensor."""

    mel = sample["mel"]

    if mel.ndim == 3:
        mel = mel.squeeze(0)

    return mel


def _as_waveform(sample: dict) -> torch.Tensor:
    """Return a sample waveform as a 1-D tensor."""

    waveform = sample["waveform"]

    if waveform.ndim == 2:
        waveform = waveform.squeeze(0)

    return waveform


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> float:
    """
    Mean absolute error over the valid frames of a padded batch.

    Padded frames are excluded so that short sentences are not penalised for
    the zero padding added by the batching function.
    """

    frames = torch.arange(target.shape[2], device=target.device)
    mask = (frames.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).float()

    error = (prediction - target).abs() * mask

    return float(error.sum() / (mask.sum() * target.shape[1]))


def evaluate_model(
    model,
    dataset,
    indices: list[int],
    device: torch.device,
    progress: bool = False,
) -> dict:
    """
    Measure Mel reconstruction error, duration accuracy and spectral detail.

    ``mel_l1`` is the mean absolute error between the predicted and real
    log-Mel values, in the same units as the log-Mel spectrogram (natural
    log of power).  Lower is better; a value near 0 would mean a perfect
    reconstruction.

    ``spectral_detail_ratio`` compares the standard deviation of the
    predicted Mel with that of the real Mel.  A value well below 1 means the
    model is predicting an over-smoothed, low-detail spectrogram.
    """

    if not hasattr(model, "predict_mel_lengths"):
        raise ValueError(
            "Model does not implement predict_mel_lengths(), "
            "so duration accuracy cannot be measured."
        )

    model.eval()

    mel_l1_values: list[float] = []
    duration_mae_values: list[float] = []
    duration_relative_values: list[float] = []
    predicted_std: list[float] = []
    target_std: list[float] = []
    skipped = 0

    for index in indices:
        sample = dataset[index]
        target = _as_mel(sample)

        if target.shape[1] < MIN_FRAMES:
            skipped += 1
            continue

        text = sample["text_sequence"].unsqueeze(0).to(device)
        text_lengths = torch.tensor([text.shape[1]], device=device)

        target_mel = target.unsqueeze(0).to(device)
        mel_lengths = torch.tensor([target_mel.shape[2]], device=device)

        with torch.inference_mode():
            prediction = model(text, text_lengths, mel_lengths)
            predicted_lengths = model.predict_mel_lengths(text, text_lengths)

        mel_l1_values.append(_masked_l1(prediction, target_mel, mel_lengths))

        true_frames = int(mel_lengths.item())
        predicted_frames = int(predicted_lengths.item())

        duration_mae_values.append(abs(predicted_frames - true_frames))
        duration_relative_values.append(
            abs(predicted_frames - true_frames) / max(true_frames, 1)
        )

        predicted_std.append(float(prediction[0, :, :true_frames].std()))
        target_std.append(float(target.std()))

        if progress and len(mel_l1_values) % 10 == 0:
            print(f"    evaluated {len(mel_l1_values)}/{len(indices)}")

    if not mel_l1_values:
        raise ValueError(
            "No valid samples were found for evaluation."
        )

    mean_predicted_std = float(np.mean(predicted_std))
    mean_target_std = float(np.mean(target_std))

    return {
        "samples": len(mel_l1_values),
        "samples_skipped": skipped,
        "mel_l1": round(float(np.mean(mel_l1_values)), 4),
        "duration_mae_frames": round(float(np.mean(duration_mae_values)), 2),
        "duration_relative_error": round(
            float(np.mean(duration_relative_values)), 4
        ),
        "predicted_mel_std": round(mean_predicted_std, 4),
        "target_mel_std": round(mean_target_std, 4),
        "spectral_detail_ratio": round(
            mean_predicted_std / max(mean_target_std, 1e-6), 4
        ),
    }


def evaluate_vocoder(
    dataset,
    indices: list[int],
    n_iter: int = 32,
    progress: bool = False,
) -> dict:
    """
    Round-trip real Mel spectrograms through Griffin-Lim.

    The input Mel is the ground truth, so every error measured here comes
    from Mel inversion and phase estimation, not from the acoustic model.
    This is how the vocoder is judged separately from the model.
    """

    rmse_values: list[float] = []
    correlation_values: list[float] = []
    skipped = 0

    for position, index in enumerate(indices, start=1):
        target_mel = _as_mel(dataset[index])

        if target_mel.shape[1] < MIN_FRAMES:
            skipped += 1
            continue

        waveform = mel_to_audio(target_mel, n_iter=n_iter)
        reconstructed = mel_spectrogram(waveform.unsqueeze(0))

        length = min(reconstructed.shape[2], target_mel.shape[1])

        estimate = reconstructed[0, :, :length]
        reference = target_mel[:, :length]

        rmse_values.append(float((estimate - reference).pow(2).mean().sqrt()))

        correlation = torch.corrcoef(
            torch.stack([estimate.flatten(), reference.flatten()])
        )[0, 1]

        correlation_values.append(float(correlation))

        if progress:
            print(f"    vocoder round-trip {position}/{len(indices)}")

    if not rmse_values:
        raise ValueError("No valid samples were found for the vocoder test.")

    return {
        "samples": len(rmse_values),
        "samples_skipped": skipped,
        "n_iter": n_iter,
        "mel_rmse": round(float(np.mean(rmse_values)), 4),
        "mel_correlation": round(float(np.mean(correlation_values)), 4),
    }


def _energy_variation(waveform: torch.Tensor) -> float | None:
    """
    Coefficient of variation of short-term energy.

    Natural speech alternates loud and quiet syllables, so this value is
    clearly above zero.  A constant buzz produced by an undertrained model
    stays close to zero.
    """

    if waveform.numel() < ENERGY_WINDOW * 2:
        return None

    energy = waveform.pow(2).unfold(0, ENERGY_WINDOW, ENERGY_WINDOW)
    energy = energy.mean(dim=1).sqrt()

    if energy.numel() < 2:
        return None

    return float(energy.std() / (energy.mean() + 1e-8))


def evaluate_generated_audio(
    model,
    dataset,
    indices: list[int],
    device: torch.device,
    n_iter: int = 32,
    progress: bool = False,
) -> dict:
    """
    Generate audio from text and compare its statistics with real speech.

    This is an indirect quality check.  It cannot tell whether words are
    intelligible, but it does show whether the generated audio has the
    amplitude structure of speech at all.
    """

    model.eval()

    generated_variation: list[float] = []
    real_variation: list[float] = []
    durations: list[float] = []
    skipped = 0

    for position, index in enumerate(indices, start=1):
        sample = dataset[index]

        text = sample["text_sequence"].unsqueeze(0).to(device)
        text_lengths = torch.tensor([text.shape[1]], device=device)

        with torch.inference_mode():
            predicted_mel = model(text, text_lengths)

        if predicted_mel.shape[2] < MIN_FRAMES:
            skipped += 1
            continue

        waveform = mel_to_audio(predicted_mel[0].cpu(), n_iter=n_iter)
        waveform = normalize_waveform(waveform)

        variation = _energy_variation(waveform)

        if variation is None:
            skipped += 1
            continue

        generated_variation.append(variation)
        durations.append(float(waveform.numel()) / SAMPLE_RATE)

        real = _energy_variation(_as_waveform(sample))

        if real is not None:
            real_variation.append(real)

        if progress:
            print(f"    generated audio {position}/{len(indices)}")

    return {
        "samples": len(generated_variation),
        "samples_skipped": skipped,
        "generated_energy_variation": round(float(np.mean(generated_variation)), 4)
        if generated_variation
        else None,
        "real_energy_variation": round(float(np.mean(real_variation)), 4)
        if real_variation
        else None,
        "average_generated_seconds": round(float(np.mean(durations)), 3)
        if durations
        else None,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    metadata_path: str | Path = DEFAULT_METADATA,
    samples: int = 64,
    vocoder_samples: int = 4,
    n_iter: int = 32,
    device: str | None = None,
    seed: int = 0,
    progress: bool = True,
) -> dict:
    """
    Evaluate a checkpoint and return the results as a dictionary.

    The evaluation uses the *last* ``samples`` rows of the dataset, which
    keeps it disjoint from training runs that use a prefix of the dataset.
    """

    if samples < 1:
        raise ValueError("--samples must be at least 1")

    if vocoder_samples < 1:
        raise ValueError("--vocoder-samples must be at least 1")

    if n_iter < 1:
        raise ValueError("--n-iter must be at least 1")

    torch.manual_seed(seed)
    np.random.seed(seed)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Loading checkpoint: {checkpoint_path}")
    model, config, checkpoint_metadata = load_model(
        checkpoint_path,
        map_location=resolved_device,
    )
    model = model.to(resolved_device)

    print(f"Architecture: {config.get('architecture', 'legacy_interpolate')}")
    print(f"Device: {resolved_device}")

    dataset = LJSpeechDataset(metadata_path)

    if len(dataset) < samples:
        raise ValueError(
            f"Dataset only has {len(dataset)} samples, "
            f"not enough for --samples {samples}"
        )

    model_indices = list(range(len(dataset) - samples, len(dataset)))

    vocoder_indices = model_indices[: min(vocoder_samples, len(model_indices))]
    audio_indices = model_indices[: min(4, len(model_indices))]

    print(f"Evaluating {samples} held-out samples...")

    results = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path),
        "config": config,
        "checkpoint_metadata": checkpoint_metadata,
        "dataset": {
            "metadata": str(metadata_path),
            "total_samples": len(dataset),
            "evaluated_indices": [model_indices[0], model_indices[-1]],
        },
        "model": evaluate_model(
            model,
            dataset,
            model_indices,
            resolved_device,
            progress=progress,
        ),
        "vocoder": evaluate_vocoder(
            dataset,
            vocoder_indices,
            n_iter,
            progress=progress,
        ),
        "generated_audio": evaluate_generated_audio(
            model,
            dataset,
            audio_indices,
            resolved_device,
            n_iter,
            progress=progress,
        ),
    }

    return results


def main(argv: list[str] | None = None) -> int:
    """Command line entry point.  Returns a process exit code."""

    args = build_parser().parse_args(argv)

    try:
        results = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            metadata_path=args.metadata,
            samples=args.samples,
            vocoder_samples=args.vocoder_samples,
            n_iter=args.n_iter,
            device=args.device,
            seed=args.seed,
            progress=not args.quiet,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))

    if args.output:
        print(f"\nSaved results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
