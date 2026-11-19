"""Train the VoiceForge acoustic model.

Examples
--------
Quick smoke test (a few minutes)::

    python -m scripts.train --max-samples 128 --epochs 1 --batch-size 16

Real training run::

    python -m scripts.train \
        --epochs 20 \
        --batch-size 64 \
        --max-seconds 5 \
        --max-samples 6000 \
        --checkpoint checkpoints/voiceforge_attn.pt \
        --history outputs/training_history.json

The script never overwrites the baseline checkpoint unless you point
``--checkpoint`` at it explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader, Subset

from voiceforge.architectures import ARCH_LEGACY, ARCH_TACOTRON
from voiceforge.batching import LengthBucketSampler, collate_batch
from voiceforge.checkpoint import build_model, load_model
from voiceforge.dataset import CachedMelDataset, LJSpeechDataset
from voiceforge.model import ModelConfig
from voiceforge.text import CHAR_TO_ID
from voiceforge.training import train_model

DEFAULT_METADATA = "data/LJSpeech-1.1/LJSpeech-1.1/metadata.csv"

# The evaluator always uses the last 64 rows of metadata.csv.  Training must
# never select them, otherwise the "held-out" numbers stop being held out.
EVALUATION_WINDOW = 64


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the VoiceForge acoustic model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--arch", default=ARCH_TACOTRON,
                        choices=[ARCH_TACOTRON, "legacy_interpolate"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Training samples to use (0 = everything).")
    parser.add_argument("--val-samples", type=int, default=64,
                        help="Held-out samples used for validation.")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Keep only clips shorter than this (0 = no limit).")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Truncate Mel spectrograms to this many frames.")
    parser.add_argument("--checkpoint", default="checkpoints/voiceforge_attn.pt")
    parser.add_argument("--history", default="outputs/training_history.json")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--guided-weight", type=float, default=0.5)
    parser.add_argument("--guided-sigma", type=float, default=0.4,
                        help="Width of the guided-attention diagonal band.")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear learning-rate warm-up steps (0 = off).")
    parser.add_argument("--min-lr", type=float, default=None,
                        help="Final learning rate of the cosine decay.")
    parser.add_argument("--cache-dir", default="",
                        help="Cache log-Mel tensors here ('' disables).")
    parser.add_argument("--no-bucket", action="store_true",
                        help="Disable length-bucketed batch construction.")
    parser.add_argument("--keep-awake", action="store_true",
                        help="Stop Windows from sleeping during training.")
    parser.add_argument("--amp", action="store_true",
                        help="Enable CUDA mixed precision.")
    parser.add_argument("--log-every", type=int, default=50)

    return parser


def keep_system_awake() -> bool:
    """
    Best-effort request that Windows keeps this session awake.

    A long training run is easily killed by the machine going to sleep, and
    the request is per-process, so it does not change any system settings.
    Returns True when the request was made.
    """

    if sys.platform != "win32":
        return False

    try:
        import ctypes

        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001

        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
        return True
    except Exception:
        return False


def audio_lengths(dataset: LJSpeechDataset, indices: list[int]) -> list[int]:
    """Mel-frame length of every selected row, without decoding audio."""

    lengths = []

    for index in indices:
        _, audio_path = dataset.samples[index]
        duration = sf.info(str(audio_path)).duration
        lengths.append(max(4, round(duration * 22050 / 256)))

    return lengths


def select_indices(
    dataset: LJSpeechDataset,
    train_count: int,
    val_count: int,
    max_seconds: float,
) -> tuple[list[int], list[int]]:
    """
    Choose training and validation rows by original metadata position.

    Selecting by row index (instead of filtering the dataset first) keeps the
    identity of each row, so the evaluator's held-out window can never be
    used for training.
    """

    selected: list[int] = []

    for index, (_, audio_path) in enumerate(dataset.samples):
        if max_seconds > 0:
            duration = sf.info(str(audio_path)).duration
            if duration > max_seconds:
                continue

        selected.append(index)

        if len(selected) >= train_count + val_count:
            break

    limit = len(dataset) - EVALUATION_WINDOW

    if selected and max(selected) >= limit:
        raise ValueError(
            "Selected training rows overlap the evaluator's held-out "
            f"window (index {limit} and above). Use fewer samples or a "
            "shorter --max-seconds."
        )

    if len(selected) < train_count:
        raise ValueError(
            f"Only {len(selected)} rows matched the filters, "
            f"but --max-samples is {train_count}."
        )

    return selected[:train_count], selected[train_count:train_count + val_count]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.batch_size < 1:
        raise ValueError("Batch size must be at least 1")

    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1")

    if args.max_samples < 0:
        raise ValueError("Maximum samples cannot be negative")

    if args.max_frames < 0:
        raise ValueError("Maximum frames cannot be negative")

    if args.val_samples < 0:
        raise ValueError("Validation samples cannot be negative")

    if args.max_seconds < 0:
        raise ValueError("Maximum clip length cannot be negative")

    if args.log_every < 0:
        raise ValueError("--log-every cannot be negative")

    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps cannot be negative")

    if args.keep_awake and not keep_system_awake():
        print("Note: could not request keep-awake on this platform.")

    metadata_path = Path(args.metadata)

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("Loading LJSpeech dataset...")

    dataset = LJSpeechDataset(
        metadata_path,
        max_mel_frames=args.max_frames or None,
        strict=False,
    )

    print(f"Rows in metadata: {len(dataset)}")

    total_rows = len(dataset) - EVALUATION_WINDOW
    train_count = min(args.max_samples, total_rows) if args.max_samples else total_rows
    val_count = min(args.val_samples, max(total_rows - train_count, 0))

    train_indices, val_indices = select_indices(
        dataset,
        train_count,
        val_count,
        args.max_seconds,
    )

    print(
        f"Training samples: {len(train_indices)} | "
        f"validation samples: {len(val_indices)} | "
        f"max clip length: {args.max_seconds or 'unlimited'} s"
    )

    # Optional on-disk Mel cache.  It wraps the base dataset before the
    # Subset so indices stay absolute, and it changes nothing about the data
    # itself - only how often the spectrogram is computed.
    cached_dataset = dataset

    if args.cache_dir:
        cached_dataset = CachedMelDataset(
            dataset,
            args.cache_dir,
            verbose=True,
        )
        print(f"Mel cache: {cached_dataset.cache_dir}")

    train_subset = Subset(cached_dataset, train_indices)

    if args.no_bucket or len(train_indices) < args.batch_size:
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_batch,
            # A final batch of one sample would break BatchNorm.
            drop_last=len(train_indices) >= args.batch_size,
        )
    else:
        lengths = audio_lengths(dataset, train_indices)

        sampler = LengthBucketSampler(
            lengths,
            batch_size=args.batch_size,
            seed=args.seed,
            drop_last=True,
        )

        train_loader = DataLoader(
            train_subset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=collate_batch,
        )

        print(
            f"Length-bucketed batches: mean "
            f"{sum(lengths) / len(lengths):.0f} frames"
        )

    val_loader = (
        DataLoader(
            Subset(cached_dataset, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )
        if val_indices
        else None
    )

    if args.arch == ARCH_TACOTRON:
        config = ModelConfig(
            vocab_size=len(CHAR_TO_ID),
            n_mels=80,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
        ).to_dict()
    else:
        config = {
            "architecture": ARCH_LEGACY,
            "vocab_size": len(CHAR_TO_ID),
            "n_mels": 80,
            "embedding_dim": 128,
            "hidden_dim": 128,
        }

    # Build the model through the same factory that checkpoint loading uses.
    # This guarantees that a trained model and an inference model are always
    # constructed from identical configuration code.
    model = build_model(config)

    parameters = sum(p.numel() for p in model.parameters())

    print(f"Architecture: {args.arch}")
    print(f"Parameters: {parameters:,}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print(
        f"Batches per epoch: {len(train_loader)} | "
        f"epochs: {args.epochs} | batch size: {args.batch_size}"
    )

    training_config = {
        "dataset": str(metadata_path),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "min_learning_rate": args.min_lr,
        "guided_attention_weight": args.guided_weight,
        "guided_attention_sigma": args.guided_sigma,
        "max_seconds": args.max_seconds,
        "max_frames": args.max_frames,
        "seed": args.seed,
        "bucketed_batches": not args.no_bucket,
        "mel_cache": str(args.cache_dir) if args.cache_dir else None,
        "device": str(device),
        # str() is required: torch.__version__ is a TorchVersion object,
        # which cannot be loaded back with weights_only=True.
        "torch": str(torch.__version__),
    }

    print(
        f"Learning rate: {args.learning_rate} "
        f"(warm-up {args.warmup_steps} steps, min {args.min_lr})"
    )
    print(
        f"Guided attention: weight {args.guided_weight}, "
        f"sigma {args.guided_sigma}"
    )

    started = time.time()

    history = train_model(
        model=model,
        dataloader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=str(device),
        checkpoint_path=args.checkpoint,
        validation_dataloader=val_loader,
        history_path=args.history,
        guided_attention_weight=args.guided_weight,
        guided_attention_sigma=args.guided_sigma,
        warmup_steps=args.warmup_steps,
        min_learning_rate=args.min_lr,
        extra_metadata=training_config,
        amp=args.amp,
        seed=args.seed,
        log_every=args.log_every,
    )

    elapsed = time.time() - started

    if device.type == "cuda":
        print(
            f"Peak GPU memory: "
            f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )

    print("\nTraining completed.")
    print(f"Total time: {elapsed / 60:.1f} minutes")

    for index, loss in enumerate(history, start=1):
        print(f"Epoch {index}: loss = {loss:.4f}")

    # Load the checkpoint back exactly as the inference pipeline will.  If
    # training and inference ever disagreed about the architecture, this
    # check fails here instead of in the application.
    print("\nVerifying the saved checkpoint loads for inference...")

    restored, restored_config, restored_metadata = load_model(
        args.checkpoint,
        map_location="cpu",
    )

    restored_parameters = sum(p.numel() for p in restored.parameters())

    if restored_config.get("architecture") != config["architecture"]:
        raise RuntimeError(
            "Saved checkpoint architecture does not match the trained model"
        )

    if restored_parameters != parameters:
        raise RuntimeError(
            f"Saved checkpoint has {restored_parameters:,} parameters, "
            f"but the trained model has {parameters:,}"
        )

    print(
        f"Checkpoint OK: {restored_config['architecture']}, "
        f"{restored_parameters:,} parameters, "
        f"epoch {restored_metadata['epoch']}"
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"History: {args.history}")

    if Path(args.history).is_file():
        entries = json.loads(Path(args.history).read_text(encoding="utf-8"))
        print(json.dumps(entries[-1], indent=2))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
