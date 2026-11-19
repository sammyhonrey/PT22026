"""
Training loop for VoiceForge.

The loop supports both model generations:

* the attention model (``voiceforge.model``) uses ``forward_train`` and a
  richer objective: Mel loss (before and after the postnet), stop-token
  loss, duration loss and guided-attention loss;
* the legacy baseline uses the original objective (Mel loss + duration).

The loop also supports validation, checkpointing, JSON history and optional
mixed precision.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from voiceforge.checkpoint import get_model_config


def _masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """L1 loss that ignores padded frames."""

    frames = torch.arange(target.shape[2], device=target.device)
    mask = (frames.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).float()

    error = (prediction - target).abs() * mask

    return error.sum() / (mask.sum() * target.shape[1])


def guided_attention_loss(
    attention: torch.Tensor,
    text_lengths: torch.Tensor,
    sigma: float = 0.4,
) -> torch.Tensor:
    """
    Encourage attention to follow a straight diagonal line.

    For frame ``t`` out of ``T`` and character ``j`` out of ``L`` the loss
    penalises attention far from ``t/T == j/L``.  This single term is what
    makes attention become monotonic quickly on small datasets, and without
    it a small Tacotron often never learns usable alignment.
    """

    frames = attention.shape[1]
    characters = attention.shape[2]

    frame_index = torch.arange(frames, device=attention.device).unsqueeze(1)
    character_index = torch.arange(characters, device=attention.device).unsqueeze(0)

    frame_index = frame_index.float() / max(frames, 1)
    character_index = character_index.float() / max(characters, 1)

    distance = (frame_index - character_index).pow(2) / (2 * sigma**2)
    penalty = 1.0 - torch.exp(-distance)

    mask = (
        torch.arange(characters, device=attention.device).unsqueeze(0)
        < text_lengths.unsqueeze(1)
    )

    weighted = attention * penalty.unsqueeze(0)
    weighted = weighted.masked_fill(~mask.unsqueeze(1), 0.0)

    return weighted.sum() / (mask.sum() * frames)


def compute_losses(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    guided_attention_weight: float = 0.5,
    stop_weight: float = 1.0,
    duration_weight: float = 0.1,
    guided_attention_sigma: float = 0.4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the training loss for one batch and return per-term values.
    """

    texts = batch["text"].to(device)
    text_lengths = batch["text_lengths"].to(device)
    targets = batch["mel"].to(device)
    mel_lengths = batch["mel_lengths"].to(device)

    if hasattr(model, "forward_train"):
        output = model.forward_train(texts, text_lengths, targets, mel_lengths)

        mel_loss = _masked_l1(output["mel"], targets, mel_lengths)
        postnet_loss = _masked_l1(output["mel_postnet"], targets, mel_lengths)

        # Stop target: 1 on the final real frame, 0 before it.
        stop_targets = torch.zeros_like(output["stop_logits"])
        last_index = (mel_lengths - 1).clamp(min=0)
        stop_targets.scatter_(1, last_index.unsqueeze(1), 1.0)

        frame_positions = torch.arange(
            stop_targets.shape[1], device=device
        ).unsqueeze(0)
        valid_frames = frame_positions < mel_lengths.unsqueeze(1)

        stop_errors = nn.functional.binary_cross_entropy_with_logits(
            output["stop_logits"],
            stop_targets,
            reduction="none",
        )
        stop_loss = (stop_errors * valid_frames).sum() / valid_frames.sum()

        memory_free_duration = model.predict_log_mel_lengths(
            texts,
            text_lengths,
        )
        duration_loss = nn.functional.mse_loss(
            memory_free_duration,
            torch.log(mel_lengths.float()),
        )

        attention_loss = guided_attention_loss(
            output["attention"],
            text_lengths,
            sigma=guided_attention_sigma,
        )

        total = (
            mel_loss
            + postnet_loss
            + stop_weight * stop_loss
            + duration_weight * duration_loss
            + guided_attention_weight * attention_loss
        )

        return total, {
            "mel": float(mel_loss.detach()),
            "postnet": float(postnet_loss.detach()),
            "stop": float(stop_loss.detach()),
            "duration": float(duration_loss.detach()),
            "attention": float(attention_loss.detach()),
        }

    # --- Legacy baseline path -----------------------------------------
    predictions = model(texts, text_lengths, mel_lengths)

    if predictions.shape != targets.shape:
        raise ValueError(
            "Prediction and target shapes do not match: "
            f"{predictions.shape} vs {targets.shape}"
        )

    mel_loss = _masked_l1(predictions, targets, mel_lengths)

    total = mel_loss
    terms = {"mel": float(mel_loss.detach())}

    if hasattr(model, "predict_log_mel_lengths"):
        predicted_log_lengths = model.predict_log_mel_lengths(
            texts, text_lengths
        )
        duration_loss = nn.functional.mse_loss(
            predicted_log_lengths,
            torch.log(mel_lengths.float()),
        )
        total = mel_loss + duration_weight * duration_loss
        terms["duration"] = float(duration_loss.detach())

    return total, terms


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
):
    """
    Linear warm-up followed by cosine decay to ``minimum_learning_rate``.

    Warm-up avoids large early updates while the attention is still random;
    the cosine tail lets the model settle instead of oscillating around the
    optimum at a constant learning rate.
    """

    warmup = max(int(warmup_steps), 0)
    floor = min(max(minimum_learning_rate / peak_learning_rate, 0.0), 1.0)

    def factor(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup

        progress = (step - warmup) / max(total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    history: list[dict[str, Any]],
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint that can rebuild the model exactly."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": train_loss,
        "model_config": get_model_config(model),
        "history": history,
    }

    if extra_metadata:
        payload["training_config"] = dict(extra_metadata)

    if val_loss is not None:
        payload["val_loss"] = val_loss

    # Write to a temporary file first so a crash cannot corrupt the
    # previous good checkpoint.
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)

    return checkpoint_path


def evaluate_validation(
    model: nn.Module,
    dataloader,
    device: torch.device,
    guided_attention_weight: float = 0.5,
    stop_weight: float = 1.0,
    duration_weight: float = 0.1,
    guided_attention_sigma: float = 0.4,
) -> float:
    """Return the mean validation loss without updating any weights."""

    model.eval()
    total = 0.0
    batches = 0

    with torch.inference_mode():
        for batch in dataloader:
            loss, _ = compute_losses(
                model,
                batch,
                device,
                guided_attention_weight=guided_attention_weight,
                stop_weight=stop_weight,
                duration_weight=duration_weight,
                guided_attention_sigma=guided_attention_sigma,
            )
            total += float(loss)
            batches += 1

    model.train()

    if batches == 0:
        raise ValueError("The validation dataloader is empty")

    return total / batches


def train_model(
    model: nn.Module,
    dataloader,
    epochs: int = 1,
    learning_rate: float = 0.001,
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    validation_dataloader=None,
    history_path: str | Path | None = None,
    guided_attention_weight: float = 0.5,
    stop_weight: float = 1.0,
    duration_weight: float = 0.1,
    guided_attention_sigma: float = 0.4,
    grad_clip: float = 1.0,
    amp: bool = False,
    seed: int | None = None,
    log_every: int = 0,
    warmup_steps: int = 0,
    min_learning_rate: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[float]:
    """
    Train ``model`` and return the average training loss of every epoch.

    Parameters
    ----------
    dataloader:
        Batches produced by :func:`voiceforge.batching.collate_batch`.
    checkpoint_path:
        Where to save the model.  With a validation loader the best epoch is
        kept; without one, every epoch overwrites the file.
    validation_dataloader:
        Optional held-out loader used to measure generalisation.
    history_path:
        Optional JSON file receiving a detailed per-epoch history.
    amp:
        Enable CUDA mixed precision.  Off by default because this model is
        small and dominated by per-frame Python overhead.
    """

    if epochs < 1:
        raise ValueError("Epochs must be at least 1")

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than zero"
        )

    if seed is not None:
        torch.manual_seed(seed)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Optional warm-up + cosine decay.  With no warm-up and no minimum the
    # learning rate stays constant, which keeps the old behaviour.
    scheduler = None

    if warmup_steps > 0 or min_learning_rate is not None:
        try:
            total_steps = epochs * len(dataloader)
        except TypeError:
            total_steps = 0

        if total_steps > 0:
            scheduler = _build_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                peak_learning_rate=learning_rate,
                minimum_learning_rate=(
                    min_learning_rate
                    if min_learning_rate is not None
                    else learning_rate * 0.1
                ),
            )

    history: list[dict[str, Any]] = []
    epoch_losses: list[float] = []

    best_validation = float("inf")

    for epoch in range(epochs):
        model.train()

        start = time.time()

        total_loss = 0.0
        batch_count = 0
        term_totals: dict[str, float] = {}

        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss, terms = compute_losses(
                        model,
                        batch,
                        device,
                        guided_attention_weight=guided_attention_weight,
                        stop_weight=stop_weight,
                        duration_weight=duration_weight,
                        guided_attention_sigma=guided_attention_sigma,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, terms = compute_losses(
                    model,
                    batch,
                    device,
                    guided_attention_weight=guided_attention_weight,
                    stop_weight=stop_weight,
                    duration_weight=duration_weight,
                    guided_attention_sigma=guided_attention_sigma,
                )
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if not torch.isfinite(loss):
                raise ValueError("Training loss is not finite")

            total_loss += float(loss.detach())
            batch_count += 1

            for key, value in terms.items():
                term_totals[key] = term_totals.get(key, 0.0) + value

            if log_every and batch_count % log_every == 0:
                elapsed = time.time() - start
                rate = batch_count / max(elapsed, 1e-6)
                remaining = (len(dataloader) - batch_count) / max(rate, 1e-6)
                print(
                    f"    batch {batch_count}/{len(dataloader)} "
                    f"loss {float(loss.detach()):.4f} "
                    f"({rate:.1f} batches/s, {remaining:.0f}s left)",
                    flush=True,
                )

        if batch_count == 0:
            raise ValueError(
                "The training dataloader is empty"
            )

        average_loss = total_loss / batch_count
        epoch_losses.append(average_loss)

        validation_loss = None

        if validation_dataloader is not None:
            validation_loss = evaluate_validation(
                model,
                validation_dataloader,
                device,
                guided_attention_weight=guided_attention_weight,
                stop_weight=stop_weight,
                duration_weight=duration_weight,
                guided_attention_sigma=guided_attention_sigma,
            )

        entry = {
            "epoch": epoch + 1,
            "train_loss": round(average_loss, 6),
            "val_loss": round(validation_loss, 6)
            if validation_loss is not None
            else None,
            "seconds": round(time.time() - start, 1),
            "learning_rate": round(optimizer.param_groups[0]["lr"], 8),
            "terms": {
                key: round(value / batch_count, 6)
                for key, value in term_totals.items()
            },
        }

        history.append(entry)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
            + (
                f" - Val: {validation_loss:.4f}"
                if validation_loss is not None
                else ""
            )
            + f" - {entry['seconds']:.0f}s",
            flush=True,
        )

        if checkpoint_path is not None:
            should_save = True

            if validation_loss is not None:
                should_save = validation_loss < best_validation
                if should_save:
                    best_validation = validation_loss

            if should_save or validation_dataloader is None:
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch + 1,
                    average_loss,
                    validation_loss,
                    history,
                    extra_metadata=extra_metadata,
                )

        if history_path is not None:
            history_file = Path(history_path)
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history_file.write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )

    return epoch_losses
