"""
Checkpoint helpers for VoiceForge.

A checkpoint is a dictionary produced by the training script:

    model_state_dict    the neural network weights
    model_config        the hyper-parameters needed to rebuild the network
    epoch / loss        training metadata (optional)
    history             per-epoch training history (optional)

Two architectures are supported:

* ``tacotron_lite`` (``voiceforge.model``) - the current attention model.
* ``legacy_interpolate`` (``voiceforge.model_legacy``) - the original
  baseline, kept so the old checkpoint still loads and can be compared.

Checkpoints created before ``model_config`` existed are treated as legacy
and their configuration is inferred from the weight shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from voiceforge.architectures import (
    ARCH_LEGACY,
    ARCH_TACOTRON,
    SUPPORTED_ARCHITECTURES,
)
from voiceforge.model import AcousticModel, ModelConfig
from voiceforge.model_legacy import LegacyAcousticModel


def infer_model_config(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """
    Rebuild the *legacy* configuration from the weight shapes.

    This allows old checkpoints, which did not store a configuration, to be
    loaded without knowing how they were created.
    """

    if "embedding.weight" not in state_dict:
        raise ValueError(
            "Checkpoint does not contain an 'embedding.weight' tensor "
            "and is not a recognised VoiceForge model."
        )

    vocab_size, embedding_dim = state_dict["embedding.weight"].shape

    if "mel_projection.weight" in state_dict:
        n_mels, doubled_hidden = state_dict["mel_projection.weight"].shape
        hidden_dim = doubled_hidden // 2
    else:
        n_mels, hidden_dim = 80, 128

    return {
        "architecture": ARCH_LEGACY,
        "vocab_size": int(vocab_size),
        "n_mels": int(n_mels),
        "embedding_dim": int(embedding_dim),
        "hidden_dim": int(hidden_dim),
    }


def get_model_config(model: torch.nn.Module) -> dict[str, Any]:
    """
    Return the configuration dictionary for a model instance.

    Used by the training script when saving a checkpoint.
    """

    if hasattr(model, "config"):
        return dict(model.config.to_dict())

    return infer_model_config(model.state_dict())


def build_model(config: Mapping[str, Any]) -> torch.nn.Module:
    """
    Create an untrained model from a configuration dictionary.
    """

    architecture = config.get("architecture", ARCH_LEGACY)

    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(
            f"Unknown model architecture: {architecture!r}. "
            f"Supported: {', '.join(SUPPORTED_ARCHITECTURES)}."
        )

    if architecture == ARCH_TACOTRON:
        resolved = ModelConfig.from_dict(config)

        return AcousticModel(
            vocab_size=resolved.vocab_size,
            n_mels=resolved.n_mels,
            embedding_dim=resolved.embedding_dim,
            hidden_dim=resolved.hidden_dim,
            attention_dim=resolved.attention_dim,
            dropout=resolved.dropout,
            postnet_layers=resolved.postnet_layers,
            location_filters=resolved.location_filters,
            location_kernel=resolved.location_kernel,
        )

    return LegacyAcousticModel(
        vocab_size=int(config["vocab_size"]),
        n_mels=int(config["n_mels"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load and validate a VoiceForge checkpoint file.

    Returns a dictionary with ``state_dict``, ``config`` and ``metadata``.
    """

    path = Path(checkpoint_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    raw = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(raw, dict):
        raise ValueError(
            "Checkpoint must be a dictionary containing at least "
            "'model_state_dict'."
        )

    if "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
        config = raw.get("model_config")
    else:
        # Some checkpoints store the bare state dict.  Only accept that when
        # every value really is a tensor, otherwise a metadata-only
        # dictionary would be mistaken for weights.
        state_dict = raw
        config = None

        if not all(torch.is_tensor(value) for value in state_dict.values()):
            raise ValueError(
                "Checkpoint does not contain a valid 'model_state_dict'."
            )

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(
            "Checkpoint does not contain a valid 'model_state_dict'."
        )

    if config is None:
        config = infer_model_config(state_dict)

    architecture = config.get("architecture", ARCH_LEGACY)

    if architecture == ARCH_TACOTRON and "hidden_dim" not in config:
        raise ValueError(
            "Attention-based checkpoints must store a complete "
            "'model_config' dictionary. This checkpoint does not, so it "
            "cannot be rebuilt reliably."
        )

    metadata = {
        "epoch": raw.get("epoch"),
        "loss": raw.get("loss"),
        "val_loss": raw.get("val_loss"),
        "history": raw.get("history"),
    }

    return {
        "state_dict": state_dict,
        "config": dict(config),
        "metadata": metadata,
    }


def load_model(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    """
    Load a trained model and its configuration from a checkpoint.

    Returns ``(model, config, metadata)``.  The model is put in evaluation
    mode so it can be used for inference straight away.
    """

    contents = load_checkpoint(checkpoint_path, map_location=map_location)

    model = build_model(contents["config"])

    try:
        model.load_state_dict(contents["state_dict"])
    except RuntimeError as error:
        message = str(error)

        # The early smoke-test checkpoint predates the duration predictor,
        # so its only mismatch is a set of missing duration weights.
        # Anything else (for example weights from a different architecture)
        # is a genuine mismatch and must use the generic message.
        is_pre_duration_checkpoint = (
            "Missing key(s)" in message
            and "duration_projection" in message
            and "Unexpected key(s)" not in message
        )

        if is_pre_duration_checkpoint:
            raise ValueError(
                "This checkpoint was created before the duration predictor "
                "existed (early smoke-test checkpoints have this shape), so "
                "it cannot be used with the current model. Train a new "
                "checkpoint with scripts/train.py, or use "
                "checkpoints/voiceforge.pt (baseline) or "
                "checkpoints/voiceforge_attn.pt (current model)."
            ) from error

        raise ValueError(
            "Checkpoint weights do not match the model architecture. "
            "The checkpoint may have been produced by a different version "
            f"of VoiceForge. Original error: {error}"
        ) from error

    model.eval()

    return model, contents["config"], contents["metadata"]
