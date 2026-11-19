
"""
Original VoiceForge acoustic model (baseline, kept for comparison).

This module is the *first* model used by the project.  It encodes the text
with a bidirectional GRU and produces a Mel spectrogram by stretching the
character-level features over time with ``F.interpolate``.

That approach has no alignment mechanism: every character is smeared evenly
across the utterance, so the model cannot learn which audio frames belong to
which sound.  It is preserved here unchanged so the old checkpoint keeps
working and so the new attention-based model can be compared against it with
the same evaluation script.

Do not modify the model code below: it is the reference point for the
"before" measurements in docs/evaluation.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from voiceforge.architectures import ARCH_LEGACY
from voiceforge.text import CHAR_TO_ID


class AcousticModel(nn.Module):
    """
    Baseline model: GRU encoder + linear time interpolation.

    ``forward(text, text_lengths, mel_lengths)`` returns a Mel spectrogram
    with exactly ``mel_lengths`` frames per sample.
    """
    def __init__(
        self,
        vocab_size: int = len(CHAR_TO_ID),
        n_mels: int = 80,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

        self.mel_projection = nn.Linear(
            hidden_dim * 2,
            n_mels,
        )

        # Predicts the logarithm of the speech duration.
        self.duration_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:

        if text.ndim != 2:
            raise ValueError(
                "Text must have shape (batch, text_length)"
            )

        if text_lengths.ndim != 1:
            raise ValueError(
                "Text lengths must be one-dimensional"
            )

        if text.shape[0] != text_lengths.shape[0]:
            raise ValueError(
                "Text batch and text lengths do not match"
            )

        if torch.any(text_lengths <= 0):
            raise ValueError(
                "Text lengths must be greater than zero"
            )

        if torch.any(text_lengths > text.shape[1]):
            raise ValueError(
                "Text lengths cannot exceed the padded text length"
            )

        embedded = self.embedding(text)

        packed = pack_padded_sequence(
            embedded,
            text_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.encoder(packed)

        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        return encoded

    def _predict_log_duration(
        self,
        encoded: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:

        positions = torch.arange(
            encoded.shape[1],
            device=encoded.device,
        )

        mask = (
            positions.unsqueeze(0)
            < text_lengths.to(encoded.device).unsqueeze(1)
        )

        mask = mask.unsqueeze(-1)

        masked_encoded = encoded * mask

        pooled = masked_encoded.sum(dim=1)

        pooled = pooled / text_lengths.to(
            encoded.device
        ).unsqueeze(1)

        return self.duration_projection(pooled).squeeze(-1)

    def predict_log_mel_lengths(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:

        encoded = self._encode(text, text_lengths)

        return self._predict_log_duration(
            encoded,
            text_lengths,
        )

    def predict_mel_lengths(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:

        log_lengths = self.predict_log_mel_lengths(
            text,
            text_lengths,
        )

        return (
            torch.exp(log_lengths)
            .round()
            .clamp(min=1, max=5000)
            .long()
        )

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:

        encoded = self._encode(text, text_lengths)

        if mel_lengths is None:
            log_lengths = self._predict_log_duration(
                encoded,
                text_lengths,
            )

            mel_lengths = (
                torch.exp(log_lengths)
                .round()
                .clamp(min=1, max=5000)
                .long()
            )
        else:
            if mel_lengths.ndim != 1:
                raise ValueError(
                    "Mel lengths must be one-dimensional"
                )

            if text.shape[0] != mel_lengths.shape[0]:
                raise ValueError(
                    "Text batch and Mel lengths do not match"
                )

            if torch.any(mel_lengths <= 0):
                raise ValueError(
                    "Mel lengths must be greater than zero"
                )

            mel_lengths = mel_lengths.to(
                device=encoded.device,
                dtype=torch.long,
            )

        mel_sequences = []

        for index in range(text.shape[0]):
            valid_text_length = int(
                text_lengths[index].item()
            )

            target_mel_length = int(
                mel_lengths[index].item()
            )

            hidden = encoded[index, :valid_text_length]

            hidden = hidden.transpose(0, 1).unsqueeze(0)

            expanded = F.interpolate(
                hidden,
                size=target_mel_length,
                mode="linear",
                align_corners=False,
            )

            expanded = (
                expanded.squeeze(0)
                .transpose(0, 1)
            )

            mel = self.mel_projection(expanded)

            mel_sequences.append(mel)

        padded_mels = nn.utils.rnn.pad_sequence(
            mel_sequences,
            batch_first=True,
            padding_value=0.0,
        )

        return padded_mels.transpose(1, 2)

#: Explicit alias used by the checkpoint loader and by tests.
LegacyAcousticModel = AcousticModel
