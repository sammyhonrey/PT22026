"""
VoiceForge acoustic model (Tacotron-lite).

The model turns a sequence of character IDs into a Mel spectrogram.  It is a
small, from-scratch re-implementation of the core ideas in Tacotron
(Wang et al., 2017) and Tacotron 2 (Shen et al., 2018), reduced to a size
that trains on a single 6 GB laptop GPU.

Pipeline
--------

::

    characters
        │  embedding
        ▼
    conv stack (3 x Conv1d + BatchNorm + ReLU)
        │
        ▼
    bidirectional GRU  ──────────────► encoder memory  (batch, text_len, hidden)
        │
        ▼
    location-sensitive attention  ◄──── previous attention weights
        │  context vector
        ▼
    decoder GRU (one step per Mel frame)
        │
        ├──► linear layer ──► Mel frame          (before postnet)
        └──► linear layer ──► stop probability
        │
        ▼
    postnet (4 x Conv1d) ──► residual added to the Mel frames

Why attention instead of interpolation
--------------------------------------
The previous baseline stretched every character evenly over the whole
utterance.  Attention instead learns, frame by frame, which characters are
being spoken.  That is what allows pronunciation and timing to be learned at
all.  The attention weights are also the alignment information that a real
TTS system needs.

Training and inference modes
----------------------------
``forward_train`` is used during training.  It receives the real Mel
spectrogram and runs the decoder with *teacher forcing* (each step sees the
previous ground-truth frame), which is stable and fast to converge.

``forward`` is the inference path and is deliberately length-controlled: it
uses the duration predictor to decide how many frames to produce, then runs
the decoder autoregressively for exactly that many frames.  Fixing the
length makes an under-trained model much less likely to stop too early or
babble forever.  A fully autoregressive mode using the stop token is also
available through ``generate(..., use_stop_token=True)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from voiceforge.architectures import ARCH_TACOTRON
from voiceforge.text import CHAR_TO_ID

#: Shortest Mel sequence the decoder is allowed to produce.
MIN_MEL_FRAMES = 4

#: Hard safety limit for autoregressive generation (about 25 seconds).
MAX_MEL_FRAMES = 2200


@dataclass(frozen=True)
class ModelConfig:
    """
    Hyper-parameters needed to rebuild the model.

    They are stored inside every checkpoint so that a saved model can be
    reconstructed without knowing how it was trained.
    """

    architecture: str = ARCH_TACOTRON
    vocab_size: int = len(CHAR_TO_ID)
    n_mels: int = 80
    embedding_dim: int = 128
    hidden_dim: int = 256
    attention_dim: int = 128
    dropout: float = 0.1
    postnet_layers: int = 4
    location_filters: int = 32
    location_kernel: int = 31

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelConfig":
        """Build a config from a dictionary, ignoring unknown keys."""

        known = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in known}
        return cls(**filtered)


class LocationSensitiveAttention(nn.Module):
    """
    Attention that also looks at where it attended in the previous step.

    Plain content-based attention can jump around the sentence.  Adding the
    convolution over the previous attention weights (the "location" part)
    encourages attention to move forward smoothly, one sound after another,
    which is what speech requires.

    Reference: Chorowski et al., "Attention-Based Models for Speech
    Recognition" (2015); used by Tacotron 2.
    """

    def __init__(
        self,
        memory_dim: int,
        decoder_dim: int,
        attention_dim: int = 128,
        location_filters: int = 32,
        location_kernel: int = 31,
    ):
        super().__init__()

        # Projects the decoder state (the query).
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)

        # Projects every encoder state (the keys).
        self.memory_layer = nn.Linear(memory_dim, attention_dim, bias=False)

        # Turns the previous attention weights into location features.
        self.location_conv = nn.Conv1d(
            1,
            location_filters,
            kernel_size=location_kernel,
            padding=location_kernel // 2,
            bias=False,
        )
        self.location_layer = nn.Conv1d(
            location_filters,
            attention_dim,
            kernel_size=location_kernel,
            padding=location_kernel // 2,
            bias=False,
        )

        # Produces one score per encoder position.
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor,
        previous_attention: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query:
            Decoder state, shape ``(batch, decoder_dim)``.
        memory:
            Encoder states, shape ``(batch, text_length, memory_dim)``.
        mask:
            Boolean tensor ``(batch, text_length)``, True for real characters.
        previous_attention:
            Attention weights from the previous step, ``(batch, text_length)``.

        Returns
        -------
        (context, attention_weights)
        """

        location_features = self.location_conv(
            previous_attention.unsqueeze(1)
        )
        location_features = self.location_layer(location_features)
        location_features = location_features.transpose(1, 2)

        processed_query = self.query_layer(query).unsqueeze(1)
        processed_memory = self.memory_layer(memory)

        energy = self.v(
            torch.tanh(
                processed_query + processed_memory + location_features
            )
        ).squeeze(-1)

        # Padding positions must never receive attention.
        energy = energy.masked_fill(~mask, -1e4)

        weights = F.softmax(energy, dim=-1)

        context = torch.bmm(weights.unsqueeze(1), memory).squeeze(1)

        return context, weights


class Postnet(nn.Module):
    """
    Convolutional post-processing network.

    A stack of 1-D convolutions predicts a *correction* that is added to the
    decoder output.  It sharpens the spectrogram and removes the blur that a
    frame-independent linear layer leaves behind.
    """

    def __init__(
        self,
        n_mels: int = 80,
        channels: int = 256,
        layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        if layers < 1:
            raise ValueError("Postnet must have at least one layer")

        blocks: list[nn.Module] = []

        for index in range(layers):
            in_channels = n_mels if index == 0 else channels
            out_channels = n_mels if index == layers - 1 else channels

            blocks.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=5,
                    padding=2,
                )
            )
            blocks.append(nn.BatchNorm1d(out_channels))

            if index < layers - 1:
                blocks.append(nn.Tanh())
                blocks.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*blocks)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Return the residual correction for ``(batch, n_mels, frames)``."""

        return self.layers(mel)


class AcousticModel(nn.Module):
    """
    Tacotron-lite: characters in, Mel spectrogram out.

    The public interface is unchanged from the baseline model:

    ``model(text, text_lengths, mel_lengths)`` -> ``(batch, n_mels, frames)``
    ``model(text, text_lengths)``              -> ``(batch, n_mels, frames)``
    ``model.predict_mel_lengths(text, text_lengths)`` -> ``(batch,)``
    """

    def __init__(
        self,
        vocab_size: int = len(CHAR_TO_ID),
        n_mels: int = 80,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        dropout: float = 0.1,
        postnet_layers: int = 4,
        location_filters: int = 32,
        location_kernel: int = 31,
    ):
        super().__init__()

        if vocab_size < 2:
            raise ValueError("Vocabulary must contain at least two symbols")
        if n_mels < 1:
            raise ValueError("Number of Mel bands must be positive")
        if hidden_dim < 8:
            raise ValueError("Hidden size must be at least 8")

        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.dropout = dropout

        self.config = ModelConfig(
            vocab_size=vocab_size,
            n_mels=n_mels,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout,
            postnet_layers=postnet_layers,
            location_filters=location_filters,
            location_kernel=location_kernel,
        )

        # --- Text encoder -------------------------------------------------
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.encoder_convs = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional GRU: reads the sentence in both directions so that
        # each character knows its context.  hidden_dim // 2 per direction
        # keeps the concatenated output at hidden_dim.
        self.encoder_rnn = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.encoder_projection = nn.Linear(hidden_dim, hidden_dim)

        # --- Duration predictor ------------------------------------------
        self.duration_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Decoder ------------------------------------------------------
        # Prenet: two small feed-forward layers that process the previous
        # Mel frame.  Dropout makes the decoder robust to its own errors.
        self.prenet = nn.Sequential(
            nn.Linear(n_mels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.attention = LocationSensitiveAttention(
            memory_dim=hidden_dim,
            decoder_dim=hidden_dim,
            attention_dim=attention_dim,
            location_filters=location_filters,
            location_kernel=location_kernel,
        )

        self.decoder_rnn = nn.GRU(
            hidden_dim + hidden_dim,
            hidden_dim,
            batch_first=True,
        )

        self.mel_projection = nn.Linear(hidden_dim * 2, n_mels)
        self.stop_projection = nn.Linear(hidden_dim * 2, 1)

        self.postnet = Postnet(
            n_mels=n_mels,
            channels=hidden_dim,
            layers=postnet_layers,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validate_text(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> None:
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

    def encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert character IDs into encoder memory.

        Returns ``(memory, mask)`` where ``memory`` has shape
        ``(batch, text_length, hidden_dim)`` and ``mask`` is a boolean tensor
        marking the real (non-padding) characters.
        """

        self._validate_text(text, text_lengths)

        embedded = self.embedding(text)

        # Conv1d expects (batch, channels, length).
        features = embedded.transpose(1, 2)
        features = self.encoder_convs(features)
        features = features.transpose(1, 2)

        packed = pack_padded_sequence(
            features,
            text_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.encoder_rnn(packed)

        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
        )

        memory = self.encoder_projection(encoded)

        positions = torch.arange(
            memory.shape[1],
            device=memory.device,
        )
        mask = positions.unsqueeze(0) < text_lengths.unsqueeze(1)

        return memory, mask

    def _predict_log_duration(
        self,
        memory: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Masked mean-pool the encoder memory, then predict log frames."""

        positions = torch.arange(memory.shape[1], device=memory.device)
        mask = (
            positions.unsqueeze(0)
            < text_lengths.to(memory.device).unsqueeze(1)
        ).unsqueeze(-1)

        pooled = (memory * mask).sum(dim=1) / text_lengths.to(
            memory.device
        ).unsqueeze(1)

        return self.duration_projection(pooled).squeeze(-1)

    # ------------------------------------------------------------------
    # Duration prediction
    # ------------------------------------------------------------------
    def predict_log_mel_lengths(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict ``log(number of Mel frames)`` for each sentence."""

        if memory is None:
            memory, _ = self.encode(text, text_lengths)

        return self._predict_log_duration(memory, text_lengths)

    def predict_mel_lengths(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the number of Mel frames, clamped to a safe range."""

        log_lengths = self.predict_log_mel_lengths(
            text,
            text_lengths,
            memory=memory,
        )

        return (
            torch.exp(log_lengths)
            .round()
            .clamp(min=MIN_MEL_FRAMES, max=5000)
            .long()
        )

    # ------------------------------------------------------------------
    # Training pass (teacher forcing)
    # ------------------------------------------------------------------
    def forward_train(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_targets: torch.Tensor,
        mel_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Run the decoder with teacher forcing and return everything the
        training loss needs.

        Returns a dictionary with:

        ``mel``          decoder output before the postnet
        ``mel_postnet``  decoder output after the postnet
        ``stop_logits``  raw stop-token scores, shape ``(batch, frames)``
        ``attention``    attention weights, shape ``(batch, frames, text_len)``
        """

        memory, mask = self.encode(text, text_lengths)

        batch_size, _, frames = mel_targets.shape
        device = mel_targets.device

        # The first decoder input is the "GO" frame (all zeros); after that
        # the decoder sees the previous ground-truth frame.
        go_frame = torch.zeros(
            batch_size,
            self.n_mels,
            1,
            device=device,
            dtype=mel_targets.dtype,
        )
        decoder_inputs = torch.cat(
            [go_frame, mel_targets[:, :, :-1]],
            dim=2,
        )

        # The prenet does not depend on the decoder state, so all frames can
        # be processed in one vectorised call.
        prenet_outputs = self.prenet(decoder_inputs.transpose(1, 2))

        hidden = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=mel_targets.dtype,
        )
        attention = torch.zeros(
            batch_size,
            memory.shape[1],
            device=device,
            dtype=mel_targets.dtype,
        )

        mel_outputs: list[torch.Tensor] = []
        stop_logits: list[torch.Tensor] = []
        attention_weights: list[torch.Tensor] = []

        for index in range(frames):
            context, attention = self.attention(
                hidden,
                memory,
                mask,
                attention,
            )

            rnn_input = torch.cat(
                [prenet_outputs[:, index], context],
                dim=-1,
            ).unsqueeze(1)

            output, new_hidden = self.decoder_rnn(
                rnn_input,
                hidden.unsqueeze(0),
            )

            hidden = new_hidden.squeeze(0)

            decoder_state = torch.cat(
                [output.squeeze(1), context],
                dim=-1,
            )

            mel_outputs.append(self.mel_projection(decoder_state))
            stop_logits.append(
                self.stop_projection(decoder_state).squeeze(-1)
            )
            attention_weights.append(attention)

        mel = torch.stack(mel_outputs, dim=2)
        mel_postnet = mel + self.postnet(mel)

        return {
            "mel": mel,
            "mel_postnet": mel_postnet,
            "stop_logits": torch.stack(stop_logits, dim=1),
            "attention": torch.stack(attention_weights, dim=1),
        }

    # ------------------------------------------------------------------
    # Inference pass
    # ------------------------------------------------------------------
    # Note: this method is intentionally *not* decorated with
    # ``torch.inference_mode`` so that ``forward`` stays differentiable and
    # can be used by training code and tests.  Inference callers
    # (VoiceForgeTTS, scripts/evaluate.py) already run inside
    # ``torch.inference_mode``.
    def generate(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        lengths: torch.Tensor | None = None,
        max_frames: int = MAX_MEL_FRAMES,
        use_stop_token: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a Mel spectrogram without ground-truth Mel frames.

        The model is temporarily switched to evaluation mode so that
        dropout is disabled and BatchNorm uses its running statistics.  This
        also keeps generation valid for a single sample, which BatchNorm in
        training mode would reject.  The previous mode is restored on exit.
        """

        was_training = self.training
        self.eval()

        try:
            return self._generate_impl(
                text,
                text_lengths,
                lengths=lengths,
                max_frames=max_frames,
                use_stop_token=use_stop_token,
            )
        finally:
            self.train(was_training)

    def _generate_impl(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        lengths: torch.Tensor | None = None,
        max_frames: int = MAX_MEL_FRAMES,
        use_stop_token: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a Mel spectrogram without ground-truth Mel frames.

        Parameters
        ----------
        lengths:
            Number of frames to generate per sample.  When omitted the
            duration predictor decides.
        use_stop_token:
            When True, generation stops as soon as the stop probability goes
            above 0.5 (fully autoregressive mode).  When False, the decoder
            runs for the requested number of frames, which is more stable for
            an under-trained model.

        Returns ``(mel, lengths)`` padded to the longest sample.
        """

        memory, mask = self.encode(text, text_lengths)

        if lengths is None:
            lengths = self.predict_mel_lengths(
                text,
                text_lengths,
                memory=memory,
            )

        lengths = lengths.clamp(
            min=MIN_MEL_FRAMES,
            max=max_frames,
        ).to(memory.device)

        batch_size = memory.shape[0]
        max_length = int(lengths.max().item())

        hidden = memory.new_zeros(batch_size, self.hidden_dim)
        attention = memory.new_zeros(batch_size, memory.shape[1])
        previous_mel = memory.new_zeros(batch_size, self.n_mels)

        outputs: list[torch.Tensor] = []

        for _ in range(max_length):
            prenet_output = self.prenet(previous_mel)

            context, attention = self.attention(
                hidden,
                memory,
                mask,
                attention,
            )

            rnn_input = torch.cat(
                [prenet_output, context],
                dim=-1,
            ).unsqueeze(1)

            output, new_hidden = self.decoder_rnn(
                rnn_input,
                hidden.unsqueeze(0),
            )

            hidden = new_hidden.squeeze(0)

            decoder_state = torch.cat(
                [output.squeeze(1), context],
                dim=-1,
            )
            mel = self.mel_projection(decoder_state)

            outputs.append(mel)
            previous_mel = mel

            if use_stop_token:
                stop_probability = torch.sigmoid(
                    self.stop_projection(decoder_state)
                )
                if bool((stop_probability > 0.5).all()):
                    break

        mel_before = torch.stack(outputs, dim=2)

        # Zero out frames beyond each sample's requested length.
        frame_positions = torch.arange(
            mel_before.shape[2],
            device=mel_before.device,
        )
        frame_mask = (
            frame_positions.unsqueeze(0) < lengths.unsqueeze(1)
        ).unsqueeze(1)

        mel_before = mel_before * frame_mask
        mel_after = mel_before + self.postnet(mel_before)
        # Mask again: the postnet has biases, so it would otherwise leave
        # non-zero values in the padding region.
        mel_after = mel_after * frame_mask

        return mel_after, lengths

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Public inference interface, compatible with the baseline model.

        With ``mel_lengths`` the decoder produces exactly that many frames
        per sample; without it the duration predictor chooses the length.
        """

        if mel_lengths is not None:
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

        mel, _ = self.generate(
            text,
            text_lengths,
            lengths=mel_lengths,
        )

        return mel
