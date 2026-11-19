
from pathlib import Path
from typing import Any

import torch

from voiceforge.audio import mel_to_audio, save_audio
from voiceforge.text import text_to_sequence

# One second of audio at the default settings.  Used to cap runaway
# generations produced by an undertrained duration predictor.
FRAMES_PER_SECOND = 22050 / 256

DEFAULT_MAX_MEL_FRAMES = 2200  # about 25 seconds


class VoiceForgeTTS:
    """
    Main Text-to-Speech interface for VoiceForge.

    Converts English text into speech using:
        1. Text processing
        2. The trained acoustic model
        3. Griffin-Lim audio reconstruction

    Example
    -------
    >>> tts = VoiceForgeTTS.from_pretrained("checkpoints/voiceforge.pt")
    >>> tts.save("Hello, welcome to VoiceForge!", "output.wav")
    """

    def __init__(
        self,
        model,
        device=None,
        sample_rate=22050,
        griffin_lim_iterations=32,
        max_mel_frames=DEFAULT_MAX_MEL_FRAMES,
    ):
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.model.eval()

        self.sample_rate = sample_rate
        self.griffin_lim_iterations = griffin_lim_iterations

        if max_mel_frames < 1:
            raise ValueError(
                "Maximum number of Mel frames must be at least 1"
            )

        self.max_mel_frames = int(max_mel_frames)

        #: Set to True when a generation was cut short by the frame cap.
        self.was_truncated = False

        #: Set to True when a very short output had to be padded.
        self.was_padded = False

        # Griffin-Lim cannot reconstruct fewer than n_fft // hop_length
        # frames.  Undertrained models sometimes predict almost no frames,
        # so the pipeline pads those degenerate outputs.
        self.min_mel_frames = 4

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: str | None = None,
        **kwargs: Any,
    ) -> "VoiceForgeTTS":
        """
        Build a TTS pipeline from a trained checkpoint file.

        Parameters
        ----------
        checkpoint_path:
            Path to a ``.pt`` checkpoint created by the training script.
        device:
            ``"cpu"``, ``"cuda"`` or ``None`` to pick automatically.
        **kwargs:
            Extra options forwarded to :class:`VoiceForgeTTS`
            (for example ``griffin_lim_iterations``).
        """

        # Imported lazily so that importing voiceforge.tts stays cheap.
        from voiceforge.checkpoint import load_model

        resolved_device = device
        if resolved_device is None:
            resolved_device = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        model, config, metadata = load_model(
            checkpoint_path,
            map_location=resolved_device,
        )

        pipeline = cls(model, device=resolved_device, **kwargs)

        pipeline.checkpoint_path = str(checkpoint_path)
        pipeline.model_config = config
        pipeline.checkpoint_metadata = metadata

        return pipeline

    @torch.inference_mode()
    def synthesize(self, text: str) -> torch.Tensor:
        """
        Convert text into a waveform tensor.

        Parameters
        ----------
        text:
            English text to speak.

        Returns
        -------
        torch.Tensor
            A one-dimensional waveform at ``self.sample_rate``.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Convert English text into numerical character IDs.
        sequence = text_to_sequence(text)

        if not sequence:
            raise ValueError("Text produced an empty sequence")

        text_tensor = torch.tensor(
            [sequence],
            dtype=torch.long,
            device=self.device,
        )

        text_lengths = torch.tensor(
            [len(sequence)],
            dtype=torch.long,
            device=self.device,
        )

        # Generate a Mel spectrogram with the acoustic model.
        predicted_mel = self.model(text_tensor, text_lengths)

        if predicted_mel.ndim != 3:
            raise ValueError(
                "The acoustic model must return a 3D Mel spectrogram "
                "with shape [batch, mel_channels, time]"
            )

        if predicted_mel.shape[0] != 1:
            raise ValueError("Expected exactly one generated sample")

        self.was_truncated = False
        self.was_padded = False

        if predicted_mel.shape[2] > self.max_mel_frames:
            predicted_mel = predicted_mel[:, :, : self.max_mel_frames]
            self.was_truncated = True

        if predicted_mel.shape[2] < self.min_mel_frames:
            # Repeat the final frame until Griffin-Lim has enough input.
            padding = predicted_mel[:, :, -1:].repeat(
                1,
                1,
                self.min_mel_frames - predicted_mel.shape[2],
            )
            predicted_mel = torch.cat([predicted_mel, padding], dim=2)
            self.was_padded = True

        if not torch.isfinite(predicted_mel).all():
            raise ValueError(
                "The acoustic model produced non-finite Mel values"
            )

        # Convert the generated Mel spectrogram into audio.
        mel = predicted_mel[0].detach().cpu()

        waveform = mel_to_audio(
            mel,
            n_iter=self.griffin_lim_iterations,
        )

        return waveform.detach().cpu().squeeze()

    def save(self, text: str, output_path) -> torch.Tensor:
        """
        Generate speech and save it as a WAV file.

        Returns the generated waveform tensor.
        """

        waveform = self.synthesize(text)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_audio(
            waveform,
            str(output_path),
            sample_rate=self.sample_rate,
        )

        return waveform
