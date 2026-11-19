
"""Audio processing utilities for VoiceForge.

The functions here convert between waveforms, log-Mel spectrograms and
WAV files.  They are used by the dataset loader during training and by the
TTS pipeline during inference, so the same preprocessing is applied in both
directions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

SAMPLE_RATE = 22050
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
LOG_EPSILON = 1e-5


@lru_cache(maxsize=8)
def _mel_transform(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> torchaudio.transforms.MelSpectrogram:
    """
    Return a cached MelSpectrogram transform.

    Creating the transform is cheap but not free; the dataset loader calls
    this function for every sample, so the result is cached.
    """

    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )


@lru_cache(maxsize=8)
def _inverse_mel_transform(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> torchaudio.transforms.InverseMelScale:
    return torchaudio.transforms.InverseMelScale(
        n_stft=n_fft // 2 + 1,
        n_mels=n_mels,
        sample_rate=sample_rate,
    )


@lru_cache(maxsize=8)
def _griffin_lim_transform(
    n_fft: int,
    hop_length: int,
    n_iter: int,
) -> torchaudio.transforms.GriffinLim:
    return torchaudio.transforms.GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        power=2.0,
        n_iter=n_iter,
        rand_init=True,
    )


def load_audio(
    file_path: str | Path,
    target_sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """
    Load an audio file as a mono float32 waveform.

    Returns a tensor with shape ``(1, samples)``.
    """

    audio, original_sample_rate = sf.read(
        str(file_path),
        dtype="float32",
        always_2d=True,
    )

    waveform = torch.from_numpy(audio.T.copy())

    # Convert stereo audio to mono.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Convert audio to the target sample rate.
    if original_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            original_sample_rate,
            target_sample_rate,
        )

    return waveform


def mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """
    Convert a waveform into a log-Mel spectrogram.

    The output has shape ``(channels, n_mels, frames)`` and contains
    ``log(power + 1e-5)`` values.
    """

    transform = _mel_transform(
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
    )

    mel = transform(waveform)

    return torch.log(mel + LOG_EPSILON)


def mel_to_audio(
    log_mel: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_iter: int = 32,
) -> torch.Tensor:
    """
    Convert a log-Mel spectrogram into a waveform using Griffin-Lim.

    Griffin-Lim is an *iterative phase reconstruction* algorithm.  It tries
    to find a waveform whose spectrogram matches the requested magnitude
    spectrogram.  It is a classical signal-processing baseline, not a neural
    vocoder, and the result is noticeably buzzy compared to a trained
    vocoder such as WaveNet or HiFi-GAN.
    """

    if log_mel.ndim == 3:
        if log_mel.shape[0] != 1:
            raise ValueError(
                "Only one Mel spectrogram can be converted at a time"
            )

        log_mel = log_mel.squeeze(0)

    if log_mel.ndim != 2:
        raise ValueError(
            "Log-Mel spectrogram must have two dimensions"
        )

    if log_mel.shape[0] != N_MELS:
        raise ValueError(
            "Expected a Mel spectrogram with 80 Mel bands"
        )

    if log_mel.shape[1] == 0:
        raise ValueError(
            "Mel spectrogram cannot be empty"
        )

    # Griffin-Lim needs at least n_fft / hop_length frames for the overlap
    # and padding arithmetic in torch.istft to be valid.
    minimum_frames = n_fft // hop_length

    if log_mel.shape[1] < minimum_frames:
        raise ValueError(
            "Mel spectrogram is too short to reconstruct: "
            f"{log_mel.shape[1]} frames, at least {minimum_frames} "
            "are required"
        )

    if not torch.isfinite(log_mel).all():
        raise ValueError(
            "Mel spectrogram contains non-finite values"
        )

    if sample_rate <= 0 or n_iter < 1:
        raise ValueError(
            "Sample rate and iteration count must be positive"
        )

    log_mel = log_mel.float()

    # Reverse the logarithm used during preprocessing.
    mel_power = (log_mel.exp() - LOG_EPSILON).clamp_min(0.0)

    spectrogram = _inverse_mel_transform(
        sample_rate,
        n_fft,
        N_MELS,
    )(mel_power)

    # GriffinLim(power=2.0) takes the square root of the power spectrogram
    # internally, so passing a power spectrogram here is correct.
    waveform = _griffin_lim_transform(
        n_fft,
        hop_length,
        n_iter,
    )(spectrogram)

    return waveform


def normalize_waveform(
    waveform: torch.Tensor,
    target_peak: float = 0.95,
) -> torch.Tensor:
    """
    Scale a waveform so that its loudest sample reaches ``target_peak``.

    Griffin-Lim output is often very quiet (peak amplitude around 0.05), so
    normalising makes the generated WAV files comfortably audible.  Silence
    is returned unchanged.
    """

    if not 0.0 < target_peak <= 1.0:
        raise ValueError("Target peak must be between 0 and 1")

    peak = waveform.abs().max()

    if not torch.isfinite(peak) or peak <= 1e-8:
        return waveform

    return waveform * (target_peak / peak)


def save_audio(
    waveform: torch.Tensor,
    file_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    normalize: bool = True,
    target_peak: float = 0.95,
) -> Path:
    """
    Save a waveform as a 16-bit PCM WAV file and return its path.
    """

    if sample_rate <= 0:
        raise ValueError(
            "Sample rate must be greater than zero"
        )

    if waveform.ndim == 2 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)

    if waveform.ndim != 1:
        raise ValueError(
            "Waveform must be one-dimensional or have shape (1, samples)"
        )

    if waveform.numel() == 0:
        raise ValueError(
            "Waveform cannot be empty"
        )

    if not torch.isfinite(waveform).all():
        raise ValueError(
            "Waveform contains non-finite values"
        )

    if normalize:
        waveform = normalize_waveform(waveform, target_peak=target_peak)

    output_path = Path(file_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_array = waveform.detach().cpu().numpy()

    sf.write(
        str(output_path),
        audio_array,
        sample_rate,
        subtype="PCM_16",
    )

    return output_path
