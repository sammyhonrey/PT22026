"""
VoiceForge - a custom English text-to-speech library.

VoiceForge converts English text into speech with its **own** acoustic model
and Griffin-Lim phase reconstruction.  It does not wrap a pretrained TTS
system or call an external service.

Quick start
-----------
>>> from voiceforge import VoiceForgeTTS
>>> tts = VoiceForgeTTS.from_pretrained("checkpoints/voiceforge_attn.pt")
>>> tts.save("Hello, welcome to VoiceForge!", "output.wav")

One-shot helper for scripts::

>>> from voiceforge import generate_speech
>>> generate_speech("Hello, welcome to VoiceForge!", "output.wav")

Lower-level building blocks are exported as well::

>>> from voiceforge import load_model, text_to_sequence, mel_to_audio

The model is small and experimental: the output is recognisably synthesised
speech but is not expected to be fully intelligible.  See
``docs/evaluation.md`` for the measured limitations.
"""

from __future__ import annotations

from pathlib import Path

from voiceforge.audio import SAMPLE_RATE
from voiceforge.checkpoint import load_model
from voiceforge.model import AcousticModel, ModelConfig
from voiceforge.text import normalize_text, text_to_sequence
from voiceforge.tts import VoiceForgeTTS

__version__ = "0.2.0"

#: Checkpoint used when no path is given explicitly.
DEFAULT_CHECKPOINT = "checkpoints/voiceforge_attn.pt"


def generate_speech(
    text: str,
    output_path: str | Path,
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str | None = None,
    griffin_lim_iterations: int = 32,
) -> Path:
    """
    Generate one sentence and write it to ``output_path``.

    This is a convenience wrapper that loads the checkpoint on every call.
    When generating several sentences, create one :class:`VoiceForgeTTS`
    with :meth:`VoiceForgeTTS.from_pretrained` and reuse it, because loading
    the model is much slower than generating speech.

    Parameters
    ----------
    text:
        English text to speak.
    output_path:
        Destination WAV file (parent folders are created).
    checkpoint:
        Path to a trained checkpoint.
    device:
        ``"cpu"``, ``"cuda"`` or ``None`` for automatic selection.
    griffin_lim_iterations:
        Reconstruction iterations; more is slower but cleaner.

    Returns
    -------
    pathlib.Path
        The written file.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    TypeError
        If ``text`` is not a string.
    ValueError
        If ``text`` is empty or contains unsupported characters.
    """

    pipeline = VoiceForgeTTS.from_pretrained(
        checkpoint,
        device=device,
        griffin_lim_iterations=griffin_lim_iterations,
    )

    pipeline.save(text, output_path)

    return Path(output_path)


__all__ = [
    "VoiceForgeTTS",
    "generate_speech",
    "load_model",
    "AcousticModel",
    "ModelConfig",
    "SAMPLE_RATE",
    "DEFAULT_CHECKPOINT",
    "normalize_text",
    "text_to_sequence",
    "__version__",
]
