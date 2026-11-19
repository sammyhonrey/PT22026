"""
Kokoro-82M pretrained text-to-speech backend.

Kokoro is a small, high-quality open-weight TTS model released by hexgrad.
It is **not** trained by this project: VoiceForge uses it as a pretrained
engine and contributes the integration, the interface and the evaluation.
The student's own acoustic model remains available as the
``voiceforge`` backend (see ``voiceforge/tts.py``).

Resources
---------
* Model: https://huggingface.co/hexgrad/Kokoro-82M
* Code:  https://github.com/hexgrad/kokoro
* License: Apache-2.0 (see the model card)

The weights are downloaded from Hugging Face on first use and cached by
``huggingface_hub``. Set ``HF_HOME`` to move the cache, or ``HF_ENDPOINT``
to a mirror if huggingface.co is unreachable.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"

#: Speed range supported by the Kokoro pipeline.
MIN_SPEED = 0.5
MAX_SPEED = 2.0
DEFAULT_SPEED = 1.0

#: Language exposed by this integration (the pipeline is created with "a").
KOKORO_LANGUAGE = "en-US"

#: Curated American-English voices exposed by the interface.
KOKORO_VOICES: dict[str, str] = {
    "af_heart": "US female — warm and clear (default)",
    "af_bella": "US female — expressive",
    "af_nicole": "US female — soft",
    "af_sarah": "US female — neutral",
    "am_michael": "US male — calm",
    "am_adam": "US male — deeper",
    "am_onyx": "US male — low",
}


class KokoroError(RuntimeError):
    """Raised when the Kokoro backend cannot be used."""


def validate_speed(value: object) -> float:
    """
    Validate a Kokoro speed value and return it as a float.

    Kokoro accepts ``speed`` per call (``KPipeline.__call__``), so a new
    value does not require rebuilding the pipeline.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Speed must be a number between {MIN_SPEED} and {MAX_SPEED}."
        )

    speed = float(value)

    if not math.isfinite(speed) or not MIN_SPEED <= speed <= MAX_SPEED:
        raise ValueError(
            f"Speed must be between {MIN_SPEED} and {MAX_SPEED} "
            f"(got {value})."
        )

    return speed


def kokoro_is_installed() -> bool:
    """True when the ``kokoro`` package is importable."""

    return importlib.util.find_spec("kokoro") is not None


def _default_cache_dir() -> Path:
    """Hugging Face's cache location when ``HF_HOME`` is unset."""

    configured = os.environ.get("HF_HOME")

    if configured:
        return Path(configured)

    return Path.home() / ".cache" / "huggingface"


def _is_writable(path: Path) -> bool:
    """True when ``path`` can be created and written to."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".voiceforge_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def configure_cache(cache_dir: str | Path = "outputs/hf") -> str | None:
    """
    Make sure Hugging Face has a writable cache directory.

    Some Windows accounts (and sandboxed environments) cannot write to
    ``~/.cache/huggingface``, which makes the weight download fail with
    ``WinError 5``.  When the default location is not writable and the user
    has not set ``HF_HOME`` explicitly, this points Hugging Face at a
    project-local folder instead.

    Returns the cache directory that was selected, or ``None`` when the
    default was already writable.  Must be called before importing
    ``kokoro``/``huggingface_hub``, because they read the environment once.
    """

    if os.environ.get("HF_HOME"):
        # The user made an explicit choice - respect it.
        return os.environ["HF_HOME"]

    default = _default_cache_dir()

    if _is_writable(default):
        return None

    target = Path(cache_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(target)
    os.environ.setdefault("HF_HUB_CACHE", str(target / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    return str(target)


def use_cached_weights_offline(repo_id: str = KOKORO_REPO_ID) -> bool:
    """
    Use the cached weights without contacting Hugging Face.

    When the model is already in the local cache, `huggingface_hub` still
    re-validates file metadata over the network on some calls. On a machine
    where huggingface.co is unreachable that stalls model loading for
    minutes, so offline mode is enabled whenever the snapshot is present.

    Returns True when offline mode was enabled. Must run before importing
    ``kokoro``/``huggingface_hub``.
    """

    hub_cache = os.environ.get("HF_HUB_CACHE")

    if not hub_cache:
        hf_home = os.environ.get("HF_HOME")
        hub_cache = str((Path(hf_home) if hf_home else _default_cache_dir()) / "hub")

    folder = Path(hub_cache) / ("models--" + repo_id.replace("/", "--"))

    if not folder.is_dir():
        return False

    snapshots = folder / "snapshots"

    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        return False

    os.environ["HF_HUB_OFFLINE"] = "1"
    return True


class KokoroBackend:
    """
    Thin wrapper around :class:`kokoro.KPipeline`.

    The model is loaded lazily so starting the application stays fast, and
    the sample rate is fixed at 24 kHz, which is what Kokoro generates.
    """

    SAMPLE_RATE = KOKORO_SAMPLE_RATE

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        device: str | None = None,
        repo_id: str = KOKORO_REPO_ID,
        speed: float = 1.0,
    ):
        if voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown Kokoro voice {voice!r}. "
                f"Available: {', '.join(KOKORO_VOICES)}"
            )

        if not 0.5 <= speed <= 2.0:
            raise ValueError("Speed must be between 0.5 and 2.0")

        self.voice = voice
        self.repo_id = repo_id
        self.speed = speed
        self.requested_device = device

        self._pipeline = None
        self._cache_dir: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def device(self) -> str:
        if self._pipeline is not None:
            value = getattr(self._pipeline, "device", None)
            if value:
                return str(value)

        return self.requested_device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def load(self):
        """Create the Kokoro pipeline (downloads weights on first use)."""

        if self._pipeline is not None:
            return self._pipeline

        if not kokoro_is_installed():
            raise KokoroError(
                "The 'kokoro' package is not installed. Install it with "
                "'pip install kokoro' (and 'pip install misaki[en]' if "
                "prompted), then restart the application."
            )

        # Must run before kokoro imports huggingface_hub.
        self._cache_dir = configure_cache()
        self._offline = use_cached_weights_offline(self.repo_id)

        try:
            from kokoro import KPipeline

            device = self.requested_device or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            self._pipeline = KPipeline(
                lang_code="a",
                repo_id=self.repo_id,
                device=device,
            )
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            hint = (
                f" A writable cache was selected at {self._cache_dir}."
                if self._cache_dir
                else ""
            )

            raise KokoroError(
                "Kokoro could not be loaded. The pretrained weights are "
                f"downloaded from Hugging Face ({self.repo_id}); check your "
                "internet connection, or set HF_ENDPOINT to a mirror."
                f"{hint} Original error: {error}"
            ) from error

        return self._pipeline

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> np.ndarray:
        """
        Generate speech and return a mono float32 waveform at 24 kHz.

        Kokoro works sentence by sentence, so the chunks are concatenated.
        ``speed`` is applied per call, which the pipeline supports directly;
        the loaded model is reused.
        """

        pipeline = self.load()

        selected_voice = voice or self.voice
        selected_speed = validate_speed(
            self.speed if speed is None else speed
        )

        if selected_voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown Kokoro voice {selected_voice!r}. "
                f"Available: {', '.join(KOKORO_VOICES)}"
            )

        chunks: list[np.ndarray] = []

        try:
            for _, _, audio in pipeline(
                text,
                voice=selected_voice,
                speed=selected_speed,
            ):
                chunks.append(
                    audio.detach().cpu().numpy()
                    if torch.is_tensor(audio)
                    else np.asarray(audio)
                )
        except Exception as error:  # noqa: BLE001 - surfaced to the user
            raise KokoroError(
                f"Kokoro could not generate speech: {error}"
            ) from error

        if not chunks:
            raise KokoroError("Kokoro returned no audio for this text.")

        waveform = np.concatenate(chunks).astype(np.float32)

        if waveform.size == 0:
            raise KokoroError("Kokoro returned an empty waveform.")

        if not np.isfinite(waveform).all():
            raise KokoroError("Kokoro produced non-finite audio samples.")

        return waveform

    def save(
        self,
        text: str,
        output_path: str | Path,
        voice: str | None = None,
        speed: float | None = None,
    ) -> Path:
        """Synthesise ``text`` and write a 24 kHz 16-bit PCM WAV file."""

        waveform = self.synthesize(text, voice=voice, speed=speed)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sf.write(
            str(output_path),
            waveform,
            self.SAMPLE_RATE,
            subtype="PCM_16",
        )

        return output_path


def describe_environment() -> dict:
    """Information for the health endpoint and the startup banner."""

    return {
        "installed": kokoro_is_installed(),
        "sample_rate": KOKORO_SAMPLE_RATE,
        "repo_id": KOKORO_REPO_ID,
        "voices": list(KOKORO_VOICES),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    }
