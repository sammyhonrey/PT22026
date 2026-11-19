"""
Application logic shared by every VoiceForge user interface.

Keeping the validation, model loading and file handling here means the web
interface, the optional Gradio interface and the tests all exercise exactly
the same code path - there is no second, "demo-only" implementation that
could behave differently from the real pipeline.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from voiceforge import DEFAULT_CHECKPOINT
from voiceforge.kokoro_backend import (
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    KOKORO_LANGUAGE,
    KOKORO_VOICES,
    MAX_SPEED,
    MIN_SPEED,
    KokoroBackend,
    KokoroError,
    kokoro_is_installed,
    validate_speed,
)
from voiceforge.text import normalize_text, text_to_sequence
from voiceforge.tts import VoiceForgeTTS

#: Longest text the interface accepts.
MAX_TEXT_LENGTH = 400

#: Default folder for generated audio (git-ignored).
DEFAULT_OUTPUT_DIR = "outputs/app"

#: Speech engines.
ENGINE_KOKORO = "kokoro"
ENGINE_VOICEFORGE = "voiceforge"
ENGINE_AUTO = "auto"

SUPPORTED_ENGINES = (ENGINE_KOKORO, ENGINE_VOICEFORGE)

ENGINE_LABELS = {
    ENGINE_KOKORO: "Kokoro-82M — natural speech",
    ENGINE_VOICEFORGE: "VoiceForge custom — experimental",
}


class AppError(Exception):
    """
    An error whose message is safe and useful to show to the user.

    ``code`` is a stable machine-readable identifier for API clients and
    ``status`` the HTTP status the server should use.
    """

    def __init__(
        self,
        message: str,
        code: str = "bad_request",
        status: int = 400,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass(frozen=True)
class SynthesisResult:
    """Everything the interface needs to display one generation."""

    text: str
    path: str
    filename: str
    duration_seconds: float
    truncated: bool
    padded: bool
    sample_rate: int = 22050
    engine: str = "voiceforge"
    voice: str | None = None
    speed: float | None = None


class VoiceForgeApp:
    """
    Facade over the available speech engines.

    Two engines are supported:

    * ``kokoro``     — the pretrained Kokoro-82M model (default when it is
      installed), which produces natural speech;
    * ``voiceforge`` — the student's own acoustic model, kept as an
      experimental backend.

    Models are loaded lazily so the interface starts quickly, and a lock
    keeps CUDA work thread-safe for the web server.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: str | None = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        griffin_lim_iterations: int = 32,
        max_mel_frames: int = 2200,
        engine: str = ENGINE_AUTO,
        kokoro_voice: str = DEFAULT_VOICE,
        kokoro_device: str | None = None,
    ):
        if engine not in (ENGINE_AUTO, *SUPPORTED_ENGINES):
            raise ValueError(
                f"Unknown engine {engine!r}. "
                f"Choose from: {', '.join((ENGINE_AUTO, *SUPPORTED_ENGINES))}"
            )

        if kokoro_voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown Kokoro voice {kokoro_voice!r}. "
                f"Available: {', '.join(KOKORO_VOICES)}"
            )

        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.output_dir = Path(output_dir)
        self.griffin_lim_iterations = griffin_lim_iterations
        self.max_mel_frames = max_mel_frames

        self.requested_engine = engine
        self.kokoro_voice = kokoro_voice
        self.kokoro_device = kokoro_device

        self._pipeline: VoiceForgeTTS | None = None
        self._kokoro: KokoroBackend | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> VoiceForgeTTS:
        """Load the checkpoint once and reuse it."""

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            if not self.checkpoint_path.is_file():
                raise AppError(
                    f"Model checkpoint not found: {self.checkpoint_path}. "
                    "Train one with 'python -m scripts.train' or point "
                    "--checkpoint at an existing file.",
                    code="checkpoint_missing",
                )

            try:
                self._pipeline = VoiceForgeTTS.from_pretrained(
                    self.checkpoint_path,
                    device=self.device,
                    griffin_lim_iterations=self.griffin_lim_iterations,
                    max_mel_frames=self.max_mel_frames,
                )
            except (ValueError, RuntimeError) as error:
                raise AppError(
                    f"The checkpoint could not be loaded: {error}"
                ) from error

            return self._pipeline

    def device_description(self) -> str:
        """Human-readable device name, loading the model if necessary."""

        pipeline = self.load()
        return str(pipeline.device)

    def load_kokoro(self) -> KokoroBackend:
        """Load the pretrained Kokoro engine once and reuse it."""

        with self._lock:
            if self._kokoro is not None:
                return self._kokoro

            backend = KokoroBackend(
                voice=self.kokoro_voice,
                device=self.kokoro_device,
            )

            try:
                backend.load()
            except KokoroError as error:
                raise AppError(str(error)) from error
            except ValueError as error:
                raise AppError(str(error)) from error

            self._kokoro = backend
            return backend

    def resolve_engine(self, engine: str | None = None) -> str:
        """
        Decide which engine to use.

        ``auto`` prefers Kokoro when its package is installed and falls back
        to the custom model otherwise, so the interface always has a working
        engine.
        """

        chosen = engine or self.requested_engine

        if chosen == ENGINE_AUTO:
            return ENGINE_KOKORO if kokoro_is_installed() else ENGINE_VOICEFORGE

        if chosen not in SUPPORTED_ENGINES:
            raise AppError(
                f"Unknown speech engine {chosen!r}. "
                f"Available: {', '.join(SUPPORTED_ENGINES)}",
                code="invalid_engine",
            )

        return chosen

    def describe_engines(self) -> dict:
        """Summary of the engines for the health endpoint and the banner."""

        return {
            "default": self.resolve_engine(),
            "requested": self.requested_engine,
            "labels": dict(ENGINE_LABELS),
            "kokoro": {
                "installed": kokoro_is_installed(),
                "loaded": self._kokoro is not None,
                "voices": list(KOKORO_VOICES),
                "voice_labels": dict(KOKORO_VOICES),
                "sample_rate": KokoroBackend.SAMPLE_RATE,
            },
            "voiceforge": {
                "checkpoint": str(self.checkpoint_path),
                "loaded": self._pipeline is not None,
                "sample_rate": 22050,
            },
        }

    def describe_engine_details(self) -> dict:
        """
        Structured engine capabilities for ``GET /api/engines``.

        The values come from the same constants the engines use, so the
        frontend never has to hardcode a voice list or a speed range.
        """

        return {
            "default": self.resolve_engine(),
            "requested": self.requested_engine,
            "limits": {"max_text_length": MAX_TEXT_LENGTH},
            "engines": {
                ENGINE_KOKORO: {
                    "id": ENGINE_KOKORO,
                    "name": ENGINE_LABELS[ENGINE_KOKORO],
                    "installed": kokoro_is_installed(),
                    "loaded": self._kokoro is not None,
                    "language": KOKORO_LANGUAGE,
                    "sample_rate": KokoroBackend.SAMPLE_RATE,
                    "voices": [
                        {
                            "id": voice_id,
                            "name": label,
                            "default": voice_id == DEFAULT_VOICE,
                        }
                        for voice_id, label in KOKORO_VOICES.items()
                    ],
                    "speed": {
                        "supported": True,
                        "min": MIN_SPEED,
                        "max": MAX_SPEED,
                        "default": DEFAULT_SPEED,
                    },
                    "max_text_length": MAX_TEXT_LENGTH,
                    "capabilities": [
                        "text-to-speech",
                        "voice-selection",
                        "speed-control",
                    ],
                },
                ENGINE_VOICEFORGE: {
                    "id": ENGINE_VOICEFORGE,
                    "name": ENGINE_LABELS[ENGINE_VOICEFORGE],
                    "installed": self.checkpoint_path.is_file(),
                    "loaded": self._pipeline is not None,
                    "language": "en-US",
                    "sample_rate": 22050,
                    "voices": [
                        {
                            "id": "default",
                            "name": "LJSpeech single speaker",
                            "default": True,
                        }
                    ],
                    "speed": {
                        "supported": False,
                        "min": None,
                        "max": None,
                        "default": None,
                    },
                    "max_text_length": MAX_TEXT_LENGTH,
                    "capabilities": ["text-to-speech"],
                },
            },
        }

    def set_engine(self, engine: str) -> None:
        """Change the default engine (used for startup fallbacks)."""

        if engine not in (ENGINE_AUTO, *SUPPORTED_ENGINES):
            raise AppError(
                f"Unknown speech engine {engine!r}. "
                f"Available: {', '.join(SUPPORTED_ENGINES)}",
                code="invalid_engine",
            )

        self.requested_engine = engine

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate_text(
        text: object,
        engine: str = ENGINE_VOICEFORGE,
    ) -> str:
        """
        Check user input and return the cleaned text.

        The custom model has a fixed 38-character vocabulary, so text for it
        must be encodable. Kokoro handles digits and richer punctuation, so
        only the common checks apply there.
        """

        if not isinstance(text, str):
            raise AppError("Please enter text as a string.", code="invalid_text")

        cleaned = text.strip()

        if not cleaned:
            raise AppError("Please enter some English text first.", code="empty_text")

        if len(cleaned) > MAX_TEXT_LENGTH:
            raise AppError(
                f"Text is too long ({len(cleaned)} characters). "
                f"Please keep it under {MAX_TEXT_LENGTH} characters.",
                code="text_too_long",
            )

        if engine == ENGINE_VOICEFORGE:
            if not normalize_text(cleaned):
                raise AppError(
                    "The text contains no usable characters. "
                    "Please enter English words.",
                    code="empty_text",
                )

            try:
                text_to_sequence(cleaned)
            except ValueError as error:
                raise AppError(
                    f"{error}. The custom VoiceForge model supports English "
                    "letters and the punctuation . , ! ? - ' \" ; : ( ). "
                    "Switch to the Kokoro engine for digits.",
                    code="unsupported_characters",
                ) from error
        elif not any(character.isalnum() for character in cleaned):
            raise AppError("Please enter some English words.", code="empty_text")

        return cleaned

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------
    def synthesize(
        self,
        text: object,
        engine: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
    ) -> SynthesisResult:
        """
        Validate the request, generate speech with the chosen engine and
        return the WAV details.

        ``engine`` may be ``"kokoro"``, ``"voiceforge"`` or ``None`` for the
        configured default. ``voice`` and ``speed`` only apply to Kokoro;
        sending them to the custom engine is an error rather than being
        silently ignored.
        """

        chosen = self.resolve_engine(engine)
        cleaned = self.validate_text(text, chosen)

        effective_voice: str | None = None
        effective_speed: float | None = None

        if chosen == ENGINE_KOKORO:
            effective_voice = voice or DEFAULT_VOICE

            if effective_voice not in KOKORO_VOICES:
                raise AppError(
                    f"Unknown Kokoro voice {effective_voice!r}. "
                    f"Available: {', '.join(KOKORO_VOICES)}",
                    code="invalid_voice",
                )

            try:
                effective_speed = validate_speed(
                    DEFAULT_SPEED if speed is None else speed
                )
            except ValueError as error:
                raise AppError(str(error), code="invalid_speed") from error
        elif speed is not None:
            raise AppError(
                "Speech speed is only supported by the Kokoro engine. "
                "The VoiceForge custom model always uses its native rate.",
                code="speed_not_supported",
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4().hex[:12]}.wav"
        output_path = self.output_dir / filename

        truncated = False
        padded = False

        if chosen == ENGINE_KOKORO:
            backend = self.load_kokoro()

            with self._lock:
                try:
                    backend.save(
                        cleaned,
                        output_path,
                        voice=effective_voice,
                        speed=effective_speed,
                    )
                except (KokoroError, ValueError) as error:
                    raise AppError(
                        str(error),
                        code="kokoro_error",
                    ) from error
        else:
            pipeline = self.load()

            # The pipeline holds CUDA state; one generation at a time.
            with self._lock:
                pipeline.save(cleaned, output_path)
                truncated = bool(pipeline.was_truncated)
                padded = bool(pipeline.was_padded)

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise AppError(
                "Speech generation failed: the audio file was not written.",
                code="generation_failed",
                status=500,
            )

        info = sf.info(str(output_path))

        if info.frames == 0:
            raise AppError(
                "Speech generation produced an empty audio file.",
                code="generation_failed",
                status=500,
            )

        return SynthesisResult(
            text=cleaned,
            path=str(output_path),
            filename=filename,
            duration_seconds=round(info.frames / info.samplerate, 2),
            truncated=truncated,
            padded=padded,
            sample_rate=info.samplerate,
            engine=chosen,
            voice=effective_voice,
            speed=effective_speed,
        )

    def resolve_audio(self, filename: str) -> Path:
        """
        Return the path of a generated file, refusing anything outside the
        output folder.
        """

        if not filename or "/" in filename or "\\" in filename:
            raise AppError("Invalid audio file name.", code="invalid_filename")

        if not filename.endswith(".wav"):
            raise AppError("Invalid audio file name.", code="invalid_filename")

        candidate = (self.output_dir / filename).resolve()
        output_root = self.output_dir.resolve()

        if not candidate.is_file() or output_root not in candidate.parents:
            raise AppError("Audio file not found.", code="audio_not_found", status=404)

        return candidate
