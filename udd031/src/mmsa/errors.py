"""Error taxonomy for the MMSA pipeline.

Every failure the application can present to a user is a subclass of
:class:`MMSAError` carrying a stable code (``MMSA-E1xx``), a user-facing message
written for a non-technical reader, and a recovery hint. Nothing else -- no raw
traceback, no ``KeyError`` text -- may reach the interface.

Warnings share the same machinery but subclass :class:`MMSAWarning`, so the UI can
render them amber and continue, while errors stop the request.
"""

from __future__ import annotations

from typing import Any, Final

# Base types


class MMSAError(Exception):
    """Base error: a user-facing code, a message and a recovery hint.

    Subclasses carry the codes in ``errors.py``; the interface shows the message and
    the hint, never the traceback.
    """

    severity: Final[str] = "error"

    def __init__(self, code: str, user_message: str, hint: str = "") -> None:
        super().__init__(f"{code}: {user_message}")
        self.code = code
        self.user_message = user_message
        self.hint = hint

    # -- presentation helpers ------------------------------------------------ #

    def to_markdown(self) -> str:
        """Render for the Gradio interface (headline + hint)."""
        text = f"**{self.user_message}**"
        if self.hint:
            text += f"\n\n<sub>{self.hint}</sub>"
        return text

    def to_dict(self) -> dict[str, Any]:
        """Serialise for logs and for ``results/`` artefacts."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.user_message,
            "hint": self.hint,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return f"{type(self).__name__}(code={self.code!r})"


class MMSAWarning(MMSAError):
    """A recoverable condition: the analysis proceeds, the user is told."""

    severity: Final[str] = "warning"


# Audio ingestion (E101-E106)


class UnsupportedFormatError(MMSAError):
    """E101 -- the file extension is not in the accepted set."""

    def __init__(self, extension: str) -> None:
        super().__init__(
            "MMSA-E101",
            f"That file type ({extension or 'unknown'}) isn't supported.",
            "Use WAV, FLAC, MP3, OGG or M4A.",
        )


class FileTooLargeError(MMSAError):
    """E102 -- the upload exceeds the size ceiling."""

    def __init__(self, size_mb: float, limit_mb: float) -> None:
        super().__init__(
            "MMSA-E102",
            f"That file is too large to analyse ({size_mb:.1f} MB).",
            f"Keep clips under {limit_mb:.0f} MB, or about 30 seconds of speech.",
        )


class AudioDecodeError(MMSAError):
    """E103 -- the container is readable but the audio payload is not."""

    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "MMSA-E103",
            "This audio file appears to be corrupted or truncated.",
            "Try re-exporting it as 16 kHz mono WAV." if not detail else f"{detail} Try a 16 kHz mono WAV.",
        )


class AudioTooShortError(MMSAError):
    """E104 -- less than the minimum analysable duration (after trimming)."""

    def __init__(self, duration_s: float, minimum_s: float) -> None:
        super().__init__(
            "MMSA-E104",
            f"That clip is too short to analyse reliably ({duration_s:.1f} s).",
            f"Record at least {minimum_s:.1f} s of continuous speech.",
        )


class AudioTruncatedWarning(MMSAWarning):
    """E105 -- longer than the ceiling; the head of the clip is used."""

    def __init__(self, duration_s: float, limit_s: float) -> None:
        super().__init__(
            "MMSA-E105",
            f"Clips are capped at {limit_s:.0f} s; only the first {limit_s:.0f} s were analysed "
            f"({duration_s:.1f} s supplied).",
            "",
        )


class SilentAudioError(MMSAError):
    """E106 -- the decoded buffer carries no usable signal."""

    def __init__(self) -> None:
        super().__init__(
            "MMSA-E106",
            "No sound was detected in this clip.",
            "Check your microphone input level and try again.",
        )


# Speech recognition and inference (E107-E112)


class NoSpeechError(MMSAError):
    """E107 -- the audio is audible but no words could be decoded."""

    def __init__(self) -> None:
        super().__init__(
            "MMSA-E107",
            "No speech could be transcribed from this clip.",
            "Speak clearly, closer to the microphone, or try a quieter room.",
        )


class LowTranscriptConfidenceWarning(MMSAWarning):
    """E108 -- the transcript is usable but uncertain; the text branch is damped."""

    def __init__(self, confidence: float) -> None:
        super().__init__(
            "MMSA-E108",
            f"Transcript confidence is low ({confidence:.2f}); the text branch was down-weighted.",
            "",
        )


class InferenceError(MMSAError):
    """E109 -- a model raised during forward inference."""

    def __init__(self, stage: str, detail: str = "") -> None:
        super().__init__(
            "MMSA-E109",
            f"The {stage} model failed on this clip.",
            "Retry, or try a different recording." if not detail else detail,
        )


class DeviceFallbackWarning(MMSAWarning):
    """E110 -- CUDA ran out of memory; inference was retried on the CPU."""

    def __init__(self) -> None:
        super().__init__(
            "MMSA-E110",
            "The GPU ran out of memory, so this clip was processed on the CPU instead.",
            "",
        )


class ModelUnavailableError(MMSAError):
    """E111 -- the Hub is unreachable and the local cache is empty."""

    def __init__(self, model_id: str, detail: str = "") -> None:
        super().__init__(
            "MMSA-E111",
            f"Models could not be loaded from Hugging Face ({model_id}).",
            "Check your internet connection, or pre-populate the cache with "
            "`python scripts/download_models.py`."
            if not detail
            else detail,
        )


class LabelMapMismatchError(MMSAError):
    """E112 -- the model's vocabulary changed upstream."""

    def __init__(self, model_id: str, expected: tuple[str, ...], found: tuple[str, ...]) -> None:
        super().__init__(
            "MMSA-E112",
            f"The label vocabulary of {model_id} does not match what this build expects.",
            f"Expected {list(expected)} but found {list(found)}. "
            "Run `python scripts/download_models.py --verify`.",
        )


# Registry (used by tests and by the report's taxonomy table)

ERROR_REGISTRY: Final[dict[str, type[MMSAError]]] = {
    "MMSA-E101": UnsupportedFormatError,
    "MMSA-E102": FileTooLargeError,
    "MMSA-E103": AudioDecodeError,
    "MMSA-E104": AudioTooShortError,
    "MMSA-E105": AudioTruncatedWarning,
    "MMSA-E106": SilentAudioError,
    "MMSA-E107": NoSpeechError,
    "MMSA-E108": LowTranscriptConfidenceWarning,
    "MMSA-E109": InferenceError,
    "MMSA-E110": DeviceFallbackWarning,
    "MMSA-E111": ModelUnavailableError,
    "MMSA-E112": LabelMapMismatchError,
}
"""All twelve codes, in numeric order. There is a unit test asserting length 12."""
