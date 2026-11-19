"""Speech recognition with two interchangeable backends.

``transformers`` (default) runs Whisper through ``WhisperProcessor`` and needs no
system ffmpeg. ``openai_whisper`` is the reference package, imported lazily. The
reported confidence is the geometric mean of token probabilities: a proxy for
decoder certainty, not a calibrated probability.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .config import ASRConfig, AppConfig
from .errors import InferenceError, ModelUnavailableError, NoSpeechError
from .logging_utils import get_logger
from .schemas import ASRResult

__all__ = ["ASRBackend", "TransformersWhisperBackend", "OpenAIWhisperBackend", "build_asr"]

logger = get_logger(__name__)

_FALLBACK_CONFIDENCE = 0.5


@runtime_checkable
class ASRBackend(Protocol):
    """Minimal contract the pipeline needs from a recogniser."""

    name: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        """Return the transcript and the decoder's confidence in it."""


# Default backend


class TransformersWhisperBackend:
    """Whisper via ``transformers`` (no ffmpeg requirement)."""

    name = "transformers-whisper"

    def __init__(self, model_id: str, device: str = "cpu", cache_dir: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self.cache_dir = cache_dir
        self._processor = None
        self._model = None

    # -- lazy loading -------------------------------------------------------- #

    def _load(self) -> tuple[object, object]:
        if self._model is not None and self._processor is not None:
            return self._processor, self._model

        try:
            import torch
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
        except ImportError as exc:  # pragma: no cover - lean installs only
            raise ModelUnavailableError(
                self.model_id, "PyTorch and Transformers are required for speech recognition."
            ) from exc

        try:
            self._processor = WhisperProcessor.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model.eval()
            model.to(self.device)
            self._model = model
        except Exception as exc:  # noqa: BLE001 - network, auth, disk all differ
            raise ModelUnavailableError(self.model_id, f"{type(exc).__name__}: {exc}") from exc

        logger.info("loaded ASR model %s on %s", self.model_id, self.device)
        return self._processor, self._model

    # -- inference ----------------------------------------------------------- #

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        """Decode a mono waveform to text, with a token-probability confidence."""
        import torch

        processor, model = self._load()
        try:
            inputs = processor(
                np.asarray(audio, dtype=np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            features = inputs.input_features.to(self.device)
            with torch.inference_mode():
                generated = model.generate(
                    features,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=200,
                )
            transcript = processor.batch_decode(generated.sequences, skip_special_tokens=True)[0]
            confidence = self._token_confidence(generated, model)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError("speech recognition", f"{type(exc).__name__}: {exc}") from exc

        transcript = " ".join(str(transcript).split())
        if not transcript.strip(" .,!?-"):
            raise NoSpeechError()

        return ASRResult(
            text=transcript,
            confidence=round(confidence, 4),
            n_tokens=len(transcript.split()),
            duration_s=round(float(len(audio) / sample_rate), 3),
            backend=self.name,
        )

    @staticmethod
    def _token_confidence(generated: object, model: object) -> float:
        """Geometric mean of the probability assigned to each emitted token."""
        import torch

        scores = getattr(generated, "scores", None)
        sequences = getattr(generated, "sequences", None)
        if not scores or sequences is None:
            return _FALLBACK_CONFIDENCE

        try:
            # Align from the end: Whisper's forced decoder prefix appears in
            # `sequences` but not in `scores`, so counting from position 1 would score
            # each token against the previous step.
            ids = sequences[0][-len(scores) :]
            total = 0.0
            counted = 0
            for step, logits in enumerate(scores):
                if step >= ids.shape[0]:
                    break
                token_id = int(ids[step])
                if token_id < 0:
                    continue
                log_probs = torch.log_softmax(logits[0].float(), dim=-1)
                total += float(log_probs[token_id])
                counted += 1
            if counted == 0:
                return _FALLBACK_CONFIDENCE
            return float(math.exp(total / counted))
        except Exception:  # noqa: BLE001 - confidence must never break a request
            logger.debug("token confidence unavailable; falling back", exc_info=True)
            return _FALLBACK_CONFIDENCE


# Reference backend (optional)


class OpenAIWhisperBackend:
    """The reference ``openai-whisper`` package. Requires ``ffmpeg`` on PATH."""

    name = "openai-whisper"

    def __init__(self, model_id: str = "small.en", device: str = "cpu") -> None:
        # The reference package addresses models by size name, not Hub id.
        self.model_name = model_id.split("/")[-1].replace("whisper-", "").replace(".en", ".en")
        self.model_name = "small.en" if "small" in self.model_name else self.model_name
        self.device = device
        self._model = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError as exc:
            raise ModelUnavailableError(
                "openai-whisper",
                "The optional openai-whisper package is not installed, and it also needs "
                "ffmpeg on the PATH. The default Transformers backend needs neither.",
            ) from exc
        self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult:
        """Persist to a temporary WAV, then decode with the reference package."""
        import soundfile as sf

        model = self._load()
        array = np.asarray(audio, dtype=np.float32)
        if sample_rate != 16_000:
            raise InferenceError("speech recognition", "openai-whisper expects 16 kHz audio.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            tmp_path = Path(handle.name)
        try:
            sf.write(str(tmp_path), array, sample_rate)
            result = model.transcribe(str(tmp_path), language="en", fp16=False)
        except Exception as exc:  # noqa: BLE001
            raise InferenceError("speech recognition", f"{type(exc).__name__}: {exc}") from exc
        finally:
            tmp_path.unlink(missing_ok=True)

        text = " ".join(str(result.get("text", "")).split())
        if not text.strip(" .,!?-"):
            raise NoSpeechError()

        segments = result.get("segments") or []
        if segments:
            # exp(mean avg_logprob) is the reference package's own uncertainty.
            avg_logprob = float(np.mean([float(s.get("avg_logprob", -1.0)) for s in segments]))
            confidence = float(min(1.0, math.exp(avg_logprob)))
        else:
            confidence = _FALLBACK_CONFIDENCE

        return ASRResult(
            text=text,
            confidence=round(confidence, 4),
            n_tokens=len(text.split()),
            duration_s=round(float(len(array) / sample_rate), 3),
            backend=self.name,
        )


# Factory


def build_asr(cfg: AppConfig) -> ASRBackend:
    """Select the configured backend. Import-light: models load on first use."""
    backend = (cfg.asr.backend or "transformers").lower()
    if backend in {"openai_whisper", "openai-whisper", "whisper"}:
        return OpenAIWhisperBackend(cfg.models.asr_id, device=cfg.device)
    if backend != "transformers":
        logger.warning("unknown ASR backend %r; using transformers", backend)
    return TransformersWhisperBackend(cfg.models.asr_id, device=cfg.device, cache_dir=cfg.cache_dir)
