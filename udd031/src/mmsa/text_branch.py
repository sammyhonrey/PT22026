"""The text branch: emotion from the transcript alone.

``bhadresh-savani/distilbert-base-uncased-emotion`` -- DistilBERT fine-tuned on the
six-class ``emotion`` dataset (sadness, joy, love, anger, fear, surprise).

The branch is intentionally independent of the audio path: it sees only a string.
That independence is what makes the conflict index meaningful, and it is why the
interface lets a user edit the transcript and re-run this branch in isolation.
"""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from .config import AppConfig
from .errors import InferenceError, LabelMapMismatchError, ModelUnavailableError, NoSpeechError
from .logging_utils import get_logger
from .ontology import Ontology
from .schemas import BranchResult

__all__ = ["TextBranch", "build_text_branch"]

logger = get_logger(__name__)

_EXPECTED_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")


class TextBranch:
    """DistilBERT emotion classifier over transcribed speech."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        *,
        ontology: Ontology,
        cache_dir: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.ontology = ontology
        self.cache_dir = cache_dir
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load(self) -> tuple[object, object]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - lean installs only
            raise ModelUnavailableError(
                self.model_id, "Transformers is required for the text branch."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model.eval()
            model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(self.model_id, f"{type(exc).__name__}: {exc}") from exc

        self._verify_labels(model)
        self._tokenizer, self._model = tokenizer, model
        logger.info("loaded text model %s on %s", self.model_id, self.device)
        return tokenizer, model

    def _verify_labels(self, model: object) -> None:
        id2label: Mapping[int, str] = getattr(getattr(model, "config", None), "id2label", {}) or {}
        found = tuple(str(id2label[i]).lower() for i in sorted(id2label))
        if set(found) != set(_EXPECTED_LABELS):
            raise LabelMapMismatchError(self.model_id, _EXPECTED_LABELS, found)

    def predict(self, text: str) -> BranchResult:
        """Canonical emotion distribution for one transcript."""
        import torch

        cleaned = " ".join(str(text or "").split())
        if not cleaned.strip(" .,!?-"):
            raise NoSpeechError()

        tokenizer, model = self._load()
        start = time.perf_counter()

        try:
            encoded = tokenizer(
                cleaned,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
            batch = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = model(**batch).logits
                probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            raise InferenceError("text", f"{type(exc).__name__}: {exc}") from exc

        id2label: Mapping[int, str] = getattr(model.config, "id2label", {})
        labels = tuple(str(id2label[i]).lower() for i in sorted(id2label))
        ordered = np.asarray(probs, dtype=np.float64)

        mapped, coverage = self.ontology.map_probs("text", labels, ordered)
        top = max(mapped.items(), key=lambda kv: kv[1])[0]

        return BranchResult(
            modality="text",
            raw_labels=labels,
            raw_probs=tuple(float(p) for p in ordered),
            mapped_probs=mapped,
            coverage=float(coverage),
            confidence=_confidence_from(ordered),
            weight=0.0,
            per_class_weight={},
            top_label=top,
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    def warmup(self) -> None:
        """Prime the graph with a short neutral sentence."""
        try:
            self.predict("This is a warmup sentence.")
        except Exception:  # noqa: BLE001
            logger.warning("text warmup failed", exc_info=True)


def _confidence_from(probs: np.ndarray) -> float:
    """1 - normalised entropy."""
    p = probs[probs > 0]
    if p.size <= 1:
        return 1.0
    p = p / p.sum()
    entropy = float(-(p * np.log(p)).sum())
    return float(max(0.0, min(1.0, 1.0 - entropy / np.log(probs.size))))


def build_text_branch(cfg: AppConfig, ontology: Ontology) -> TextBranch:
    """Construct the text branch from application configuration."""
    return TextBranch(
        cfg.models.text_id,
        device=cfg.device,
        ontology=ontology,
        cache_dir=cfg.cache_dir,
    )
