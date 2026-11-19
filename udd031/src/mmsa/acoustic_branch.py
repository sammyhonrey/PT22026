"""Emotion from the waveform.

The fused head is a six-class probe over the frozen SUPERB encoder, if
``models/acoustic_probe.npz`` exists. Without it the branch falls back to the
zero-shot four-label classifier, whose missing fear/disgust mass is reported as
coverage. The optional dimensional head feeds the stress index and is never fused.
Held-out speakers 21-24: 0.6875 accuracy, 0.6745 macro-F1 (results/acoustic_probe.json).
"""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from .config import AppConfig
from .errors import InferenceError, LabelMapMismatchError, ModelUnavailableError
from .logging_utils import get_logger
from .ontology import Ontology
from .schemas import BranchResult, DimensionalResult

__all__ = ["AcousticBranch", "build_acoustic_branch"]

logger = get_logger(__name__)

_EXPECTED_LABELS = ("ang", "hap", "neu", "sad")
_DIMENSIONAL_ORDER = ("arousal", "dominance", "valence")
_TARGET_PEAK = 0.7079
"""Gain normalisation shared with ``scripts/train_acoustic_probe.py``.

Always -3 dBFS peak, the level ``audio_io.load_audio`` already produces. The fitted
head was trained on clips at that level; feeding it a quiet recording unchanged moves
its answer because of gain alone (measured: half amplitude turned ``angry 0.94`` into
``disgust 0.49``). Idempotent, so applying it after ``load_audio`` is safe.
"""


def _normalise_level(wave: np.ndarray, target_peak: float = _TARGET_PEAK) -> np.ndarray:
    """Scale a waveform to -3 dBFS peak, the level the trainer used."""
    x = np.asarray(wave, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(peak) or peak < 1e-6:
        return x
    return np.clip(x * (target_peak / peak), -1.0, 1.0).astype(np.float32)


class AcousticBranch:
    """Wav2vec2-based emotion recognition over the raw waveform."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        *,
        ontology: Ontology,
        cache_dir: str | None = None,
        dimensional_model_id: str | None = None,
        probe_path: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.ontology = ontology
        self.cache_dir = cache_dir
        self.dimensional_model_id = dimensional_model_id
        self.probe_path = probe_path
        self._feature_extractor = None
        self._model = None
        self._dim_extractor = None
        self._dim_model = None
        self._probe: dict | None = None
        self._probe_loaded = False

    # -- loading ------------------------------------------------------------- #

    def _load(self) -> tuple[object, object]:
        if self._model is not None and self._feature_extractor is not None:
            return self._feature_extractor, self._model

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:  # pragma: no cover - lean installs only
            raise ModelUnavailableError(
                self.model_id, "Transformers is required for the acoustic branch."
            ) from exc

        try:
            extractor = AutoFeatureExtractor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            model = AutoModelForAudioClassification.from_pretrained(
                self.model_id, cache_dir=self.cache_dir
            )
            model.eval()
            model.to(self.device)
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(self.model_id, f"{type(exc).__name__}: {exc}") from exc

        self._verify_labels(model)
        self._feature_extractor, self._model = extractor, model
        logger.info("loaded acoustic model %s on %s", self.model_id, self.device)
        return extractor, model

    def _verify_labels(self, model: object) -> None:
        """Fail loudly if the Hub repository's vocabulary changed under us."""
        id2label: Mapping[int, str] = getattr(getattr(model, "config", None), "id2label", {}) or {}
        found = tuple(str(id2label[i]).lower() for i in sorted(id2label))
        if set(found) != set(_EXPECTED_LABELS):
            raise LabelMapMismatchError(self.model_id, _EXPECTED_LABELS, found)

    def _load_dimensional(self) -> tuple[object, object] | None:
        if not self.dimensional_model_id:
            return None
        if self._dim_model is not None and self._dim_extractor is not None:
            return self._dim_extractor, self._dim_model

        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError:  # pragma: no cover
            return None

        try:
            extractor = AutoFeatureExtractor.from_pretrained(
                self.dimensional_model_id, cache_dir=self.cache_dir
            )
            model = AutoModelForAudioClassification.from_pretrained(
                self.dimensional_model_id, cache_dir=self.cache_dir
            )
            model.eval()
            model.to(self.device)
        except Exception as exc:  # noqa: BLE001 - optional head must not break the app
            logger.warning("dimensional head unavailable (%s); using proxies", exc)
            return None

        self._dim_extractor, self._dim_model = extractor, model
        logger.info("loaded dimensional head %s", self.dimensional_model_id)
        return extractor, model

    # -- inference ----------------------------------------------------------- #

    def _load_probe(self) -> dict | None:
        """Load the fitted six-class head, or return ``None`` for the fallback path."""
        if self._probe_loaded:
            return self._probe
        self._probe_loaded = True
        if not self.probe_path:
            return None

        from pathlib import Path

        path = Path(self.probe_path)
        if not path.exists():
            logger.info(
                "acoustic probe absent at %s; falling back to the zero-shot four-class head "
                "(train one with `python scripts/train_acoustic_probe.py`)",
                path,
            )
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                probe = {k: np.asarray(data[k]) for k in ("W", "b", "mean", "scale", "classes")}
        except Exception as exc:  # noqa: BLE001 - a corrupt probe must not break the app
            logger.warning("acoustic probe at %s could not be read (%s); using the fallback head", path, exc)
            return None

        n_classes = probe["classes"].size
        if probe["W"].shape != (n_classes, probe["mean"].size) or probe["b"].shape != (n_classes,):
            logger.warning("acoustic probe has inconsistent shapes (%s); using the fallback head", probe["W"].shape)
            return None
        classes = tuple(str(c) for c in probe["classes"])
        if set(classes) != set(self.ontology.canonical):
            logger.warning(
                "acoustic probe speaks %s but the ontology expects %s; using the fallback head",
                classes,
                self.ontology.canonical,
            )
            return None

        self._probe = probe
        logger.info("loaded fitted acoustic probe from %s (%d classes, %d features)",
                    path, n_classes, probe["mean"].size)
        return probe

    def _predict_probe(
        self, audio: np.ndarray, sample_rate: int, probe: dict
    ) -> BranchResult:
        """Score the clip with the fitted head over the frozen encoder."""
        import torch

        extractor, model = self._load()
        encoder = getattr(model, "wav2vec2", None)
        if encoder is None:  # pragma: no cover - defensive
            raise InferenceError("acoustic", "the loaded checkpoint exposes no wav2vec2 encoder")

        start = time.perf_counter()
        try:
            inputs = extractor(
                _normalise_level(np.asarray(audio, dtype=np.float32)),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                hidden = encoder(**inputs).last_hidden_state
                frame_mask = encoder._get_feature_vector_attention_mask(
                    hidden.shape[1], inputs["attention_mask"]
                ).unsqueeze(-1).float()
                count = frame_mask.sum(1).clamp(min=1)
                mean = (hidden * frame_mask).sum(1) / count
                var = (((hidden - mean.unsqueeze(1)) ** 2) * frame_mask).sum(1) / count
                pooled = torch.cat([mean, var.clamp(min=0).sqrt()], dim=1)
                features = pooled.cpu().numpy().astype(np.float64)[0]
        except Exception as exc:  # noqa: BLE001
            raise InferenceError("acoustic", f"{type(exc).__name__}: {exc}") from exc

        scaled = (features - probe["mean"]) / probe["scale"]
        logits = probe["W"] @ scaled + probe["b"]
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        classes = tuple(str(c) for c in probe["classes"])
        mapped = {c: float(probs[classes.index(c)]) for c in self.ontology.canonical}
        top = max(mapped.items(), key=lambda kv: kv[1])[0]

        return BranchResult(
            modality="acoustic",
            raw_labels=classes,
            raw_probs=tuple(float(p) for p in probs),
            mapped_probs=mapped,
            # The fitted head was trained on the canonical six-class ontology, so
            # nothing is unexpressible: coverage is complete and the conflict index
            # is meaningful on every request.
            coverage=1.0,
            confidence=float(_confidence_from(probs)),
            weight=0.0,
            per_class_weight={},
            top_label=top,
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    def predict(self, audio: np.ndarray, sample_rate: int) -> BranchResult:
        """Categorical emotion distribution projected onto the canonical space."""
        probe = self._load_probe()
        if probe is not None:
            return self._predict_probe(audio, sample_rate, probe)

        import torch

        extractor, model = self._load()
        start = time.perf_counter()

        try:
            inputs = extractor(
                np.asarray(audio, dtype=np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = model(**batch).logits
                probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            raise InferenceError("acoustic", f"{type(exc).__name__}: {exc}") from exc

        id2label: Mapping[int, str] = getattr(model.config, "id2label", {})
        labels = tuple(str(id2label[i]).lower() for i in sorted(id2label))
        ordered = np.asarray(probs, dtype=np.float64)

        mapped, coverage = self.ontology.map_probs("acoustic", labels, ordered)
        confidence = _confidence_from(ordered)
        top = max(mapped.items(), key=lambda kv: kv[1])[0]

        return BranchResult(
            modality="acoustic",
            raw_labels=labels,
            raw_probs=tuple(float(p) for p in ordered),
            mapped_probs=mapped,
            coverage=float(coverage),
            confidence=float(confidence),
            weight=0.0,
            per_class_weight={},
            top_label=top,
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    def predict_dimensional(self, audio: np.ndarray, sample_rate: int) -> DimensionalResult | None:
        """Arousal/dominance/valence, or ``None`` when the head is unavailable."""
        loaded = self._load_dimensional()
        if loaded is None:
            return None

        import torch

        extractor, model = loaded
        try:
            inputs = extractor(
                np.asarray(audio, dtype=np.float32),
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = model(**batch).logits
            values = logits.float().cpu().numpy().reshape(-1)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dimensional inference failed (%s); using proxies", exc)
            return None

        if values.size < 3:
            return None

        # The published output order is arousal, dominance, valence. Asserting the
        # length rather than trusting the name is what prevents a silent swap.
        arousal, dominance, valence = (float(np.clip(v, 0.0, 1.0)) for v in values[:3])
        return DimensionalResult(
            arousal=arousal,
            dominance=dominance,
            valence=valence,
            model_id=str(self.dimensional_model_id),
        )

    def warmup(self) -> None:
        """Run one synthetic second through the graph to pay the first-call cost."""
        rng = np.random.default_rng(0)
        dummy = (0.05 * rng.standard_normal(16_000)).astype(np.float32)
        try:
            self.predict(dummy, 16_000)
            self.predict_dimensional(dummy, 16_000)
        except Exception:  # noqa: BLE001 - warmup must never be fatal
            logger.warning("acoustic warmup failed", exc_info=True)


def _confidence_from(probs: np.ndarray) -> float:
    """1 - normalised entropy: 1 for a point mass, 0 for a uniform vector."""
    p = probs[probs > 0]
    if p.size <= 1:
        return 1.0
    p = p / p.sum()
    entropy = float(-(p * np.log(p)).sum())
    return float(max(0.0, min(1.0, 1.0 - entropy / np.log(probs.size))))


def build_acoustic_branch(cfg: AppConfig, ontology: Ontology) -> AcousticBranch:
    """Construct the acoustic branch from application configuration."""
    probe = getattr(cfg.models, "acoustic_probe", "") or ""
    return AcousticBranch(
        cfg.models.acoustic_id,
        device=cfg.device,
        ontology=ontology,
        cache_dir=cfg.cache_dir,
        dimensional_model_id=cfg.models.dimensional_id if cfg.stress.dimensional_enabled else None,
        probe_path=str(cfg.resolve(probe)) if probe else None,
    )
