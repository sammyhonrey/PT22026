"""End-to-end orchestration.

``analyse()`` takes an audio path and returns an immutable ``AnalysisResult``.
Branches are injected or built lazily, so the tests cover the whole flow without
loading a model. Input is validated before anything heavy is imported.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from .audio_io import lexical_negativity, load_audio
from .config import AppConfig, load_config
from .errors import MMSAError, MMSAWarning, LowTranscriptConfidenceWarning
from .evidence import lexical_evidence
from .fusion import fuse
from .logging_utils import Timer, get_logger, log_summary, request_context
from .narrative import build_narrative
from .ontology import Ontology
from .prosody import extract_prosody
from .schemas import AnalysisResult, ASRResult, BranchResult
from .stress import compute_stress

__all__ = ["AnalysisPipeline", "build_pipeline"]

logger = get_logger(__name__)


class AnalysisPipeline:
    """End-to-end analysis: audio in, fused affect and stress out."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        ontology: Ontology | None = None,
        asr: object | None = None,
        acoustic: object | None = None,
        text: object | None = None,
        prosody_reference: Mapping[str, Mapping[str, float]] | None = None,
        lexicon: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.ontology = ontology or Ontology.from_yaml(cfg.resolve("config/ontology.yaml"))

        # Built lazily so that importing this module never downloads anything.
        self._asr = asr
        self._acoustic = acoustic
        self._text = text

        self._prosody_reference = prosody_reference
        self._lexicon = lexicon
        self._loaded = False

    # -- lazy construction --------------------------------------------------- #

    def _ensure_components(self) -> None:
        if self._loaded:
            return
        if self._asr is None:
            from .asr import build_asr

            self._asr = build_asr(self.cfg)
        if self._acoustic is None:
            from .acoustic_branch import build_acoustic_branch

            self._acoustic = build_acoustic_branch(self.cfg, self.ontology)
        if self._text is None:
            from .text_branch import build_text_branch

            self._text = build_text_branch(self.cfg, self.ontology)

        if self._prosody_reference is None:
            self._prosody_reference = _load_json_if_present(
                self.cfg.resolve(self.cfg.stress.prosody_reference_file)
            )
        if self._lexicon is None:
            self._lexicon = _load_yaml_mapping(self.cfg.resolve(self.cfg.stress.lexicon_file))
        self._loaded = True

    def warmup(self) -> None:
        """Pay the first-call cost of every model behind one progress bar."""
        self._ensure_components()
        for name in ("acoustic", "text", "asr"):
            component = getattr(self, f"_{name}")
            hook = getattr(component, "warmup", None)
            if callable(hook):
                hook()
        logger.info("pipeline warmup complete")

    # -- main entry point ---------------------------------------------------- #

    def analyse(self, audio_path: str | Path, *, request_id: str | None = None) -> AnalysisResult:
        """Analyse one clip.

        Raises
        ------
        MMSAError
            For any user-visible failure; the message is safe to display.
        """
        self._ensure_components()
        warnings: list[MMSAWarning] = []
        timer = Timer()

        with request_context(request_id) as rid:
            logger.info("analysis started for %s", Path(audio_path).name)

            with timer.stage("audio"):
                audio, audio_info = load_audio(
                    audio_path,
                    min_duration_s=self.cfg.min_duration_s,
                    max_duration_s=self.cfg.max_duration_s,
                    max_file_mb=self.cfg.max_file_mb,
                    allowed_extensions=self.cfg.allowed_extensions,
                    warnings=warnings,
                )

            with timer.stage("asr"):
                asr_result: ASRResult = self._asr.transcribe(audio, audio_info.sample_rate)  # type: ignore[union-attr]

            if asr_result.confidence < self.cfg.asr.low_confidence_threshold:
                warnings.append(LowTranscriptConfidenceWarning(asr_result.confidence))

            with timer.stage("acoustic"):
                acoustic_result: BranchResult = self._acoustic.predict(audio, audio_info.sample_rate)  # type: ignore[union-attr]

            with timer.stage("text"):
                text_result: BranchResult = self._text.predict(asr_result.text)  # type: ignore[union-attr]

            with timer.stage("prosody"):
                prosody = extract_prosody(audio, audio_info.sample_rate)

            # How much the transcript itself says about the speaker's state. On read
            # speech this is ~0, and fusion then lets the audio decide instead of
            # averaging it against a text opinion that has no evidence behind it.
            evidence = lexical_evidence(asr_result.text, self._lexicon)

            with timer.stage("fusion"):
                fusion = fuse(acoustic_result, text_result, self.cfg.fusion,
                              text_evidence=evidence.score)

            acoustic_result = replace(
                acoustic_result,
                weight=fusion.per_class_weights["acoustic"][acoustic_result.top_label],
                per_class_weight=dict(fusion.per_class_weights["acoustic"]),
            )
            text_result = replace(
                text_result,
                weight=fusion.per_class_weights["text"][text_result.top_label],
                per_class_weight=dict(fusion.per_class_weights["text"]),
            )

            with timer.stage("stress"):
                lexicon_score, _hits = lexical_negativity(
                    asr_result.text,
                    dict(self._lexicon or {}),
                    max_hits=self.cfg.stress.lexicon_max_hits,
                )
                dimensional = None
                predictor = getattr(self._acoustic, "predict_dimensional", None)
                if callable(predictor):
                    dimensional = predictor(audio, audio_info.sample_rate)

                stress = compute_stress(
                    fusion.probs,
                    cfg=self.cfg.stress,
                    features=prosody,
                    asr_confidence=asr_result.confidence,
                    lexical_negative=lexicon_score,
                    dimensional=dimensional,
                    prosody_reference=self._prosody_reference,
                )

            with timer.stage("narrative"):
                narrative, short = build_narrative(
                    request_id=rid,
                    fused_probs=fusion.probs,
                    acoustic=acoustic_result,
                    text=text_result,
                    stress=stress,
                    agreement=fusion.agreement,
                    conflict_index=fusion.conflict_index,
                    ignorance=fusion.ignorance,
                    asr_confidence=asr_result.confidence,
                    n_canonical=len(self.ontology.canonical),
                )

            timings = dict(timer.timings_ms)
            timings["total"] = timer.total_ms

            result = AnalysisResult(
                request_id=rid,
                created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                audio=audio_info,
                asr=asr_result,
                acoustic=acoustic_result,
                text=text_result,
                fused_probs=dict(fusion.probs),
                fused_emotion=fusion.top_label,
                fused_confidence=fusion.confidence,
                entropy=fusion.entropy,
                ignorance=fusion.ignorance,
                conflict_index=fusion.conflict_index,
                agreement=fusion.agreement,
                stress=stress,
                narrative=narrative,
                narrative_short=short,
                timings_ms=timings,
                dimensional=dimensional,
                warnings=tuple(w.user_message for w in warnings),
                text_evidence=evidence.score,
            )

            log_summary(
                logger,
                {
                    "fused_emotion": result.fused_emotion,
                    "fused_confidence": result.fused_confidence,
                    "stress_score": result.stress.score,
                    "stress_band": result.stress.band,
                    "conflict_index": result.conflict_index,
                    "agreement": result.agreement,
                    "total_ms": timings["total"],
                },
            )
            return result

    # -- transcript-only rerun (used by the "edit transcript" control) ------- #

    def reanalyse_text(self, transcript: str, previous: AnalysisResult) -> AnalysisResult:
        """Re-run the text branch on an edited transcript and refuse everything."""
        self._ensure_components()
        text_result: BranchResult = self._text.predict(transcript)  # type: ignore[union-attr]
        self._ensure_components()
        evidence = lexical_evidence(transcript, self._lexicon)
        fusion = fuse(previous.acoustic, text_result, self.cfg.fusion, text_evidence=evidence.score)

        text_result = replace(
            text_result,
            weight=fusion.per_class_weights["text"][text_result.top_label],
            per_class_weight=dict(fusion.per_class_weights["text"]),
        )

        narrative, short = build_narrative(
            request_id=previous.request_id,
            fused_probs=fusion.probs,
            acoustic=previous.acoustic,
            text=text_result,
            stress=previous.stress,
            agreement=fusion.agreement,
            conflict_index=fusion.conflict_index,
            ignorance=fusion.ignorance,
            asr_confidence=previous.asr.confidence,
            n_canonical=len(self.ontology.canonical),
        )

        return replace(
            previous,
            text=text_result,
            asr=replace(previous.asr, text=transcript),
            fused_probs=dict(fusion.probs),
            fused_emotion=fusion.top_label,
            fused_confidence=fusion.confidence,
            entropy=fusion.entropy,
            ignorance=fusion.ignorance,
            conflict_index=fusion.conflict_index,
            agreement=fusion.agreement,
            narrative=narrative,
            narrative_short=short,
            text_evidence=evidence.score,
        )


# Helpers


def _load_json_if_present(path: Path) -> dict | None:
    if not path.exists():
        logger.info("prosody reference not found at %s; instability term disabled", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - corrupt file path
        logger.warning("could not read %s; instability term disabled", path)
        return None


def _load_yaml_mapping(path: Path) -> dict:
    if not path.exists():
        logger.warning("lexicon not found at %s; lexical term disabled", path)
        return {}
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: v for k, v in data.items() if isinstance(v, list)}


def build_pipeline(cfg: AppConfig | None = None, **kwargs: object) -> AnalysisPipeline:
    """Convenience constructor used by the app and the scripts."""
    return AnalysisPipeline(cfg or load_config(), **kwargs)  # type: ignore[arg-type]


def result_to_json(result: AnalysisResult, path: str | Path) -> Path:
    """Persist a result for the evaluation scripts and the report figures."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    return target


def safe_analyse(pipeline: AnalysisPipeline, audio_path: str) -> tuple[AnalysisResult | None, MMSAError | None]:
    """Analyse without raising; the UI layer wants a value, not an exception."""
    try:
        return pipeline.analyse(audio_path), None
    except MMSAError as exc:
        logger.warning("analysis failed: %s", exc.code)
        return None, exc
