"""Gradio page assembly.

Input and transcript on the left; acoustic, text and fused verdicts on the right,
followed by a sentence of interpretation. Tabs below hold branch comparison,
stress decomposition, diagnostics and the text-branch counterfactual. Handlers
fail into a toast rather than a traceback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import gradio as gr

from .. import charts
from ..config import AppConfig, load_config
from ..errors import MMSAError
from ..pipeline import AnalysisPipeline, build_pipeline, safe_analyse
from ..schemas import AnalysisResult
from . import components as C
from .theme import build_theme, load_css

__all__ = ["build_demo", "launch"]

_CSS_PATH = Path(__file__).with_name("styles.css")

_MODEL_FALLBACK = {
    "asr": "openai/whisper-small.en",
    "acoustic": "superb/wav2vec2-base-superb-er",
    "text": "bhadresh-savani/distilbert-base-uncased-emotion",
}

_AGREEMENT_TONE = {
    "aligned": "ok",
    "mild_conflict": "warn",
    "strong_conflict": "alert",
    "indeterminate": "",
}


# Formatting helpers


def _model_id(cfg: AppConfig, branch: str) -> str:
    """Read ``cfg.models.<branch>_id`` with a documented fallback.

    The ids live on :class:`~mmsa.config.ModelConfig`, not on the branch configs,
    so this reads one place and falls back to the verified Hub ids from the
    model-selection record rather than printing ``unknown`` to an examiner.
    """
    models = getattr(cfg, "models", None)
    value = getattr(models, f"{branch}_id", None) if models is not None else None
    if isinstance(value, str) and value:
        return value
    return _MODEL_FALLBACK.get(branch, "unknown")


def _branch_panel(label: str, branch: Any, *, fused: bool = False) -> str:
    """Render one branch verdict with its top class and coverage."""
    confidence = float(getattr(branch, "confidence", 0.0))
    coverage = float(getattr(branch, "coverage", 0.0))
    top = getattr(branch, "top_label", "—")
    meta = (
        f"confidence {confidence:.0%} · ontology coverage {coverage:.0%}"
        f" · {getattr(branch, 'latency_ms', 0.0):.0f} ms"
    )
    return C.result_panel_html(label, top.capitalize(), meta, fused=fused)


def _chip_set(result: AnalysisResult) -> str:
    """Status chips: agreement, stress band, warnings, request id."""
    chips: list[tuple[str, str]] = [
        (result.agreement.replace("_", " "), _AGREEMENT_TONE.get(result.agreement, "")),
        (f"stress: {result.stress.band_label}", "warn" if result.stress.band == "High" else ""),
    ]
    if result.conflict_index is not None:
        chips.append((f"conflict {result.conflict_index:.2f}", ""))
    if result.ignorance > 0.25:
        chips.append((f"ignorance {result.ignorance:.0%}", "warn"))
    for warning in result.warnings:
        chips.append((warning, "alert"))
    chips.append((f"id {result.request_id}", ""))
    return C.chip_row_html(chips)


def _metrics(result: AnalysisResult) -> str:
    """Compact diagnostic card row."""
    snr = result.audio.snr_db
    items = [
        ("duration", f"{result.audio.duration_s:.2f}s", f"{result.audio.sample_rate} Hz"),
        ("asr confidence", f"{result.asr.confidence:.2f}", f"{result.asr.n_tokens} tokens"),
        ("entropy", f"{result.entropy:.3f}", "nats, fused"),
        ("ignorance", f"{result.ignorance:.2f}", "unexpressed mass"),
        ("snr", f"{snr:.1f} dB" if snr is not None else "n/a", "estimated"),
        ("latency", f"{result.timings_ms.get('total', 0.0):.0f} ms", "warm path"),
    ]
    return C.metric_row_html(items)


def _render(result: AnalysisResult) -> tuple:
    """Map one result onto the page's output contract (see ``_outputs``)."""
    return (
        C.transcript_html(result.asr.text or "_(silence detected — no words decoded)_"),
        _chip_set(result),
        _branch_panel("Acoustic branch · voice only", result.acoustic),
        _branch_panel("Text branch · words only", result.text),
        _fused_panel(result),
        charts.confidence_ring(result),
        charts.emotion_bars(result),
        charts.radar_overlay(result),
        charts.stress_waterfall(result),
        C.narrative_html(result.narrative),
        _metrics(result),
        charts.latency_bars(result),
        result.as_dict(),
        result.asr.text,
    )


def _fused_panel(result: AnalysisResult) -> str:
    """The emphasised fused column: class, stress band, agreement in one block."""
    meta = (
        f"confidence {result.fused_confidence:.0%} · stress {result.stress.score:.2f}"
        f" · {result.agreement.replace('_', ' ')}"
    )
    return C.result_panel_html("Fused estimate · both channels", result.fused_emotion.capitalize(), meta, fused=True)


def _blank(narrative: str = "") -> tuple:
    """Empty state, used for the first paint and after a handled failure."""
    note = narrative or '<span style="color:#93A4C3">Awaiting a clip. Record or upload audio, then press <b>Analyse</b>.</span>'
    return (
        C.transcript_html(""),
        "",
        C.result_panel_html("Acoustic branch · voice only", "—", "no result yet"),
        C.result_panel_html("Text branch · words only", "—", "no result yet"),
        C.result_panel_html("Fused estimate · both channels", "—", "no result yet", fused=True),
        charts.empty_figure("confidence"),
        charts.empty_figure("per-class probabilities"),
        charts.empty_figure("branch geometry"),
        charts.empty_figure("stress decomposition"),
        C.narrative_html(note),
        C.metric_row_html([("status", "idle", "nothing analysed")]),
        charts.empty_figure("latency"),
        None,
        "",
    )


# Page


def build_demo(cfg: AppConfig | None = None, pipeline: AnalysisPipeline | None = None) -> gr.Blocks:
    """Construct the Blocks application.

    ``pipeline`` may be injected so the interface can be driven end-to-end by the
    test-suite's fakes, with no models and no network.
    """
    cfg = cfg or load_config()
    pipe = pipeline or build_pipeline(cfg)

    css = load_css(str(_CSS_PATH))
    theme = build_theme()

    examples = _examples(cfg)

    with gr.Blocks(
        title="Multimodal Mental State Analyser · UDDIN MEJBAH",
        theme=theme,
        css=css,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(C.hero_html())
        gr.HTML(C.disclaimer_html())

        state = gr.State(value=None)

        # -- row 1: input / verdicts ---------------------------------------- #
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=320):
                audio_in = gr.Audio(
                    label="Speech clip (WAV / MP3 / FLAC / OGG / M4A)",
                    sources=["microphone", "upload"],
                    type="filepath",
                    interactive=True,
                )
                with gr.Row():
                    analyse_btn = gr.Button("Analyse clip", variant="primary", scale=3)
                    clear_btn = gr.Button("Reset", variant="secondary", scale=1)
                transcript_box = gr.HTML(C.transcript_html(""))
                chips_out = gr.HTML("")
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[audio_in],
                        label="Demo clips (RAVDESS, CC BY-NC-SA 4.0)",
                    )

            with gr.Column(scale=6, min_width=380):
                with gr.Row(equal_height=True):
                    acoustic_out = gr.HTML(C.result_panel_html("Acoustic branch · voice only", "—", "no result yet"))
                    text_out = gr.HTML(C.result_panel_html("Text branch · words only", "—", "no result yet"))
                with gr.Row(equal_height=True):
                    fused_out = gr.HTML(
                        C.result_panel_html("Fused estimate · both channels", "—", "no result yet", fused=True)
                    )
                    ring_out = gr.Plot(value=charts.empty_figure("confidence"), show_label=False)
                narrative_out = gr.HTML(C.narrative_html(
                    '<span style="color:#93A4C3">The interpretation appears here after analysis.</span>'
                ))

        # -- row 2: evidence ------------------------------------------------- #
        with gr.Tabs():
            with gr.Tab("Branch comparison"):
                with gr.Row():
                    bars_out = gr.Plot(value=charts.empty_figure("per-class probabilities"), show_label=False)
                    radar_out = gr.Plot(value=charts.empty_figure("branch geometry"), show_label=False)
                gr.HTML(
                    '<div class="mmsa-panel__meta" style="padding:0 4px">'
                    "Acoustic and text bars show <b>mapped mass</b>, which sums to that branch's coverage and "
                    "may therefore stop short of 1.0. The shortfall is ignorance the branch cannot express, not "
                    "a rendering error.</div>"
                )

            with gr.Tab("Stress decomposition"):
                stress_out = gr.Plot(value=charts.empty_figure("stress decomposition"), show_label=False)
                gr.HTML(
                    '<div class="mmsa-panel__meta" style="padding:0 4px">'
                    "Bars are signed contributions to the logit, which a sigmoid maps to the 0–1 stress index. "
                    "This is a transparent composite of prosodic, lexical and uncertainty terms — it is "
                    "<b>not</b> a trained predictor and no accuracy is claimed for it.</div>"
                )

            with gr.Tab("Diagnostics"):
                latency_out = gr.Plot(value=charts.empty_figure("latency"), show_label=False)
                metrics_out = gr.HTML(C.metric_row_html([("status", "idle", "nothing analysed")]))
                with gr.Accordion("Raw result (JSON)", open=False):
                    json_out = gr.JSON(value=None, show_label=False)
                with gr.Accordion("Counterfactual probe — edit the transcript", open=False):
                    gr.HTML(
                        '<div class="mmsa-panel__meta" style="padding:2px 0 8px">'
                        "Re-run the text branch on a corrected transcript while holding the acoustics fixed. "
                        "This isolates how much of the verdict came from the words alone.</div>"
                    )
                    edited_transcript = gr.Textbox(
                        label="Transcript", lines=3, placeholder="Analyse a clip first…"
                    )
                    rerun_btn = gr.Button("Re-run text branch", variant="secondary")

            with gr.Tab("About the system"):
                gr.HTML(_about_html(cfg))

        # -- output contract ------------------------------------------------- #
        outputs = [
            transcript_box,
            chips_out,
            acoustic_out,
            text_out,
            fused_out,
            ring_out,
            bars_out,
            radar_out,
            stress_out,
            narrative_out,
            metrics_out,
            latency_out,
            json_out,
            edited_transcript,
        ]

        # -- handlers -------------------------------------------------------- #

        def _on_analyse(path: str | None) -> tuple:
            if not path:
                gr.Warning("Record or upload a clip first.")
                return _blank()
            result, error = safe_analyse(pipe, path)
            if error is not None or result is None:
                gr.Warning(_error_text(error))
                return _blank(_error_html(error))
            state.value = result
            return _render(result)

        def _on_rerun(transcript: str | None) -> tuple:
            previous = state.value
            if not isinstance(previous, AnalysisResult):
                gr.Warning("Analyse a clip before editing the transcript.")
                return _blank()
            if not (transcript or "").strip():
                gr.Warning("The edited transcript is empty.")
                return _blank()
            try:
                updated = pipe.reanalyse_text(transcript.strip(), previous)
            except MMSAError as exc:
                gr.Warning(_error_text(exc))
                return _blank(_error_html(exc))
            except Exception as exc:  # noqa: BLE001 - never leak a traceback
                gr.Warning(f"Re-run failed: {type(exc).__name__}")
                return _blank()
            state.value = updated
            return _render(updated)

        def _on_clear() -> tuple:
            state.value = None
            return (None, *_blank())

        analyse_btn.click(_on_analyse, inputs=[audio_in], outputs=outputs, api_name="analyse")
        rerun_btn.click(_on_rerun, inputs=[edited_transcript], outputs=outputs)
        clear_btn.click(_on_clear, inputs=None, outputs=[audio_in, *outputs])

        demo.load(lambda: _blank()[9], inputs=None, outputs=[narrative_out])

    return demo.queue(default_concurrency_limit=1, max_size=16)


def _error_text(error: MMSAError | None) -> str:
    if error is None:
        return "Analysis failed for an unknown reason."
    return f"{getattr(error, 'code', 'E???')} — {getattr(error, 'user_message', str(error))}"


def _error_html(error: MMSAError | None) -> str:
    if error is None:
        return C.narrative_html("Analysis failed. Check the clip and try again.")
    code = getattr(error, "code", "E???")
    hint = getattr(error, "hint", "") or ""
    message = getattr(error, "user_message", str(error))
    body = f"<b>{code}</b> · {message}"
    if hint:
        body += f'<br><span style="color:#93A4C3">{hint}</span>'
    return C.narrative_html(body)


def _examples(cfg: AppConfig) -> list[list[str]]:
    """Demo clips, if they have been fetched. Never invent file paths."""
    folder = Path(cfg.project_root) / "demo_samples"
    if not folder.is_dir():
        return []
    clips = sorted(p for p in folder.glob("*.wav") if p.is_file())[:6]
    return [[str(p)] for p in clips]


def _about_html(cfg: AppConfig) -> str:
    """Provenance and design notes, shown in the last tab."""
    return f"""
<div class="mmsa-panel">
  <div class="mmsa-panel__label">System card</div>
  <div class="mmsa-panel__value" style="font-size:18px">One clip, three opinions, one honest verdict</div>
  <div class="mmsa-panel__meta" style="margin-top:12px;line-height:1.7">
    <b>ASR</b> · <code>{_model_id(cfg, "asr")}</code><br>
    <b>Acoustic branch</b> · <code>{_model_id(cfg, "acoustic")}</code><br>
    <b>Text branch</b> · <code>{_model_id(cfg, "text")}</code><br>
    <b>Fusion</b> · per-class reliability weighting with explicit ignorance; JSD-based conflict index<br>
    <b>Stress</b> · transparent seven-term composite, reported with its contributors, never as an accuracy claim
  </div>
  <div class="mmsa-chip-row">
    <span class="mmsa-chip">UDDIN MEJBAH</span>
    <span class="mmsa-chip">20243120031</span>
    <span class="mmsa-chip">Professional Training · YN3012170034</span>
    <span class="mmsa-chip">Research prototype</span>
  </div>
</div>
"""


def launch(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    cfg: AppConfig | None = None,
) -> None:
    """Build and serve the interface."""
    demo = build_demo(cfg)
    demo.launch(server_name=host, server_port=port, share=share, show_api=False, inbrowser=False)
