"""Reusable interface fragments.

Kept separate from ``app_ui`` so that the page assembly reads as layout rather than
as a wall of HTML strings, and so a fragment can be unit-tested or reused on a
Hugging Face Space landing page.
"""

from __future__ import annotations

from .theme import TOKENS

__all__ = [
    "hero_html",
    "disclaimer_html",
    "panel_header_html",
    "result_panel_html",
    "chip_row_html",
    "narrative_html",
    "transcript_html",
    "metric_row_html",
]


def hero_html(
    *,
    student: str = "UDDIN MEJBAH",
    student_id: str = "20243120031",
    course: str = "Professional Training · YN3012170034",
) -> str:
    """Title block: course line, project title, one-sentence pitch, identity chips."""
    return f"""
<div class="mmsa-hero">
  <div class="mmsa-kicker">{course}</div>
  <h1 class="mmsa-title">Multimodal Mental State Analyser</h1>
  <p class="mmsa-sub">
    Emotion and stress prediction by fusing speech acoustics with the transcript of the same
    utterance. Two frozen transformer branches read the voice and the words independently; a
    coverage-aware fusion engine reconciles them and reports what it cannot know.
  </p>
  <div class="mmsa-badge-row">
    <span class="mmsa-badge mmsa-badge--accent">{student}</span>
    <span class="mmsa-badge">{student_id}</span>
    <span class="mmsa-badge">Software Engineering</span>
    <span class="mmsa-badge">Whisper · wav2vec2 · DistilBERT</span>
  </div>
</div>
"""


def disclaimer_html() -> str:
    """The ethics block. Present on first paint, never behind a click."""
    return """
<div class="mmsa-disclaimer">
  <strong>Research prototype — not a diagnostic instrument.</strong>
  This system is an experimental AI screening aid. It does not diagnose any condition and
  cannot replace assessment by a psychologist or psychiatrist. Do not use it to make
  decisions about employment, insurance, or care. Audio is processed in memory for the
  duration of the request and is not written to disk by this application.
</div>
"""


def panel_header_html(label: str, value: str, meta: str = "") -> str:
    """A labelled value block used inside the three result columns."""
    meta_html = f'<div class="mmsa-panel__meta">{meta}</div>' if meta else ""
    return f"""
<div>
  <div class="mmsa-panel__label">{label}</div>
  <div class="mmsa-panel__value">{value}</div>
  {meta_html}
</div>
"""


def result_panel_html(label: str, value: str, meta: str, *, fused: bool = False) -> str:
    """Full glass panel, optionally the emphasised fused variant."""
    modifier = " mmsa-panel--fused" if fused else ""
    return f"""
<div class="mmsa-panel{modifier}">
  <div class="mmsa-panel__label">{label}</div>
  <div class="mmsa-panel__value">{value}</div>
  <div class="mmsa-panel__meta">{meta}</div>
</div>
"""


def chip_row_html(chips: list[tuple[str, str]]) -> str:
    """Render ``[(text, tone)]`` where tone is ``""``, ``ok``, ``warn`` or ``alert``."""
    parts = []
    for text, tone in chips:
        cls = f"mmsa-chip mmsa-chip--{tone}" if tone else "mmsa-chip"
        parts.append(f'<span class="{cls}">{text}</span>')
    return '<div class="mmsa-chip-row">' + "".join(parts) + "</div>"


def narrative_html(text: str) -> str:
    """The interpretive callout."""
    return f'<div class="mmsa-narrative">{text}</div>'


def transcript_html(text: str, *, placeholder: str = "The transcript will appear here.") -> str:
    """Monospace transcript block."""
    body = text if text else f'<span style="color:{TOKENS["muted"]}">{placeholder}</span>'
    return f'<div class="mmsa-transcript">{body}</div>'


def metric_row_html(items: list[tuple[str, str, str]]) -> str:
    """Render ``[(label, value, meta)]`` as a row of compact metric cards."""
    cards = []
    for label, value, meta in items:
        cards.append(
            f"""
<div class="mmsa-panel" style="padding:12px 14px">
  <div class="mmsa-panel__label">{label}</div>
  <div class="mmsa-panel__value" style="font-size:19px">{value}</div>
  <div class="mmsa-panel__meta" style="font-size:10.5px">{meta}</div>
</div>"""
        )
    return '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px">' + "".join(cards) + "</div>"
