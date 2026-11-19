"""ReviewLens - English Product Review Sentiment Analyzer.

Single-file Gradio app. Uses the same model as version 2/3:
models/sentiment_pipeline_v2.joblib  (TF-IDF + Logistic Regression).

What it shows
  Single review : verdict card with emoji, four metric boxes, probability
                  gauge, highlighted words, contribution bar chart,
                  mixed-signal warning, and a history of recent analyses.
  Batch         : paste or upload many reviews -> donut chart, histogram of
                  positive probability, most influential words, table, CSV.
  About         : accuracy cards, pipeline diagram, how to read results.
"""

import html
import inspect
import re
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "sentiment_pipeline_v2.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Complete the version 2 training first "
        f"(expected: {MODEL_PATH})."
    )

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------- model parts
vectorizer = model.named_steps["tfidf"]
classifier = model.steps[-1][1]
vocabulary = vectorizer.vocabulary_
feature_names = vectorizer.get_feature_names_out()

classes = list(model.classes_)
POSITIVE_COL = classes.index(1)
NEGATIVE_COL = classes.index(0)

# Binary coef_ belongs to classes_[1]; flip the sign if 1 is not that class.
coef = None
COEF_SIGN = 1.0
if hasattr(classifier, "coef_"):
    coef = np.asarray(classifier.coef_[0], dtype=float)
    COEF_SIGN = 1.0 if POSITIVE_COL == 1 else -1.0
signed_coef = None if coef is None else coef * COEF_SIGN

# A display rule only; this threshold has not been calibrated.
UNCERTAINTY_THRESHOLD = 0.60
MAX_CHARS = 5000
MAX_BATCH = 500
HISTORY_SIZE = 6

VERDICTS = {
    "positive": ("👍", "Positive review"),
    "negative": ("👎", "Negative review"),
    "uncertain": ("⚖️", "Too close to call"),
}
TABLE_LABELS = {
    "positive": "👍 Positive",
    "negative": "👎 Negative",
    "uncertain": "⚖️ Uncertain",
    "unreadable": "🔤 No known words",
}

# ------------------------------------------------------------------------ CSS
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --rl-ink: #1b1f2a;
    --rl-muted: #5b6475;
    --rl-surface: #ffffff;
    --rl-soft: #f3f5f9;
    --rl-line: #dfe4ee;
    --rl-pos: #17794f;
    --rl-pos-rgb: 23, 150, 90;
    --rl-pos-bg: #e6f4ec;
    --rl-neg: #b8312a;
    --rl-neg-rgb: 210, 60, 50;
    --rl-neg-bg: #fbe9e7;
    --rl-unc: #9a5f0a;
    --rl-unc-bg: #fbf1dc;
    --rl-info: #1f4fd8;
    --rl-info-bg: #e8eefc;
}
.dark {
    --rl-ink: #eef1f7;
    --rl-muted: #a3adc0;
    --rl-surface: #161b26;
    --rl-soft: #1c2230;
    --rl-line: #2d3548;
    --rl-pos: #5fd39a;
    --rl-pos-rgb: 70, 200, 130;
    --rl-pos-bg: #12301f;
    --rl-neg: #f58f86;
    --rl-neg-rgb: 245, 120, 110;
    --rl-neg-bg: #3b1a18;
    --rl-unc: #ecbb6c;
    --rl-unc-bg: #3a2a0d;
    --rl-info: #8aa7ff;
    --rl-info-bg: #1a2447;
}

.gradio-container {
    max-width: 1120px !important;
    margin: auto !important;
    font-family: 'Public Sans', 'Segoe UI', Arial, sans-serif !important;
}

/* ---- header ---- */
#rl-header {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 22px 24px;
    margin-bottom: 14px;
    border: 1px solid var(--rl-line);
    border-radius: 16px;
    background: var(--rl-surface);
}
#rl-header .logo {
    flex: none;
    width: 64px;
    height: 64px;
    display: grid;
    place-items: center;
    font-size: 34px;
    border-radius: 16px;
    background: var(--rl-info-bg);
}
#rl-header h1 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: clamp(28px, 4.4vw, 42px);
    font-weight: 600;
    line-height: 1.1;
    letter-spacing: -0.4px;
    color: var(--rl-ink) !important;
    margin: 0 0 6px;
}
#rl-header p { color: var(--rl-muted) !important; font-size: 15.5px; line-height: 1.55; margin: 0; max-width: 620px; }
.rl-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.rl-tags span {
    background: var(--rl-soft);
    border: 1px solid var(--rl-line);
    color: var(--rl-muted);
    border-radius: 999px;
    padding: 4px 11px;
    font-size: 12.5px;
}

/* ---- panels ---- */
.rl-panel {
    border: 1px solid var(--rl-line) !important;
    border-radius: 14px !important;
    padding: 20px !important;
    background: var(--rl-surface) !important;
}
.rl-panel h3 { margin-top: 0; }
#review-input textarea { font-size: 16px !important; line-height: 1.7 !important; }
#analyze-button, #batch-button { min-height: 48px; border-radius: 10px !important; font-weight: 600 !important; }
.rl-counter { color: var(--rl-muted); font-size: 12.5px; text-align: right; margin-top: -4px; }
.rl-counter.over { color: var(--rl-neg); font-weight: 600; }

/* ---- small boxes ---- */
.rl-note {
    display: flex;
    gap: 10px;
    padding: 12px 14px;
    border-radius: 10px;
    font-size: 13.5px;
    line-height: 1.55;
    background: var(--rl-info-bg);
    color: var(--rl-ink);
    border: 1px solid var(--rl-line);
}
.rl-note.warn { background: var(--rl-unc-bg); }
.rl-note .ico { flex: none; font-size: 18px; line-height: 1.3; }
.rl-note ul { margin: 4px 0 0; padding-left: 18px; }
.rl-note b { color: var(--rl-ink); }

/* ---- verdict card ---- */
.rl-verdict {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 18px 20px;
    border-radius: 14px;
    border: 1px solid var(--rl-line);
    background: var(--rl-soft);
    color: var(--rl-ink);
}
.rl-verdict.positive { background: var(--rl-pos-bg); border-color: var(--rl-pos); }
.rl-verdict.negative { background: var(--rl-neg-bg); border-color: var(--rl-neg); }
.rl-verdict.uncertain { background: var(--rl-unc-bg); border-color: var(--rl-unc); }
.rl-verdict .emoji { flex: none; font-size: 46px; line-height: 1; }
.rl-verdict h2 {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 28px;
    font-weight: 600;
    margin: 0 0 4px;
    color: var(--rl-ink) !important;
}
.rl-verdict.positive h2 { color: var(--rl-pos) !important; }
.rl-verdict.negative h2 { color: var(--rl-neg) !important; }
.rl-verdict.uncertain h2 { color: var(--rl-unc) !important; }
.rl-verdict p { margin: 0; line-height: 1.55; font-size: 14.5px; color: var(--rl-ink) !important; }

/* ---- metric boxes ---- */
.rl-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 10px; margin-top: 14px; }
.rl-metric { background: var(--rl-soft); border: 1px solid var(--rl-line); border-radius: 12px; padding: 10px 12px; }
.rl-metric span { display: block; font-size: 12.5px; color: var(--rl-muted); }
.rl-metric b { display: block; font-size: 20px; margin-top: 2px; font-variant-numeric: tabular-nums; color: var(--rl-ink); }
.rl-metric.pos b { color: var(--rl-pos); }
.rl-metric.neg b { color: var(--rl-neg); }
.rl-metric.unc b { color: var(--rl-unc); }

/* ---- content boxes ---- */
.rl-box { margin-top: 14px; padding: 14px 16px; border: 1px solid var(--rl-line); border-radius: 12px; background: var(--rl-surface); }
.rl-box h3 { font-size: 14.5px; font-weight: 700; margin: 0 0 4px; color: var(--rl-ink) !important; }
.rl-hint { color: var(--rl-muted); font-size: 12.5px; margin: 0 0 10px; line-height: 1.5; }
.rl-two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }

/* ---- gauge ---- */
.rl-gauge-track {
    position: relative;
    height: 18px;
    border-radius: 999px;
    overflow: hidden;
    background: linear-gradient(90deg, var(--rl-neg-bg) 50%, var(--rl-pos-bg) 50%);
    border: 1px solid var(--rl-line);
}
.rl-gauge-neg, .rl-gauge-pos { position: absolute; top: 0; bottom: 0; }
.rl-gauge-neg { right: 50%; background: var(--rl-neg); opacity: .9; }
.rl-gauge-pos { left: 50%; background: var(--rl-pos); opacity: .9; }
.rl-gauge-zone {
    position: absolute; top: 0; bottom: 0;
    background: repeating-linear-gradient(135deg, transparent 0 4px, rgba(128,128,128,.45) 4px 6px);
}
.rl-gauge-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: var(--rl-ink); opacity: .55; }
.rl-gauge-labels { display: flex; justify-content: space-between; margin-top: 8px; font-size: 13.5px; font-weight: 600; }
.rl-gauge-labels .neg { color: var(--rl-neg); }
.rl-gauge-labels .pos { color: var(--rl-pos); }

/* ---- highlighted text ---- */
.rl-text {
    background: var(--rl-soft);
    border: 1px solid var(--rl-line);
    border-radius: 10px;
    padding: 14px 16px;
    line-height: 2.1;
    font-size: 16px;
    color: var(--rl-ink);
    overflow-wrap: anywhere;
}
.rl-mark { color: inherit; border-radius: 5px; padding: 2px 4px; }
.rl-mark.pos { background: rgba(var(--rl-pos-rgb), var(--a)); box-shadow: inset 0 -2px 0 rgba(var(--rl-pos-rgb), .9); }
.rl-mark.neg { background: rgba(var(--rl-neg-rgb), var(--a)); box-shadow: inset 0 -2px 0 rgba(var(--rl-neg-rgb), .9); }
.rl-legend { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; font-size: 12.5px; color: var(--rl-muted); }
.rl-legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -2px; }
.rl-legend .p { background: rgba(var(--rl-pos-rgb), .6); }
.rl-legend .n { background: rgba(var(--rl-neg-rgb), .6); }

/* ---- contribution bars ---- */
.rl-bars { display: grid; gap: 6px; }
.rl-bar-row { display: grid; grid-template-columns: minmax(70px, 32%) 1fr 54px; align-items: center; gap: 10px; font-size: 13.5px; color: var(--rl-ink); }
.rl-bar-name { text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rl-bar-area { position: relative; height: 18px; background: var(--rl-soft); border-radius: 5px; }
.rl-bar-area::after { content: ""; position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--rl-line); }
.rl-bar { position: absolute; top: 2px; bottom: 2px; border-radius: 3px; }
.rl-bar.pos { left: 50%; background: var(--rl-pos); }
.rl-bar.neg { right: 50%; background: var(--rl-neg); }
.rl-bar-val { font-variant-numeric: tabular-nums; color: var(--rl-muted); font-size: 12.5px; }

/* ---- history ---- */
.rl-hrow { display: grid; grid-template-columns: 26px 1fr auto; gap: 10px; align-items: center; padding: 7px 0; border-top: 1px solid var(--rl-line); font-size: 13.5px; color: var(--rl-ink); }
.rl-hrow:first-of-type { border-top: none; }
.rl-hrow .t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.rl-hrow .p { color: var(--rl-muted); font-variant-numeric: tabular-nums; font-size: 12.5px; }
.rl-empty { color: var(--rl-muted); font-size: 13.5px; }

/* ---- batch ---- */
.rl-donut-wrap { display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
.rl-donut { width: 150px; height: 150px; flex: none; color: var(--rl-ink); }
.rl-donut .ring { stroke: var(--rl-line); }
.rl-donut .seg.positive { stroke: var(--rl-pos); }
.rl-donut .seg.negative { stroke: var(--rl-neg); }
.rl-donut .seg.uncertain { stroke: var(--rl-unc); }
.rl-donut .seg.unreadable { stroke: var(--rl-muted); }
.rl-donut text { fill: var(--rl-ink); }
.rl-donut .sub { fill: var(--rl-muted); }
.rl-stats { flex: 1; min-width: 200px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.rl-stat { background: var(--rl-soft); border: 1px solid var(--rl-line); border-radius: 12px; padding: 10px 12px; }
.rl-stat b { display: block; font-size: 22px; font-variant-numeric: tabular-nums; color: var(--rl-ink); }
.rl-stat span { font-size: 12.5px; color: var(--rl-muted); }
.rl-stat.positive b { color: var(--rl-pos); }
.rl-stat.negative b { color: var(--rl-neg); }
.rl-stat.uncertain b { color: var(--rl-unc); }
.rl-hist { display: flex; align-items: flex-end; gap: 6px; height: 120px; padding-top: 6px; }
.rl-hist-col { flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; font-size: 11.5px; color: var(--rl-muted); }
.rl-hist-bar { width: 100%; border-radius: 4px 4px 0 0; }
.rl-hist-bar.pos { background: var(--rl-pos); }
.rl-hist-bar.neg { background: var(--rl-neg); }
.rl-hist-bar.unc { background: var(--rl-unc); }
.rl-hist-axis { display: flex; justify-content: space-between; font-size: 12px; color: var(--rl-muted); margin-top: 6px; border-top: 1px solid var(--rl-line); padding-top: 4px; }

/* ---- about ---- */
.rl-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.rl-card { border: 1px solid var(--rl-line); border-radius: 14px; padding: 16px; background: var(--rl-soft); }
.rl-card .ico { font-size: 26px; }
.rl-card b { display: block; font-size: 28px; margin: 6px 0 2px; font-variant-numeric: tabular-nums; color: var(--rl-ink); font-family: 'Newsreader', Georgia, serif; }
.rl-card span { color: var(--rl-muted); font-size: 13px; }
.rl-flow { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; }
.rl-step { flex: 1; min-width: 150px; border: 1px solid var(--rl-line); background: var(--rl-surface); border-radius: 12px; padding: 12px; font-size: 13px; color: var(--rl-muted); line-height: 1.5; }
.rl-step b { display: block; color: var(--rl-ink); font-size: 14.5px; margin-bottom: 2px; }
.rl-arrow { align-self: center; color: var(--rl-muted); font-size: 18px; }

#footer-note { text-align: center; opacity: .75; padding-top: 16px; }

@media (max-width: 640px) {
    #rl-header { flex-direction: column; align-items: flex-start; }
    .rl-verdict h2 { font-size: 24px; }
    .rl-verdict .emoji { font-size: 38px; }
    .rl-two { grid-template-columns: 1fr; }
    .rl-bar-row { grid-template-columns: 80px 1fr 48px; }
    .rl-arrow { display: none; }
}
"""

# ----------------------------------------------------------------- HTML pieces
HEADER = """
<div id="rl-header">
    <div class="logo">🔎</div>
    <div>
        <h1>ReviewLens</h1>
        <p>Paste an English product review and see whether it reads as positive
        or negative, how sure the model is, and which words made the difference.</p>
        <div class="rl-tags">
            <span>🧠 Version 4</span>
            <span>🛒 English product reviews</span>
            <span>👍 Positive or 👎 negative</span>
        </div>
    </div>
</div>
"""

TIPS = """
<div class="rl-note"><div class="ico">💡</div><div><b>For the best result</b>
<ul>
<li>Write full sentences about one product.</li>
<li>Mention what you liked or disliked: sound, battery, quality, price.</li>
<li>Personal updates and other languages are outside the model's scope.</li>
</ul></div></div>
"""

EMPTY_RESULT = """
<div class="rl-verdict">
    <div class="emoji">✍️</div>
    <div><h2>Ready when you are</h2>
    <p>Write or pick a review, then select <b>Analyze review</b>. You will get a
    verdict, a probability chart and the words that drove the result.</p></div>
</div>
"""

EMPTY_BATCH = """
<div class="rl-verdict">
    <div class="emoji">🗂️</div>
    <div><h2>No batch yet</h2>
    <p>Paste one review per line or upload a .csv / .txt file, then select
    <b>Analyze batch</b>. Not sure where to start? Use <b>Load sample reviews</b>.</p></div>
</div>
"""

EMPTY_HISTORY = '<div class="rl-empty">No analyses yet. Your last few results will appear here.</div>'

SAMPLE_BATCH = "\n".join([
    "Great value and fast delivery. I would buy it again.",
    "Stopped working after a week. Complete waste of money.",
    "The sound quality is excellent and the battery lasts all day.",
    "Terrible customer service and the box arrived damaged.",
    "Comfortable to wear and very easy to set up.",
    "The screen is dim and the buttons feel cheap.",
    "It is okay, nothing special.",
    "Absolutely love this blender, it crushes ice perfectly.",
])

EXAMPLES = [
    ["This headset is excellent and works perfectly."],
    ["The charger broke after one day. Terrible quality."],
    ["The sound quality is not good."],
    ["The sound is great, but the battery is terrible."],
    ["It arrived on time, but the screen is dim and the buttons feel cheap."],
]


def pct(value):
    return f"{value * 100:.1f}%"


def esc(text):
    return html.escape(str(text))


def message_box(emoji, title, message):
    return (
        '<div class="rl-verdict uncertain">'
        f'<div class="emoji">{emoji}</div>'
        f"<div><h2>{esc(title)}</h2><p>{esc(message)}</p></div></div>"
    )


def counter_html(text):
    n = len(text or "")
    over = " over" if n > MAX_CHARS else ""
    return f'<div class="rl-counter{over}">{n:,} / {MAX_CHARS:,} characters</div>'


def strength_label(top_score):
    if top_score >= 0.95:
        return "Very strong"
    if top_score >= 0.85:
        return "Strong"
    if top_score >= 0.70:
        return "Moderate"
    return "Weak"


def gauge_html(p_neg, p_pos):
    """Diverging bar: negative grows left from the centre, positive right."""
    half_zone = (UNCERTAINTY_THRESHOLD - 0.5) * 100
    return f"""
<div class="rl-gauge-track" role="img" aria-label="Negative {pct(p_neg)}, positive {pct(p_pos)}">
    <div class="rl-gauge-neg" style="width:{p_neg * 50:.2f}%"></div>
    <div class="rl-gauge-pos" style="width:{p_pos * 50:.2f}%"></div>
    <div class="rl-gauge-zone" style="left:{50 - half_zone:.2f}%;width:{half_zone * 2:.2f}%"></div>
    <div class="rl-gauge-mid"></div>
</div>
<div class="rl-gauge-labels">
    <span class="neg">👎 Negative {pct(p_neg)}</span>
    <span class="pos">👍 Positive {pct(p_pos)}</span>
</div>
"""


def highlight_html(review, by_index):
    """Mark known words green or red; faint contributors stay unmarked."""
    chunks = re.findall(r"\w+|\W+", review)
    values = []
    for chunk in chunks:
        index = vocabulary.get(chunk.lower()) if re.match(r"\w", chunk) else None
        values.append(by_index.get(index) if index is not None else None)

    strongest = max((abs(v) for v in values if v is not None), default=0.0)
    cutoff = strongest * 0.15

    parts = []
    for chunk, value in zip(chunks, values):
        if value is None or strongest == 0 or abs(value) < cutoff:
            parts.append(esc(chunk))
            continue
        side = "pos" if value > 0 else "neg"
        alpha = 0.22 + 0.5 * min(abs(value) / strongest, 1.0)
        label = "pushes positive" if value > 0 else "pushes negative"
        parts.append(
            f'<mark class="rl-mark {side}" style="--a:{alpha:.2f}" '
            f'title="{label} ({value:+.2f})">{esc(chunk)}</mark>'
        )
    return "".join(parts)


def bars_html(items):
    if not items:
        return '<div class="rl-empty">Nothing to show.</div>'
    strongest = max(abs(v) for _, v in items) or 1.0
    rows = []
    for name, value in items:
        width = abs(value) / strongest * 50
        side = "pos" if value > 0 else "neg"
        rows.append(
            '<div class="rl-bar-row">'
            f'<div class="rl-bar-name" title="{esc(name)}">{esc(name)}</div>'
            f'<div class="rl-bar-area"><div class="rl-bar {side}" style="width:{width:.2f}%"></div></div>'
            f'<div class="rl-bar-val">{value:+.2f}</div>'
            "</div>"
        )
    return '<div class="rl-bars">' + "".join(rows) + "</div>"


def donut_svg(segments, total):
    radius = 15.9155  # circumference of 100 units
    parts = ['<circle class="ring" cx="21" cy="21" r="15.9155" fill="transparent" stroke-width="5.5"></circle>']
    cumulative = 0.0
    for key, label, n in segments:
        if not n:
            continue
        share = n / total * 100
        parts.append(
            f'<circle class="seg {key}" cx="21" cy="21" r="{radius}" fill="transparent" '
            f'stroke-width="5.5" stroke-dasharray="{share:.2f} {100 - share:.2f}" '
            f'stroke-dashoffset="{25 - cumulative:.2f}"><title>{label}: {n}</title></circle>'
        )
        cumulative += share
    parts.append(
        f'<text x="21" y="22.5" text-anchor="middle" font-size="8" font-weight="700">{total}</text>'
        '<text class="sub" x="21" y="27.5" text-anchor="middle" font-size="3.2">reviews</text>'
    )
    return (
        '<svg class="rl-donut" viewBox="0 0 42 42" role="img" aria-label="Sentiment distribution">'
        + "".join(parts)
        + "</svg>"
    )


def histogram_html(p_values):
    bins = np.histogram(p_values, bins=10, range=(0.0, 1.0))[0]
    peak = max(int(bins.max()), 1)
    cols = []
    for i, count in enumerate(bins):
        centre = (i + 0.5) / 10
        tone = "neg" if centre < 0.5 - (UNCERTAINTY_THRESHOLD - 0.5) else (
            "pos" if centre > UNCERTAINTY_THRESHOLD else "unc"
        )
        height = 0 if count == 0 else max(4, int(count / peak * 88))
        cols.append(
            f'<div class="rl-hist-col" title="{i * 10}-{(i + 1) * 10}% positive: {int(count)} reviews">'
            f'<span>{int(count) if count else ""}</span>'
            f'<div class="rl-hist-bar {tone}" style="height:{height}px"></div></div>'
        )
    return (
        '<div class="rl-hist">' + "".join(cols) + "</div>"
        '<div class="rl-hist-axis"><span>0% positive</span><span>50%</span><span>100% positive</span></div>'
    )


def history_html(history):
    if not history:
        return EMPTY_HISTORY
    rows = []
    for entry in history:
        rows.append(
            '<div class="rl-hrow">'
            f'<span>{entry["emoji"]}</span>'
            f'<span class="t" title="{esc(entry["text"])}">{esc(entry["text"])}</span>'
            f'<span class="p">{entry["time"]} · 👍 {pct(entry["p_pos"])}</span>'
            "</div>"
        )
    return "".join(rows)


# ------------------------------------------------------------------ prediction
def verdict_for(p_pos, p_neg):
    top = max(p_pos, p_neg)
    leaning = "positive" if p_pos >= p_neg else "negative"
    if top < UNCERTAINTY_THRESHOLD:
        return "uncertain", leaning, top
    return leaning, leaning, top


def explain(review):
    """Per-feature contributions (linear models only)."""
    if signed_coef is None:
        return {}, []
    row = vectorizer.transform([review]).tocsr()
    contributions = row.data * signed_coef[row.indices]
    by_index = dict(zip(row.indices.tolist(), contributions.tolist()))
    ranked = sorted(by_index.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    top = [(str(feature_names[i]), v) for i, v in ranked]
    return by_index, top


def analyze_review(review, history):
    """Validate input and return (result html, new history, history html)."""
    history = list(history or [])

    def fail(emoji, title, message):
        return message_box(emoji, title, message), history, history_html(history)

    if not isinstance(review, str) or not review.strip():
        return fail("✍️", "Add a review first",
                    "Enter an English product review, then select Analyze review.")

    if len(review) > MAX_CHARS:
        return fail("📏", "Review is too long",
                    f"Shorten it to {MAX_CHARS:,} characters or fewer (it has {len(review):,}).")

    review = " ".join(review.split())

    if vectorizer.transform([review]).nnz == 0:
        return fail("🔤", "No known words found",
                    "Write a product review using words rather than punctuation or numbers "
                    "alone. Text in other languages is not supported.")

    probabilities = model.predict_proba([review])[0]
    p_pos = float(probabilities[POSITIVE_COL])
    p_neg = float(probabilities[NEGATIVE_COL])
    verdict, leaning, top_score = verdict_for(p_pos, p_neg)
    emoji, heading = VERDICTS[verdict]

    if verdict == "uncertain":
        explanation = (
            f"The model leans {leaning} ({pct(top_score)}), but neither side reaches "
            f"{UNCERTAINTY_THRESHOLD:.0%}. This does not mean the review is neutral or mixed."
        )
    else:
        explanation = (
            f"The model gives {pct(top_score)} probability to {verdict} sentiment. "
            "This is an estimate, not a guarantee."
        )

    words = re.findall(r"\w+", review)
    known_words = sum(1 for w in words if w.lower() in vocabulary)
    by_index, top = explain(review)

    # Mixed-signal warning: strong evidence on both sides.
    mixed = ""
    # Single words only: a phrase such as "not good" is negation, not a mix.
    positives = [v for n, v in top if v > 0 and " " not in n]
    negatives = [-v for n, v in top if v < 0 and " " not in n]
    # With negation ("not good") a positive word is usually flipped, so skip.
    negated = re.search(r"\b(not|no|never|cannot)\b|n[’']t\b", review.lower())
    if positives and negatives and not negated:
        strongest = max(max(positives), max(negatives))
        if min(max(positives), max(negatives)) >= max(0.5, 0.4 * strongest):
            mixed = (
                '<div class="rl-note warn" style="margin-top:14px"><div class="ico">⚠️</div>'
                "<div><b>Mixed signals.</b> This review has strong positive and strong negative "
                "wording. The model only has two classes, so it picks the side with more "
                "evidence. It cannot report a mixed or neutral review.</div></div>"
            )

    metrics = f"""
<div class="rl-metrics">
    <div class="rl-metric pos"><span>👍 Positive</span><b>{pct(p_pos)}</b></div>
    <div class="rl-metric neg"><span>👎 Negative</span><b>{pct(p_neg)}</b></div>
    <div class="rl-metric unc"><span>💪 Signal</span><b>{strength_label(top_score)}</b></div>
    <div class="rl-metric"><span>🔤 Known words</span><b>{known_words} / {len(words)}</b></div>
</div>
"""

    explain_blocks = ""
    if signed_coef is not None:
        explain_blocks = f"""
<div class="rl-box">
    <h3>🖍️ Words that shaped the result</h3>
    <p class="rl-hint">Strong words are marked. Hover a mark to see its score.</p>
    <div class="rl-text">{highlight_html(review, by_index)}</div>
    <div class="rl-legend">
        <span><i class="p"></i>pushes positive</span>
        <span><i class="n"></i>pushes negative</span>
    </div>
</div>
<div class="rl-box">
    <h3>📊 Strongest contributions</h3>
    <p class="rl-hint">Single words and two-word phrases, ranked by their effect on the score.</p>
    {bars_html(top)}
</div>
"""

    result = f"""
<div class="rl-verdict {verdict}">
    <div class="emoji">{emoji}</div>
    <div><h2>{heading}</h2><p>{explanation}</p></div>
</div>
{metrics}
{mixed}
<div class="rl-box">
    <h3>📈 Probability estimate</h3>
    <p class="rl-hint">Hatched area: results inside it are shown as uncertain.</p>
    {gauge_html(p_neg, p_pos)}
</div>
{explain_blocks}
"""

    history.insert(0, {
        "emoji": emoji,
        "text": review if len(review) <= 70 else review[:67] + "...",
        "p_pos": p_pos,
        "time": datetime.now().strftime("%H:%M"),
    })
    history = history[:HISTORY_SIZE]
    return result, history, history_html(history)


# ----------------------------------------------------------------------- batch
def read_batch_input(text, file):
    reviews = []
    if text:
        reviews.extend(line.strip() for line in text.splitlines() if line.strip())

    if file is not None:
        path = Path(getattr(file, "name", file))
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            lowered = {str(c).lower(): c for c in frame.columns}
            column = next(
                (lowered[n] for n in ("review", "text", "comment", "content") if n in lowered),
                frame.columns[0],
            )
            reviews.extend(str(v).strip() for v in frame[column].dropna() if str(v).strip())
        else:
            content = path.read_text(encoding="utf-8", errors="ignore")
            reviews.extend(line.strip() for line in content.splitlines() if line.strip())
    return reviews


def batch_error(emoji, title, message):
    return (
        message_box(emoji, title, message),
        gr.Dataframe(visible=False),
        gr.File(visible=False),
    )


def analyze_batch(text, file):
    try:
        reviews = read_batch_input(text, file)
    except Exception as error:  # unreadable file, bad encoding, empty csv...
        return batch_error("📄", "Could not read the file", str(error))

    if not reviews:
        return batch_error("🗂️", "Nothing to analyze",
                           "Paste one review per line, or upload a .csv or .txt file.")

    note = ""
    if len(reviews) > MAX_BATCH:
        note = f" Only the first {MAX_BATCH} of {len(reviews)} reviews were analyzed."
        reviews = reviews[:MAX_BATCH]

    cleaned = [" ".join(r.split())[:MAX_CHARS] for r in reviews]
    matrix = vectorizer.transform(cleaned).tocsr()
    known = matrix.getnnz(axis=1) > 0
    probabilities = model.predict_proba(cleaned)

    records, p_values = [], []
    for i, review in enumerate(cleaned):
        if not known[i]:
            records.append((i + 1, review, "unreadable", None, None))
            continue
        p_pos = float(probabilities[i][POSITIVE_COL])
        p_neg = float(probabilities[i][NEGATIVE_COL])
        verdict, _, _ = verdict_for(p_pos, p_neg)
        p_values.append(p_pos)
        records.append((i + 1, review, verdict, p_pos, p_neg))

    frame = pd.DataFrame(records, columns=["#", "Review", "key", "P(positive)", "P(negative)"])
    total = len(frame)
    counts = frame["key"].value_counts()
    segments = [
        ("positive", "Positive", int(counts.get("positive", 0))),
        ("negative", "Negative", int(counts.get("negative", 0))),
        ("uncertain", "Uncertain", int(counts.get("uncertain", 0))),
        ("unreadable", "No known words", int(counts.get("unreadable", 0))),
    ]
    icons = {"positive": "👍", "negative": "👎", "uncertain": "⚖️", "unreadable": "🔤"}

    stats = "".join(
        f'<div class="rl-stat {key}"><b>{icons[key]} {n}</b><span>{label} ({n / total:.0%})</span></div>'
        for key, label, n in segments
    )
    mean_text = pct(float(np.mean(p_values))) if p_values else "n/a"

    histogram = ""
    if p_values:
        histogram = f"""
<div class="rl-box">
    <h3>📶 How positive is each review?</h3>
    <p class="rl-hint">Each bar counts reviews in a 10% range of positive probability.</p>
    {histogram_html(p_values)}
</div>
"""

    words_block = ""
    if signed_coef is not None and known.any():
        scores = np.asarray(matrix[known].multiply(signed_coef).sum(axis=0)).ravel()
        order = np.argsort(scores)
        top_pos = [(str(feature_names[i]), float(scores[i])) for i in order[::-1][:6] if scores[i] > 0]
        top_neg = [(str(feature_names[i]), float(scores[i])) for i in order[:6] if scores[i] < 0]
        words_block = f"""
<div class="rl-box">
    <h3>🔑 Words that mattered most in this batch</h3>
    <p class="rl-hint">Total push across all reviews (word weight added up per review).</p>
    <div class="rl-two">
        <div><div class="rl-hint">👍 Pushing positive</div>{bars_html(top_pos)}</div>
        <div><div class="rl-hint">👎 Pushing negative</div>{bars_html(top_neg)}</div>
    </div>
</div>
"""

    plural = "s" if total != 1 else ""
    summary = f"""
<div class="rl-verdict">
    <div class="emoji">📊</div>
    <div><h2>{total} review{plural} analyzed</h2>
    <p>Average positive probability: <b>{mean_text}</b>.{esc(note)}</p></div>
</div>
<div class="rl-box">
    <h3>🍩 Sentiment distribution</h3>
    <div class="rl-donut-wrap">{donut_svg(segments, total)}<div class="rl-stats">{stats}</div></div>
</div>
{histogram}
{words_block}
"""

    display = pd.DataFrame({
        "#": frame["#"],
        "Review": frame["Review"].map(lambda t: t if len(t) <= 110 else t[:107] + "..."),
        "Verdict": frame["key"].map(TABLE_LABELS),
        "P(positive)": frame["P(positive)"].map(lambda v: "" if pd.isna(v) else pct(v)),
        "P(negative)": frame["P(negative)"].map(lambda v: "" if pd.isna(v) else pct(v)),
    })

    export = pd.DataFrame({
        "#": frame["#"],
        "Review": frame["Review"],
        "Verdict": frame["key"].map(lambda k: TABLE_LABELS[k].split(" ", 1)[1]),
        "P(positive)": frame["P(positive)"],
        "P(negative)": frame["P(negative)"],
    })
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="reviewlens_", delete=False,
        encoding="utf-8-sig", newline="",
    )
    export.to_csv(handle, index=False)
    handle.close()

    return (
        summary,
        gr.Dataframe(value=display, visible=True),
        gr.File(value=handle.name, visible=True),
    )


def load_sample():
    return SAMPLE_BATCH


def clear_batch():
    return "", None, EMPTY_BATCH, gr.Dataframe(value=None, visible=False), gr.File(value=None, visible=False)


def clear_review():
    return "", EMPTY_RESULT, counter_html("")


def clear_history():
    return [], EMPTY_HISTORY


# ----------------------------------------------------------------- about page
ABOUT = """
<div class="rl-cards">
    <div class="rl-card"><div class="ico">🎯</div><b>87.56%</b><span>Test accuracy on 5,000 Amazon review texts</span></div>
    <div class="rl-card"><div class="ico">⚖️</div><b>0.8754</b><span>Macro F1 score on the same test sample</span></div>
    <div class="rl-card"><div class="ico">📚</div><b>19,993</b><span>Review texts used for training</span></div>
    <div class="rl-card"><div class="ico">🏷️</div><b>2 classes</b><span>Positive or negative only</span></div>
</div>

<div class="rl-box">
    <h3>🧩 How the model works</h3>
    <p class="rl-hint">Four steps from your text to a probability.</p>
    <div class="rl-flow">
        <div class="rl-step"><b>✍️ Review text</b>Your sentence, cleaned of extra spaces.</div>
        <div class="rl-arrow">➜</div>
        <div class="rl-step"><b>🔡 TF-IDF</b>Turns single words and two-word phrases into weighted numbers.</div>
        <div class="rl-arrow">➜</div>
        <div class="rl-step"><b>📐 Logistic Regression</b>Adds up each feature's weight for positive or negative.</div>
        <div class="rl-arrow">➜</div>
        <div class="rl-step"><b>📈 Probabilities</b>Two numbers that sum to 100%.</div>
    </div>
</div>

<div class="rl-box">
    <h3>🧭 How to read the results</h3>
    <div class="rl-note" style="margin-bottom:8px"><div class="ico">📈</div><div><b>Probability gauge.</b> The model's estimate for each class. If neither reaches 60%, the result is shown as uncertain. This is a display rule, not a validated confidence threshold.</div></div>
    <div class="rl-note" style="margin-bottom:8px"><div class="ico">🖍️</div><div><b>Highlighted words.</b> Green words raise the positive score, red words raise the negative score. Faint contributors are left unmarked, and words the model never saw in training are not marked.</div></div>
    <div class="rl-note"><div class="ico">📊</div><div><b>Contribution bars.</b> The same scores for single words and word pairs, sorted by size.</div></div>
</div>

<div class="rl-box">
    <h3>⚠️ Limits</h3>
    <div class="rl-note warn"><div class="ico">🚧</div><div>The model predicts only <b>positive</b> or <b>negative</b>. Neutral statements, mixed opinions, sarcasm and unrelated text may be misclassified. Probability estimates are not calibrated guarantees of correctness. Exact normalized overlaps with previously used data were excluded from the test sample.</div></div>
</div>
"""


# ------------------------------------------------------------------------- UI
def build_demo():
    theme = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
        neutral_hue="slate",
    )  # fonts are set in CSS (Public Sans / Newsreader with fallbacks)

    styling = {"theme": theme, "css": CSS}
    styling_at_launch = "theme" in inspect.signature(gr.Blocks.launch).parameters
    block_options = {} if styling_at_launch else styling

    with gr.Blocks(title="ReviewLens | Product Review Analyzer", **block_options) as demo:
        history_state = gr.State([])
        gr.HTML(HEADER)

        with gr.Tabs():
            # ---------------------------------------------------- single review
            with gr.Tab("✍️ Single review"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, elem_classes=["rl-panel"]):
                        gr.Markdown("### ✍️ Write your review")
                        review = gr.Textbox(
                            label="English product review",
                            placeholder=(
                                "For example: These headphones are comfortable "
                                "and the sound is excellent..."
                            ),
                            lines=8,
                            elem_id="review-input",
                        )
                        counter = gr.HTML(counter_html(""))

                        with gr.Row():
                            analyze = gr.Button(
                                "🔍 Analyze review",
                                variant="primary",
                                scale=3,
                                elem_id="analyze-button",
                            )
                            clear = gr.Button("🧹 Clear", scale=1)

                        gr.Examples(
                            examples=EXAMPLES,
                            inputs=review,
                            label="🧪 Try an example",
                        )
                        gr.HTML(TIPS)

                        gr.Markdown("### 🕘 Recent analyses")
                        history_box = gr.HTML(EMPTY_HISTORY)
                        clear_hist = gr.Button("Clear history", size="sm")

                    with gr.Column(scale=6, elem_classes=["rl-panel"]):
                        gr.Markdown("### 📋 Your result")
                        result = gr.HTML(EMPTY_RESULT)

            # ---------------------------------------------------------- batch
            with gr.Tab("📊 Batch analysis"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, elem_classes=["rl-panel"]):
                        gr.Markdown("### 🗂️ Add many reviews")
                        gr.HTML(
                            '<div class="rl-note"><div class="ico">ℹ️</div><div>'
                            f"One review per line, up to {MAX_BATCH}. For a .csv, use a column named "
                            "<b>review</b> or <b>text</b> (otherwise the first column is used)."
                            "</div></div>"
                        )
                        batch_text = gr.Textbox(
                            label="Reviews, one per line",
                            lines=8,
                            placeholder="Great value and fast delivery.\nStopped working after a week.",
                        )
                        batch_file = gr.File(
                            label="📎 Or upload a file (.csv or .txt)",
                            file_types=[".csv", ".txt"],
                        )
                        sample_button = gr.Button("🧪 Load sample reviews", size="sm")
                        with gr.Row():
                            batch_button = gr.Button(
                                "📊 Analyze batch",
                                variant="primary",
                                scale=3,
                                elem_id="batch-button",
                            )
                            batch_clear = gr.Button("🧹 Clear", scale=1)

                    with gr.Column(scale=6, elem_classes=["rl-panel"]):
                        gr.Markdown("### 📋 Batch summary")
                        batch_summary = gr.HTML(EMPTY_BATCH)
                        batch_table = gr.Dataframe(
                            label="🧾 Results",
                            headers=["#", "Review", "Verdict", "P(positive)", "P(negative)"],
                            interactive=False,
                            wrap=True,
                            visible=False,
                        )
                        batch_download = gr.File(
                            label="⬇️ Download full results (CSV)",
                            interactive=False,
                            visible=False,
                        )

            # ---------------------------------------------------------- about
            with gr.Tab("🧠 About the model"):
                with gr.Column(elem_classes=["rl-panel"]):
                    gr.HTML(ABOUT)

        gr.Markdown(
            "Created by Mohammad Alizada · Professional Training · "
            "Yunnan University",
            elem_id="footer-note",
        )

        # ------------------------------------------------------------ events
        single_io = dict(
            fn=analyze_review,
            inputs=[review, history_state],
            outputs=[result, history_state, history_box],
        )
        analyze.click(**single_io)
        review.submit(**single_io)

        clear.click(clear_review, inputs=[], outputs=[review, result, counter])
        clear_hist.click(clear_history, inputs=[], outputs=[history_state, history_box])

        # Remove stale results whenever the review changes.
        review.change(
            lambda text: (counter_html(text), EMPTY_RESULT),
            inputs=review,
            outputs=[counter, result],
            queue=False,
        )

        sample_button.click(load_sample, inputs=[], outputs=batch_text)
        batch_button.click(
            analyze_batch,
            inputs=[batch_text, batch_file],
            outputs=[batch_summary, batch_table, batch_download],
        )
        batch_clear.click(
            clear_batch,
            inputs=[],
            outputs=[batch_text, batch_file, batch_summary, batch_table, batch_download],
        )

    launch_options = styling if styling_at_launch else {}
    return demo, launch_options


if __name__ == "__main__":
    demo, launch_options = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        **launch_options,
    )
