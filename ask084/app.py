"""Gradio app: NER Studio (analyzer + model arena)."""
import os

if os.environ.get("NER_OFFLINE") == "1":  # set on networks that block Hugging Face
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"

import csv
import tempfile

import gradio as gr

from src.arena import ARENA_CHOICES, SPACY_NAME, compare, to_segments
from src.inference import DEFAULT_MODEL, MODELS, predict

MAX_CHARS = 1500
LABELS = ["PER", "ORG", "LOC", "MISC"]
COLORS = {"PER": "#6366F1", "ORG": "#0EA5E9", "LOC": "#10B981", "MISC": "#F59E0B"}

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=("Segoe UI", "system-ui", "sans-serif"),
)
CSS = """
.app-header {background: linear-gradient(90deg, #1F3864 0%, #4F46E5 100%);
             color: white; padding: 20px 26px; border-radius: 12px; margin-bottom: 8px;}
.app-header h1 {margin: 0 0 4px 0; font-size: 1.9rem; color: white;}
.app-header p {margin: 0; opacity: 0.9; color: white;}
"""

ANALYZER_EXAMPLES = [
    ["Elon Musk visited Google headquarters in California last Monday.", 0.5, LABELS],
    ["Angela Merkel met Emmanuel Macron in Paris to discuss the European Union budget.", 0.5, LABELS],
    ["Apple released the iPhone in San Francisco, and Tim Cook thanked Microsoft.", 0.5, LABELS],
    ["Jordan scored 40 points for the Chicago Bulls against the Lakers.", 0.5, LABELS],
]
ARENA_EXAMPLES = [
    ["Elon Musk visited Google headquarters in California last Monday."],
    ["Apple released the iPhone in San Francisco, and Tim Cook thanked Microsoft."],
    ["Washington signed the agreement with Jordan in Paris."],
    ["Jordan scored 40 points for the Chicago Bulls against the Lakers."],
]


def check_text(text):
    if not text or not text.strip():
        raise gr.Error("Please enter some text first.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text is too long ({len(text)} characters). Keep it under {MAX_CHARS}.")


def analyze(text, threshold, selected_labels, model_name):
    check_text(text)
    entities = [e for e in predict(text, model_name)
                if e["score"] >= threshold and e["label"] in selected_labels]
    entities.sort(key=lambda e: e["start"])
    segments = to_segments(text, entities)

    rows = [[e["text"], e["label"], round(e["score"], 3)] for e in entities]
    if rows:
        counts = {l: sum(1 for e in entities if e["label"] == l) for l in LABELS}
        parts = [f"{n} {l}" for l, n in counts.items() if n]
        summary = f"**Found {len(rows)} entities:** " + ", ".join(parts)
    else:
        summary = "**No entities found** with the current settings. Try lowering the confidence threshold."

    csv_path = None
    if rows:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8")
        writer = csv.writer(tmp)
        writer.writerow(["entity", "label", "confidence"])
        writer.writerows(rows)
        tmp.close()
        csv_path = tmp.name
    return segments, summary, rows, csv_path


def run_arena(text, name_a, name_b, threshold):
    check_text(text)
    if name_a == name_b:
        raise gr.Error("Please choose two different models.")
    a, b, rows = compare(text, name_a, name_b, threshold)
    agree = sum(1 for r in rows if r[3] == "Agree")
    if rows:
        summary = (f"**{name_a}** vs **{name_b}**: {len(rows)} entity spans compared, "
                   f"{agree} agreed, {len(rows) - agree} disagreements.")
    else:
        summary = "**Neither model found entities.**"
    return to_segments(text, a), to_segments(text, b), summary, rows


with gr.Blocks(title="NER Studio") as demo:
    gr.HTML("<div class='app-header'><h1>NER Studio</h1>"
            "<p>Named Entity Recognition for English text: compare pre-trained, fine-tuned and spaCy models.</p></div>")

    with gr.Tab("Analyzer"):
        with gr.Row():
            with gr.Column(scale=1):
                text_in = gr.Textbox(label="Input text", lines=7, placeholder="Paste or type English text here...")
                model_dd = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
                threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence threshold")
                labels = gr.CheckboxGroup(LABELS, value=LABELS, label="Entity types to show")
                run_btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                highlighted = gr.HighlightedText(label="Detected entities", color_map=COLORS, show_legend=True)
                summary_md = gr.Markdown()
                table = gr.Dataframe(headers=["Entity", "Type", "Confidence"], interactive=False, label="Entity table")
                csv_file = gr.File(label="Download results (CSV)")
        gr.Examples(examples=ANALYZER_EXAMPLES, inputs=[text_in, threshold, labels], label="Try an example")
        run_btn.click(analyze, inputs=[text_in, threshold, labels, model_dd],
                      outputs=[highlighted, summary_md, table, csv_file])

    with gr.Tab("Model Arena"):
        gr.Markdown("**Same text, two models.** Pick any two models. Disagreements are classified as label "
                    "conflicts, boundary conflicts, or entities found by only one model.")
        arena_in = gr.Textbox(label="Input text", lines=4, placeholder="Paste or type English text here...")
        with gr.Row():
            model_a = gr.Dropdown(ARENA_CHOICES, value=DEFAULT_MODEL, label="Model A")
            model_b = gr.Dropdown(ARENA_CHOICES, value=SPACY_NAME, label="Model B")
        arena_thr = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="BERT confidence threshold")
        arena_btn = gr.Button("Compare models", variant="primary")
        with gr.Row():
            out_a = gr.HighlightedText(label="Model A", color_map=COLORS, show_legend=True)
            out_b = gr.HighlightedText(label="Model B", color_map=COLORS, show_legend=True)
        arena_summary = gr.Markdown()
        arena_table = gr.Dataframe(headers=["Text", "Model A", "Model B", "Status"], interactive=False,
                                   label="Comparison")
        gr.Examples(examples=ARENA_EXAMPLES, inputs=[arena_in], label="Try an example")
        arena_btn.click(run_arena, inputs=[arena_in, model_a, model_b, arena_thr],
                        outputs=[out_a, out_b, arena_summary, arena_table])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
