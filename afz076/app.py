"""SignGuard - Gradio app: recognise a traffic sign, simulate bad weather, see how sure the model is.

Run:  python app.py
Set SIGNGUARD_CKPTS to a comma-separated list of  label=path  pairs, for example
  SIGNGUARD_CKPTS="Clean=outputs/resnet18_clean/model.pt,Weather-robust=outputs/resnet18_weather/model.pt"
"""
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from src.config import CONDITIONS, DEFAULT_REJECT_THRESHOLD
from src.corruptions import corrupt
from src.predict import SignPredictor

_PREDICTORS = {}   # loaded lazily, one per checkpoint


def _discover_checkpoints() -> dict:
    env = os.getenv("SIGNGUARD_CKPTS", "")
    if env:
        return {k.strip(): v.strip() for k, v in (pair.split("=", 1) for pair in env.split(",") if "=" in pair)}
    return {p.parent.name: str(p) for p in sorted(Path("outputs").glob("*/model.pt"))}


CHECKPOINTS = _discover_checkpoints()


def _get(label: str) -> SignPredictor:
    if label not in _PREDICTORS:
        _PREDICTORS[label] = SignPredictor(CHECKPOINTS[label])
    return _PREDICTORS[label]


def recognise(image, model_label, condition, severity, threshold, show_heatmap):
    """Gradio callback -> (image the model saw, verdict text, top-3 probabilities, heatmap)."""
    if image is None:
        return None, "Please upload or take a photo of a traffic sign.", None, None
    if not CHECKPOINTS:
        return None, "No trained model found. Train one first (see README) or set SIGNGUARD_CKPTS.", None, None
    try:
        img = SignPredictor.prepare(Image.fromarray(image))
        if condition != "none":
            img = corrupt(img, condition, int(severity), np.random.default_rng(0))
        pred = _get(model_label).predict(img, threshold=threshold, with_heatmap=bool(show_heatmap))
    except Exception as exc:                                    # show a friendly message instead of crashing
        return None, f"Something went wrong: {exc}", None, None

    best = pred.top[0]
    if pred.accepted:
        verdict = f"### {best['name']}\nConfidence **{pred.confidence:.0%}**"
    else:
        verdict = (f"### Not sure\nBest guess is *{best['name']}* but confidence is only "
                   f"**{pred.confidence:.0%}** (threshold {threshold:.0%}). "
                   "A safety system should ask for a better image instead of guessing.")
    probs = {t["name"]: t["probability"] for t in pred.top}
    return Image.fromarray(img).resize((256, 256), Image.NEAREST), verdict, probs, pred.heatmap


with gr.Blocks(title="SignGuard") as demo:
    gr.Markdown("# SignGuard\nTraffic sign recognition that tells you **when it is not sure**. "
                "Upload a sign photo, then add fog, night, rain, blur, noise or glare and watch the confidence change.\n\n"
                "*Trained on German signs (GTSRB); educational demo, not for real driving decisions.*")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Traffic sign photo", type="numpy")
            model_dd = gr.Dropdown(list(CHECKPOINTS) or ["(none)"], value=(list(CHECKPOINTS) or ["(none)"])[0],
                                   label="Model")
            cond = gr.Dropdown(["none", *CONDITIONS], value="none", label="Simulate weather / quality problem")
            sev = gr.Slider(1, 5, value=3, step=1, label="Severity")
            thr = gr.Slider(0.3, 0.99, value=DEFAULT_REJECT_THRESHOLD, step=0.01, label="'Not sure' threshold")
            heat = gr.Checkbox(value=True, label="Show Grad-CAM heatmap (what the model looked at)")
            btn = gr.Button("Recognise", variant="primary")
        with gr.Column():
            seen = gr.Image(label="What the model sees (64x64)")
            verdict = gr.Markdown()
            top3 = gr.Label(label="Top 3 predictions", num_top_classes=3)
            heatmap = gr.Image(label="Grad-CAM")
    btn.click(recognise, [inp, model_dd, cond, sev, thr, heat], [seen, verdict, top3, heatmap])

if __name__ == "__main__":
    demo.launch()
