"""SmartSort AI: upload a photo of one waste item and get its material, the right bin and a heatmap.

Run from the ans082 folder:
    python app/app.py
and open http://127.0.0.1:7860 (the page also opens by itself).

This file only builds the web page. The model work is in src/predict.py.
"""
import html
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")      # no network calls from the page itself
for _name in ("NO_PROXY", "no_proxy"):                          # a VPN or proxy must not catch local traffic
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1,localhost"]))

import gradio as gr  # noqa: E402

from predict import ImageProblem, ModelLoadError, load_predictor  # noqa: E402

logger = logging.getLogger("smartsort.app")

EXAMPLE_DIR = ROOT / "app" / "examples"
THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="sky", neutral_hue="slate")
GRADIO_6_OR_NEWER = int(gr.__version__.split(".")[0]) >= 6
NEUTRAL = "#64748B"        # blue-grey, used for "not sure" and problem messages

ABOUT = """
**What it does.** The photo is cut into 14 x 14 small patches. A Vision Transformer (ViT-B/16, fine-tuned
on about 10,800 training photos from a public waste dataset) compares the patches and gives a probability for each of 10 classes.
The class is then matched to a bin of China's four-bin system.

**The heatmap** (Grad-CAM) shows which parts of the photo changed the answer most (red and yellow = most,
blue = little). It is a rough guide to where the model looked. It does not prove why the model chose a class.

**"Not sure".** If the model is less than 90% sure, it says so instead of guessing.

**Limits.**
- It was trained on web-style photos. On real phone photos with a busy background it makes many more mistakes
  (about 57% correct class, about 78% correct bin in my own test of 214 photos).
- It knows only 10 classes. Medicine, electronics, people, screenshots or several items in one photo are not
  handled well. Photograph one item, in good light, on a plain background.
- Local sorting rules can differ between cities. This is a school project, not official advice.
"""

HEADER = """<div style="background:linear-gradient(90deg,#1E3A8A,#2563EB 60%,#38BDF8);color:white;padding:20px 26px;border-radius:14px">
<div style="font-size:1.9em;font-weight:800">SmartSort AI</div>
<div style="margin-top:4px;font-size:1.05em">Which bin does this go in? Upload a photo of <b>one waste item</b>.</div></div>"""

HEATMAP_HELP = ("**What is this picture?** It shows where the model looked when it chose its answer. "
                "**Red and yellow** = parts of the photo that pushed the answer most. "
                "**Blue** = parts that mattered little.")

WELCOME = """<div style="padding:14px 18px;border-left:10px solid #2563EB;border-radius:8px;background:rgba(37,99,235,0.10)">
Upload a photo of <b>one waste item</b> (or use the camera) and press <b>Classify</b>.</div>"""


def problem_html(message: str) -> str:
    """Friendly box for a photo or model problem."""
    return (f'<div style="padding:14px 18px;border-left:10px solid {NEUTRAL};border-radius:8px;'
            f'background:rgba(128,128,128,0.10)">{html.escape(message)}</div>')


def result_html(analysis) -> str:
    """The answer box: bin and tip when the model is sure, a polite 'not sure' otherwise."""
    advice = analysis.advice
    name = html.escape(analysis.label.capitalize())
    percent = f"{analysis.confidence:.0%}"
    if not advice["confident"]:
        return (f'<div style="padding:14px 18px;border-left:10px solid {NEUTRAL};border-radius:8px;'
                'background:rgba(128,128,128,0.10)">'
                "<div style='font-size:1.25em;font-weight:700'>I'm not sure what this item is.</div>"
                f"<div style='margin-top:6px'>Best guess: {name} ({percent}). "
                "Please take a clearer photo with a plain background and only one item.</div></div>")
    color = html.escape(advice["color"])
    return (f'<div style="padding:14px 18px;border-left:10px solid {color};border-radius:8px;'
            'background:rgba(128,128,128,0.10)">'
            f"<div style='font-size:1.4em;font-weight:700'>{name} &middot; {percent} sure</div>"
            f"<div style='margin-top:8px;font-size:1.2em'>Put it in: "
            f"<b style='color:{color}'>{html.escape(advice['bin'])}</b> ({html.escape(advice['zh'])})</div>"
            f"<div style='margin-top:6px'>{html.escape(advice['tip'])}</div></div>")


_predictor = None


def get_predictor():
    """Load the model on first use and keep it. A failed load is tried again on the next click."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = load_predictor()
        except ModelLoadError as problem:
            logger.error("%s | technical reason: %r", problem, problem.__cause__)
            raise
    return _predictor


def classify(image):
    """Button action. Returns (answer box, class scores, heatmap); never raises for bad input."""
    try:
        analysis = get_predictor().analyse(image)
    except (ImageProblem, ModelLoadError) as problem:
        return problem_html(str(problem)), None, None
    except Exception:
        logger.exception("Unexpected error while classifying")
        return problem_html("Something went wrong while looking at this photo. "
                            "Please try another photo."), None, None
    return result_html(analysis), analysis.probabilities, analysis.overlay


def build_demo():
    """Create the web page."""
    examples = [[str(path)] for path in sorted(EXAMPLE_DIR.glob("*.jpg"))]
    blocks_options = {} if GRADIO_6_OR_NEWER else {"theme": THEME}
    with gr.Blocks(title="SmartSort AI", **blocks_options) as demo:
        gr.HTML(HEADER)
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="filepath", sources=["upload", "webcam"], label="Photo of one item",
                                 height=360)
                button = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=image, label="Example photos (click one)")
            with gr.Column():
                answer = gr.HTML(value=WELCOME)
                scores = gr.Label(num_top_classes=3, label="Top 3 guesses")
                heatmap = gr.Image(label="Where the model looked (red = most important)", interactive=False,
                                   height=300)
                gr.Markdown(HEATMAP_HELP)
        with gr.Accordion("How it works and its limits", open=False):
            gr.Markdown(ABOUT)
        button.click(classify, inputs=image, outputs=[answer, scores, heatmap])
    return demo


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        get_predictor()                      # load now, so the first click is fast
    except ModelLoadError:
        logger.warning("Starting without the model. The page opens and shows the problem when you press Classify.")
    port = int(os.environ.get("SMARTSORT_PORT", "7860"))
    try:
        launch_options = {"theme": THEME} if GRADIO_6_OR_NEWER else {}
        build_demo().launch(server_name="127.0.0.1", server_port=port, inbrowser=True, **launch_options)
    finally:
        if _predictor is not None:
            _predictor.close()               # clean shutdown (no error text when you press Ctrl+C)


if __name__ == "__main__":
    main()