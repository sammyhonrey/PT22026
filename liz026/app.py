#######################################################
##                                                   ##
##                  Mid-Term Project                 ##
##          Course: Professional Training (2)        ##
##              Instructor: Ahmed Zahir              ##
##    Gradio App Project: ChestCAM: Explainable      ##
##       Pneumonia Detection from Chest X-Rays       ##
##              using CNN and Grad-CAM               ##
##             Name: LIZA MST SADIA AFRIN            ##
##               Student ID:20243120026              ##
##                                                   ##
#######################################################

import html
import logging

from src import config  # must come first: selects the Keras backend
import gradio as gr

from src.model_loader import get_model
from src.predictor import analyze
from src.preprocessing import InvalidImageError
from src.report import build_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

HISTORY_HEADERS = ["#", "Patient", "Result", "Confidence", "P(Pneumonia)", "Threshold"]

# --------------------------------------------------------------------------- #
#  Styling                                                                    #
# --------------------------------------------------------------------------- #
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="sky",
    neutral_hue="slate",
    radius_size="lg",
    font=["Inter", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
)

CSS = """
.gradio-container {max-width: 1220px !important; margin: auto !important;}

/* ---------- hero header ---------- */
#hero {
  background: linear-gradient(135deg, #312e81 0%, #4338ca 45%, #0ea5e9 100%);
  border-radius: 20px; padding: 26px 30px 24px; color: #fff; text-align: center;
  box-shadow: 0 10px 30px rgba(67, 56, 202, .25);
}
#hero, #hero p, #hero h1, #hero span {color: #fff !important;}
#hero .course {font-size: 15px; opacity: .85; margin: 0 0 10px; line-height: 1.6;}
#hero h1 {font-size: 42px; margin: 0; letter-spacing: -.5px; color: #fff;}
#hero .tagline {font-size: 20px; margin: 6px 0 14px; opacity: .95;}
#hero .chips {display: flex; gap: 10px; justify-content: center; flex-wrap: wrap;}
#hero .chip {background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3);
  padding: 5px 14px; border-radius: 999px; font-size: 14px;}

/* ---------- cards ---------- */
.card {border: 1px solid var(--border-color-primary); border-radius: 18px !important;
  padding: 18px !important; background: var(--background-fill-primary);
  box-shadow: 0 2px 12px rgba(15, 23, 42, .05);}
.step-title {font-size: 18px; font-weight: 700; margin: 0 0 4px;}
.step-title .num {display: inline-flex; width: 28px; height: 28px; border-radius: 50%;
  background: #4338ca; color: #fff; align-items: center; justify-content: center;
  font-size: 14px; margin-right: 8px;}
.step-sub {font-size: 14px; color: var(--body-text-color-subdued); margin: 0 0 6px;}
#analyze-btn {font-size: 17px; font-weight: 700; min-height: 50px;}

/* ---------- result card ---------- */
.result {border-radius: 16px; padding: 18px 20px; border: 2px solid;}
.result.normal    {border-color: #16a34a; background: rgba(22,163,74,.08);}
.result.pneumonia {border-color: #dc2626; background: rgba(220,38,38,.08);}
.result .row {display: flex; align-items: center; gap: 14px;}
.result .icon {font-size: 40px; line-height: 1;}
.result .label {font-size: 30px; font-weight: 800; margin: 0;}
.result.normal .label {color: #16a34a;}
.result.pneumonia .label {color: #dc2626;}
.result .sub {font-size: 14px; color: var(--body-text-color-subdued); margin: 2px 0 0;}
.bar {height: 12px; border-radius: 999px; background: rgba(148,163,184,.3);
  overflow: hidden; margin: 14px 0 6px;}
.bar > div {height: 100%; border-radius: 999px;}
.result.normal .bar > div {background: linear-gradient(90deg, #4ade80, #16a34a);}
.result.pneumonia .bar > div {background: linear-gradient(90deg, #f87171, #dc2626);}
.meta {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;}
.meta span {font-size: 13px; padding: 4px 10px; border-radius: 999px;
  background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary);}
.warn {margin-top: 12px; padding: 10px 12px; border-radius: 10px; font-size: 14px;
  background: rgba(245,158,11,.12); border-left: 4px solid #f59e0b;}
.empty {border: 2px dashed var(--border-color-primary); border-radius: 16px;
  padding: 34px 20px; text-align: center; color: var(--body-text-color-subdued);}
.empty .big {font-size: 44px;}

/* ---------- comparison slider: show the whole X-ray, never crop ---------- */
#compare img {object-fit: contain !important;}

/* ---------- heat-map legend ---------- */
.legend {display: flex; align-items: center; gap: 10px; font-size: 13px;
  color: var(--body-text-color-subdued); margin-top: 6px;}
.legend .grad {flex: 1; height: 10px; border-radius: 999px;
  background: linear-gradient(90deg, #00007f, #0000ff, #00ffff, #ffff00, #ff0000, #7f0000);}

/* ---------- footer ---------- */
#footer {text-align: center; font-size: 13px; color: var(--body-text-color-subdued);
  padding: 10px; border-top: 1px solid var(--border-color-primary); margin-top: 8px;}
"""

HERO_HTML = """
<div id="hero">
  <p class="course">
    Mid-Term Project &nbsp;|&nbsp; Course: Professional Training (2) &nbsp;|&nbsp; Instructor: Ahmed Zahir<br>
    Name: LIZA MST SADIA AFRIN &nbsp;|&nbsp; Student ID: 20243120026
  </p>
  <h1>🩻 ChestCAM</h1>
  <p class="tagline">Explainable Pneumonia Detection from Chest X-Rays using CNN and Grad-CAM</p>
  <div class="chips">
    <span class="chip">🧠 MobileNetV2 CNN</span>
    <span class="chip">🔥 Grad-CAM explanations</span>
    <span class="chip">⚡ Results in &lt; 1 second</span>
    <span class="chip">📄 Downloadable report</span>
  </div>
</div>
"""

EMPTY_RESULT_HTML = """
<div class="empty">
  <div class="big">🫁</div>
  <p style="font-size:17px;margin:6px 0 2px;"><b>No X-ray analyzed yet</b></p>
  <p style="margin:0;">Upload a chest X-ray or click an example on the left.<br>
  The analysis starts automatically.</p>
</div>
"""

LEGEND_HTML = """
<div class="legend"><span>Low influence</span><div class="grad"></div><span>High influence</span></div>
<p class="step-sub" style="margin-top:6px;">Drag the slider to compare the original X-ray with the
Grad-CAM heat-map. Red areas influenced the decision most.</p>
"""


# --------------------------------------------------------------------------- #
#  Callbacks                                                                  #
# --------------------------------------------------------------------------- #
def _result_html(pred) -> str:
    """Colored result card shown after each analysis."""
    is_pn = pred.label == "Pneumonia"
    css_class = "pneumonia" if is_pn else "normal"
    icon = "⚠️" if is_pn else "✅"
    advice = ("Signs consistent with pneumonia were detected. Please consult a doctor."
              if is_pn else "No signs of pneumonia were detected by the model.")
    conf = pred.confidence * 100
    warnings = "".join(f'<div class="warn">⚠️ {html.escape(w)}</div>' for w in pred.warnings)
    return f"""
<div class="result {css_class}">
  <div class="row">
    <div class="icon">{icon}</div>
    <div>
      <p class="label">{pred.label}</p>
      <p class="sub">{advice}</p>
    </div>
  </div>
  <div class="bar"><div style="width:{conf:.1f}%"></div></div>
  <div style="display:flex;justify-content:space-between;font-size:14px;">
    <span><b>Confidence</b></span><span><b>{conf:.1f}%</b></span>
  </div>
  <div class="meta">
    <span>P(Pneumonia): {pred.prob_pneumonia:.3f}</span>
    <span>Threshold: {pred.threshold:.2f}</span>
    <span>⏱ {pred.seconds:.2f} s</span>
  </div>
  {warnings}
</div>"""


def run_analysis(image, patient_name, threshold, history):
    """Analyze one image. Returns result card, probabilities, comparison, report, history."""
    history = history or []
    if image is None:
        # Image was removed: reset outputs quietly instead of showing an error.
        return EMPTY_RESULT_HTML, None, None, None, history, history
    try:
        pred = analyze(image, threshold)
    except InvalidImageError as exc:
        raise gr.Error(str(exc))
    except Exception as exc:  # model missing, download failure, etc.
        logger.exception("Analysis failed")
        raise gr.Error(f"Analysis failed: {exc}")

    report_path = build_report(patient_name or "", pred.label, pred.prob_pneumonia,
                               threshold, pred.resized, pred.heatmap, pred.warnings)
    history = history + [[len(history) + 1, patient_name or "Anonymous", pred.label,
                          f"{pred.confidence*100:.1f}%", round(pred.prob_pneumonia, 4),
                          threshold]]
    return (_result_html(pred), pred.probabilities, (pred.resized, pred.heatmap),
            report_path, history, history)


def clear_all():
    return None, "", config.DEFAULT_THRESHOLD, EMPTY_RESULT_HTML, None, None, None


def clear_history():
    return [], []


ABOUT_MD = """
### 🧠 Model
* **Architecture:** MobileNetV2 (ImageNet-pretrained) backbone → Global Average Pooling →
  BatchNorm → Dropout(0.45) → Dense(ReLU) → Dropout(0.25) → Dense(ReLU) → Dense(1, sigmoid).
* **Task:** binary classification, *Normal* vs *Pneumonia*.
* **Training data:** Kaggle *Chest X-Ray Images (Pneumonia)* (Kermany et al., 2018), pediatric
  chest X-rays from Guangzhou Women and Children's Medical Center.
* **Weights:** pretrained model from the open-source project
  [Xray-Analysis-using-CNN-and-gradio-deployment](https://github.com/Youssefx64/Xray-Analysis-using-CNN-and-gradio-deployment).

### 🔥 How Grad-CAM works
1. The X-ray passes through MobileNetV2, producing 1,280 feature maps of size 7 × 7.
2. We compute how much the pneumonia score changes with each feature map (its gradient).
3. Feature maps are weighted by these gradients and summed → a 7 × 7 relevance map.
4. The map is enlarged and colored over the X-ray: **red = most influential**.

### 🎚️ Decision threshold
The model outputs P(Pneumonia) between 0 and 1. If it is ≥ the threshold, the result is
*Pneumonia*. **Lower** the threshold to catch more cases (higher sensitivity, more false alarms);
**raise** it to reduce false alarms (higher specificity, more missed cases).

### ⚠️ Limitations
Trained on one hospital's pediatric images; performance on adult or other-scanner images is
unknown. It cannot tell bacterial from viral pneumonia, nor detect other diseases.
Grad-CAM is a coarse 7 × 7 explanation, and it sometimes highlights shoulders or text markers,
showing the model may partly rely on non-lung cues. **Not a medical device – educational use only.**
"""


# --------------------------------------------------------------------------- #
#  Layout                                                                     #
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    examples = sorted(str(p) for p in config.EXAMPLES_DIR.glob("*.jp*g"))

    with gr.Blocks(title="ChestCAM – Pneumonia Detection") as demo:
        history_state = gr.State([])

        gr.HTML(HERO_HTML)

        with gr.Tabs():
            # ============================ Analyze tab ============================
            with gr.Tab("🔍 Analyze"):
                with gr.Row(equal_height=False):
                    # ---------------- left: input ----------------
                    with gr.Column(scale=5, elem_classes="card"):
                        gr.HTML('<p class="step-title"><span class="num">1</span>Upload a chest X-ray</p>'
                                '<p class="step-sub">Frontal (PA/AP) view, JPG or PNG. '
                                'Analysis starts automatically.</p>')
                        image_in = gr.Image(type="pil", image_mode=None, height=330,
                                            sources=["upload", "clipboard"], show_label=False,
                                            placeholder="⬆️ Drop an X-ray here or click to browse")
                        if examples:
                            gr.Examples(examples=examples, inputs=image_in,
                                        label="No X-ray at hand? Try an example:")
                        patient_in = gr.Textbox(label="👤 Patient name (optional, for the report)",
                                                placeholder="e.g. John Doe")
                        with gr.Accordion("⚙️ Advanced settings", open=False):
                            threshold_in = gr.Slider(
                                0.05, 0.99, value=config.DEFAULT_THRESHOLD, step=0.01,
                                label="Decision threshold",
                                info="Lower = catch more pneumonia cases · Higher = fewer false alarms "
                                     "(0.99 gave the best balance on the Kaggle test set)")
                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                            analyze_btn = gr.Button("🔍 Analyze X-ray", variant="primary",
                                                    scale=2, elem_id="analyze-btn")

                    # ---------------- right: output ----------------
                    with gr.Column(scale=6, elem_classes="card"):
                        gr.HTML('<p class="step-title"><span class="num">2</span>Result</p>')
                        result_out = gr.HTML(EMPTY_RESULT_HTML)

                        with gr.Tabs():
                            with gr.Tab("🔥 Heat-map"):
                                compare_out = gr.ImageSlider(type="pil", show_label=False,
                                                             height=420, max_height=420, slider_position=50,
                                                             interactive=False, elem_id="compare")
                                gr.HTML(LEGEND_HTML)
                            with gr.Tab("📊 Probabilities"):
                                probs_out = gr.Label(num_top_classes=2, show_label=False)
                            with gr.Tab("📄 Report"):
                                gr.HTML('<p class="step-sub">A self-contained HTML report with '
                                        'the X-ray, heat-map and result. Open it in any browser '
                                        'or print it to PDF.</p>')
                                report_out = gr.File(label="Download report")

            # ============================ History tab ============================
            with gr.Tab("🕘 History"):
                gr.HTML('<p class="step-sub">All X-rays analyzed in this session.</p>')
                history_out = gr.Dataframe(headers=HISTORY_HEADERS, interactive=False,
                                           wrap=True)
                clear_hist_btn = gr.Button("Clear history", size="sm")

            # ============================ About tab ==============================
            with gr.Tab("ℹ️ About"):
                gr.Markdown(ABOUT_MD)

        gr.HTML('<div id="footer">⚠️ For educational purposes only – not a substitute for '
                'professional medical diagnosis. · Built with Keras, PyTorch and Gradio.</div>')

        # ------------------------------ events ------------------------------
        inputs = [image_in, patient_in, threshold_in, history_state]
        outputs = [result_out, probs_out, compare_out, report_out, history_out, history_state]

        analyze_btn.click(run_analysis, inputs, outputs)
        image_in.change(run_analysis, inputs, outputs)      # auto-run on upload / example / paste
        threshold_in.release(run_analysis, inputs, outputs)  # re-run when threshold changes
        clear_btn.click(clear_all, outputs=[image_in, patient_in, threshold_in, result_out,
                                            probs_out, compare_out, report_out])
        clear_hist_btn.click(clear_history, outputs=[history_out, history_state])
    return demo


if __name__ == "__main__":
    get_model()  # load (and download if needed) before the UI starts
    build_ui().launch(server_name=config.SERVER_HOST, server_port=config.SERVER_PORT,
                      theme=THEME, css=CSS, inbrowser=True)
