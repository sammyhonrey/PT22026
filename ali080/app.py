import os
import glob
import gradio as gr
from ultralytics import YOLO

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Download best.pt (see README) "
        "and place it in the 'models' folder."
    )

model = YOLO(MODEL_PATH)
CLASS_CHOICES = ["All", "With Helmet", "Without Helmet"]


# ------------------------------------------------------------------
# DESIGN SETTINGS: change only the colour codes in quotes
# ------------------------------------------------------------------
BG_LIGHT = "#fff7ed"          # page background (light mode)
BG_DARK = "#1c1917"           # page background (dark mode)
HEADER_COLOR_1 = "#f97316"    # header gradient: left colour
HEADER_COLOR_2 = "#fbbf24"    # header gradient: right colour

# Main colour scale: labels, slider, icons and buttons all use these shades.
# Lower numbers are lighter, higher numbers are darker.
PRIMARY = gr.themes.Color(
    c50="#fff7ed",
    c100="#ffedd5",
    c200="#fed7aa",
    c300="#fdba74",
    c400="#fb923c",
    c500="#f97316",
    c600="#ea580c",
    c700="#c2410c",
    c800="#9a3412",
    c900="#7c2d12",
    c950="#431407",
    name="custom_primary",
)

THEME = gr.themes.Soft(
    primary_hue=PRIMARY,
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill=BG_LIGHT,
    body_background_fill_dark=BG_DARK,
)

HEADER_GRADIENT = f"linear-gradient(90deg, {HEADER_COLOR_1}, {HEADER_COLOR_2})"


# Helmet drawing shown in the header badge
HELMET_SVG = """
<svg width="60" height="60" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
<path d="M8 40 C8 20 20 10 34 10 C48 10 58 20 58 38 L58 46 C58 48 56 50 54 50 L14 50 C10 50 8 46 8 40 Z" fill="#ffffff"/>
<path d="M15 24 C22 16 34 13 46 17" fill="none" stroke="#f97316" stroke-width="4" stroke-linecap="round"/>
<rect x="30" y="24" width="28" height="13" rx="6" fill="#7c2d12"/>
<rect x="12" y="50" width="42" height="5" rx="2.5" fill="#7c2d12"/>
</svg>
"""

CSS = f"""
.gradio-container {{ max-width: 1150px !important; margin: auto !important; }}
.app-header {{
    background: {HEADER_GRADIENT};
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 12px;
}}
.app-header, .app-header * {{ color: #ffffff !important; }}
.app-header h1 {{ margin: 0 0 6px 0; font-size: 2rem; }}
.app-header p {{ margin: 0; font-size: 1.05rem; }}
.app-footer {{ text-align: center; opacity: 0.75; font-size: 0.9rem; }}
button.primary {{ font-weight: 600; letter-spacing: 0.3px; }}
"""


def make_banner(with_n, without_n, class_filter):
    """Build a coloured status message from the detection counts."""
    if class_filter != "All":
        text = f"Filter active: showing only '{class_filter}' detections."
        bg, fg = "#e2e3e5", "#383d41"
    elif without_n > 0:
        text = f"⚠ Warning: {without_n} rider(s) without a helmet detected."
        bg, fg = "#f8d7da", "#721c24"
    elif with_n > 0:
        text = f"✅ All {with_n} detected rider(s) are wearing a helmet."
        bg, fg = "#d4edda", "#155724"
    else:
        text = "No riders detected. Try lowering the confidence threshold."
        bg, fg = "#e2e3e5", "#383d41"
    return (
        f"<div style='padding:14px;border-radius:10px;font-size:18px;"
        f"font-weight:600;background:{bg};color:{fg};'>{text}</div>"
    )


def detect(image, confidence, class_filter, show_conf):
    """Run YOLOv8 on one image and return banner, annotated image and table."""
    if image is None:
        raise gr.Error("Please upload an image or take a photo first.")

    bgr = image[:, :, ::-1]  # Gradio gives RGB, YOLO expects BGR

    # Optional class filter: keep only the chosen class
    classes = None
    if class_filter != "All":
        classes = [i for i, name in model.names.items() if name == class_filter]

    result = model.predict(
        bgr, conf=confidence, imgsz=640, classes=classes, verbose=False
    )[0]
    annotated = result.plot(conf=show_conf)[:, :, ::-1]  # BGR -> RGB for display

    rows = []
    with_n = without_n = 0
    for box in result.boxes:
        name = model.names[int(box.cls[0])]
        score = float(box.conf[0])
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        rows.append([name, round(score, 2), f"({x1}, {y1}) to ({x2}, {y2})"])
        if name == "With Helmet":
            with_n += 1
        elif name == "Without Helmet":
            without_n += 1

    rows.sort(key=lambda r: r[1], reverse=True)
    return make_banner(with_n, without_n, class_filter), annotated, rows


example_paths = sorted(
    glob.glob(os.path.join(BASE_DIR, "examples", "*.jpg"))
    + glob.glob(os.path.join(BASE_DIR, "examples", "*.png"))
)


def pick_example(evt: gr.SelectData):
    """Return the file path of the thumbnail that was clicked."""
    return example_paths[evt.index]


with gr.Blocks(title="Helmet Detection") as demo:
    gr.HTML(
        "<div style='display:flex;align-items:center;gap:22px;'>"
        "<div style='width:84px;height:84px;border-radius:20px;"
        "background:rgba(255,255,255,0.22);display:flex;align-items:center;"
        "justify-content:center;flex-shrink:0;'>" + HELMET_SVG + "</div>"
        "<div>"
        "<h1>Helmet Detection</h1>"
        "<p>Upload a photo or take one with your webcam. The model finds each "
        "rider and tells you whether they are wearing a helmet.</p>"
        "</div></div>",
        elem_classes=["app-header"],
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="numpy",
                sources=["upload", "webcam", "clipboard"],
                label="Image or webcam photo",
            )
            conf_slider = gr.Slider(0.1, 0.9, value=0.4, step=0.05,
                                    label="Confidence threshold")
            class_filter = gr.Dropdown(CLASS_CHOICES, value="All",
                                       label="Show classes")
            show_conf = gr.Checkbox(value=True, label="Show confidence scores")
            detect_btn = gr.Button("Detect", variant="primary", size="lg")
        with gr.Column():
            banner = gr.HTML()
            output_image = gr.Image(label="Detections")
            table = gr.Dataframe(
                headers=["Class", "Confidence", "Box (x1, y1 to x2, y2)"],
                interactive=False,
                label="Detected objects",
            )

    # Collapsible gallery of example images: click one to load and detect it
    if example_paths:
        with gr.Accordion(" Example Images ",
                          open=False):
            gallery = gr.Gallery(
                value=example_paths,
                columns=5,
                height="auto",
                object_fit="cover",
                allow_preview=False,
                show_label=False,
            )

    detect_btn.click(
        detect,
        [input_image, conf_slider, class_filter, show_conf],
        [banner, output_image, table],
    )

    if example_paths:
        # Click a thumbnail: load it into the input box, then run the detection
        gallery.select(pick_example, None, input_image).then(
            detect,
            [input_image, conf_slider, class_filter, show_conf],
            [banner, output_image, table],
        )

    gr.Markdown(
        "Model: YOLOv8n fine-tuned on the Bike Helmet Detection dataset "
        "(Roboflow Universe, Public Domain). Classes: With Helmet, Without Helmet.",
        elem_classes=["app-footer"],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, inbrowser=True)