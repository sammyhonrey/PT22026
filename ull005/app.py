"""Gradio app: pneumonia detection from chest X-rays with Grad-CAM."""
from pathlib import Path

import gradio as gr
import torch

from gradcam import GradCAM, overlay_heatmap
from model import build_model
from utils import get_transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISCLAIMER = (
    "**Educational tool only.** This is a university project, not a medical "
    "device. It must not be used for diagnosis or treatment decisions. "
    "Always consult a qualified doctor."
)

_loaded = {}  # cache: model name -> (model, GradCAM or None)


def available_models():
    """Model names that have a trained checkpoint in checkpoints/."""
    return [m for m in ("resnet50", "vit")
            if (CHECKPOINT_DIR / f"{m}_best.pt").exists()]


def load_model(name):
    """Load a fine-tuned model once and reuse it."""
    if name in _loaded:
        return _loaded[name]
    ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = build_model(name, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.to(DEVICE).eval()
    cam = GradCAM(model, model.layer4[-1]) if name == "resnet50" else None
    _loaded[name] = (model, cam)
    return _loaded[name]


def predict(image, model_name):
    """Return (class probabilities, Grad-CAM overlay or None)."""
    if image is None:
        raise gr.Error("Please upload a chest X-ray image first.")
    try:
        model, cam_tool = load_model(model_name)
        _, eval_tf = get_transforms(model_name)
        pil = image.convert("RGB")
        tensor = eval_tf(pil).unsqueeze(0).to(DEVICE)

        if cam_tool is not None:  # ResNet-50: prediction + heatmap
            cam, _, probs = cam_tool.generate(tensor)
            heatmap = overlay_heatmap(pil, cam)
        else:  # ViT: prediction only
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
            heatmap = None
    except FileNotFoundError as err:
        raise gr.Error(str(err))
    except Exception as err:  # corrupted file, model failure, etc.
        raise gr.Error(f"Could not analyze this image: {err}")

    return {c: float(p) for c, p in zip(CLASSES, probs)}, heatmap


def build_demo(models):
    with gr.Blocks(title="Pneumonia Detection from Chest X-rays") as demo:
        gr.Markdown("# Pneumonia Detection from Chest X-rays")
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Chest X-ray")
                model_name = gr.Dropdown(models, value=models[0], label="Model")
                button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                label = gr.Label(num_top_classes=2, label="Prediction")
                heatmap = gr.Image(label="Grad-CAM heatmap (ResNet-50 only)")
        button.click(predict, [image, model_name], [label, heatmap])
    return demo


if __name__ == "__main__":
    models = available_models()
    if not models:
        raise SystemExit(
            "No trained checkpoint found. Put resnet50_best.pt (and/or "
            "vit_best.pt) in the checkpoints/ folder. See README.md.")
    build_demo(models).launch() 