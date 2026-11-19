"""
AI Food Analyzer - Gradio Application
This app uses a pre-trained hybrid ResNet50 + DeiT-Base model to classify images
into one of 101 Food-101 categories and displays nutrition information.
Run with:
       python app.py
"""
from __future__ import annotations

import html
import logging
import os
import threading
from typing import Any, Dict, List, Tuple

import gradio as gr
import torch
from PIL import Image, ImageOps

from food_info import FOOD_INFO
from model_loader import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("food_classifier.app")

# Lazy, thread-safe model loading so the UI still starts if the download
# fails; the error is then shown to the user on the first prediction.
_MODEL: Any = None
_TRANSFORM: Any = None
_IDX_TO_LABEL: Dict[int, str] | None = None
_LOAD_ERROR: str | None = None
_LOAD_LOCK = threading.Lock()

_READY_PREDICTED = (
    "### 🍽️ Predicted Food\n\n"
    "Upload or capture an image, then press **Analyze Food**."
)
_EMPTY_NUTRITION = (
    "### 🔥 Calories\n\n—\n\n"
    "### 🧂 Ingredients\n\n—\n\n"
    "### 📖 Description\n\n—"
)

# Label compact 
CUSTOM_CSS = """
.compact-label,
.compact-label .label-container {
    min-height: 0 !important;
}

.compact-label {
    padding: 0.25rem !important;
}
"""


def _get_model():
    """Load the model, transform, and class mapping only once and cache them.
    Uses a threading lock to ensure that if multiple users click "Analyze"
    at the same time, the heavy 1.2GB model is only loaded into memory once.
    """
    global _MODEL, _TRANSFORM, _IDX_TO_LABEL, _LOAD_ERROR

    if _MODEL is not None:
        return _MODEL, _TRANSFORM, _IDX_TO_LABEL

    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL, _TRANSFORM, _IDX_TO_LABEL

        try:
            LOGGER.info("Loading Food-101 hybrid model (first request) ...")
            requested_device = os.getenv("FOOD_CLASSIFIER_DEVICE") or None
            _MODEL, _TRANSFORM, _IDX_TO_LABEL = load_model(device=requested_device)
            _LOAD_ERROR = None
        except Exception as exc:
            _LOAD_ERROR = str(exc)
            LOGGER.exception("Model loading failed")
            raise RuntimeError(
                "The classification model could not be loaded. "
                "Please check your internet connection and try again."
            ) from exc

    return _MODEL, _TRANSFORM, _IDX_TO_LABEL


def _pretty(name: str) -> str:
    """Turn fish_and_chips into Fish And Chips."""
    return name.replace("_", " ").title()


def _message_outputs(
    message: str,
) -> Tuple[Dict[str, float], str, str]:
    """Return an empty prediction plus a readable message in each panel."""
    return (
        {},
        f"### 🍽️ Predicted Food\n\n{message}",
        _EMPTY_NUTRITION,
    )


def _calorie_badge(calories: int | float) -> str:
    """Return a colour-coded calorie badge for the nutrition panel."""
    if calories < 200:
        return f"🟢 Low ({calories} kcal)"
    if calories <= 400:
        return f"🟡 Moderate ({calories} kcal)"
    return f"🔴 High ({calories} kcal)"


def _format_nutrition(info: Dict | None) -> str:
    """Build one combined Markdown block for the nutrition column."""
    if info is None:
        unavailable = "> **Nutrition information not available** for this category."
        return (
            "### 🔥 Calories\n\n"
            f"{unavailable}\n\n"
            "### 🧂 Ingredients\n\n"
            f"{unavailable}\n\n"
            "### 📖 Description\n\n"
            f"{unavailable}"
        )

    calories = info.get("calories_per_100g")
    calories_text = (
        "Nutrition information not available."
        if calories is None
        else _calorie_badge(calories)
    )
    # If the food dictionary ever contains HTML tags, this ensures they are rendered as text, not code.
    ingredients = info.get("ingredients") or []
    ingredients_text = (
        ", ".join(html.escape(str(item)) for item in ingredients)
        if ingredients
        else "Not available"
    )

    description = info.get("description") or "No description available."

    return (
        "### 🔥 Calories\n\n"
        f"{calories_text}\n\n"
        "### 🧂 Ingredients\n\n"
        f"{ingredients_text}\n\n"
        "### 📖 Description\n\n"
        f"{html.escape(str(description))}"
    )


def _format_predicted(predicted: str, confidence: float) -> str:
    """Build the prominent predicted-food panel."""
    warning = ""
    if confidence < 0.30:
        warning = (
            "\n\n> ⚠️ **Low confidence.** The image may not contain a "
            "Food-101 category."
        )
    return (
        "### 🍽️ Predicted Food\n\n"
        f"## {html.escape(_pretty(predicted))}\n\n"
        f"**Confidence:** {confidence * 100:.2f}%"
        f"{warning}"
    )


def predict(
    image: Image.Image | None,
) -> Tuple[Dict[str, float], str, str]:
    """Classify an image and return the three UI output values.
    Returns
    top3
        Mapping of pretty food names to probabilities for ``gr.Label``.
    predicted
        Markdown for the prominent predicted-food panel.
    nutrition
        Combined Markdown block with calories, ingredients, and description.

    This function is deliberately defensive: every failure mode returns a
    readable message instead of raising and breaking the Gradio event.
    """
    if image is None:
        return _message_outputs(
            "Please upload an image or take a webcam photo first."
        )

    if not isinstance(image, Image.Image):
        return _message_outputs(
            "The input could not be interpreted as an image. Please try a "
            "PNG, JPEG, or WebP file."
        )

    try:
        rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:  # noqa: BLE001 - corrupt/truncated image
        LOGGER.exception("Could not convert image to RGB")
        return _message_outputs(
            "The image file appears to be corrupted or unreadable."
        )

    try:
        model, transform, idx_to_label = _get_model()
    except Exception as exc:  # noqa: BLE001
        detail = (
            f"\n\n**Details:** `{html.escape(_LOAD_ERROR)}`"
            if _LOAD_ERROR
            else ""
        )
        return _message_outputs(
            f"**Model unavailable:** {html.escape(str(exc))}{detail}"
        )

    try:
        device = getattr(model, "device", torch.device("cpu"))
        # 1. Preprocessing: Convert image to RGB, resize to 224x224, and normalize.
        # 2. Unsqueeze(0): The model expects a batch, so we add a dimension (Batch, Channel, Height, Width).
        tensor = transform(rgb_image).unsqueeze(0).to(device)
        # inference_mode() turns off gradient tracking to save memory and speed up inference.
        with torch.inference_mode():
            output = model(tensor)
        if isinstance(output, dict):
            logits = output.get("logits")
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if not isinstance(logits, torch.Tensor):
            raise ValueError("Model did not return a tensor of logits.")

        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if logits.ndim != 2:
            raise ValueError(
                f"Unexpected model output shape: {tuple(logits.shape)}. "
                "Expected (batch, classes)."
            )

        if logits.shape[0] != 1:
            raise ValueError(
                f"Unexpected batch dimension: {logits.shape[0]} (expected 1)."
            )

        if logits.shape[1] != len(idx_to_label):
            raise ValueError(
                f"Model returned {logits.shape[1]} classes but the class "
                f"mapping contains {len(idx_to_label)}."
            )
        
        # The model returns logits (raw scores). We apply Softmax to convert 
        # these into probabilities that sum to 1.0.
        probabilities = torch.softmax(logits.float(), dim=1)[0]
        k = min(3, probabilities.numel())
        top_probs, top_indices = torch.topk(probabilities, k=k)

        top: List[Tuple[str, float]] = []
        for score, index in zip(top_probs.tolist(), top_indices.tolist()):
            top.append(
                (idx_to_label.get(int(index), f"class_{index}"), float(score))
            )

        if not top:
            raise ValueError("The model returned no predictions.")

        predicted = top[0][0]
        info = FOOD_INFO.get(predicted)
        top3 = {_pretty(name): float(score) for name, score in top}

        return (
            top3,
            _format_predicted(predicted, top[0][1]),
            _format_nutrition(info),
        )

    except Exception as exc:  # noqa: BLE001 - never break the UI
        LOGGER.exception("Inference failed")
        return _message_outputs(
            "**Classification failed:** "
            f"`{html.escape(type(exc).__name__)}: {html.escape(str(exc))}`\n\n"
            "Please try another image. If the problem persists, delete the "
            "Hugging Face cache and restart the app."
        )

#  Reset the image and all result panels.
def _clear_outputs():
    return (
        None,
        {},
        _READY_PREDICTED,
        _EMPTY_NUTRITION,
    )

# Create the Gradio Blocks interface.
def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="AI Food Analyzer",
        theme=gr.themes.Default(), #theme=gr.themes.Soft() theme=gr.themes.Monochrome() theme=gr.themes.Glass()
        # theme=gr.themes.Default(primary_hue="indigo", secondary_hue="slate")
        #theme=gr.themes.Default(font=["Arial", "sans-serif"]),
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown(
            "# AI Food Analyzer\n"
            "Upload a photo to instantly identify the food, estimate calories, and see the ingredients."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300): #This is the column 1 (Image Upload)
                image_input = gr.Image(
                    type="pil",
                    sources=["upload", "webcam"],
                    label="📷 Upload or capture a food image",
                    height=360,
                )
                with gr.Row():
                    classify_button = gr.Button(
                        "🔍 Analyze Food", variant="primary", scale=2
                    )
                    clear_button = gr.Button(
                        "Clear", variant="secondary", scale=1
                    )

                gr.Markdown(
                    "<sub>First prediction downloads the model checkpoint "
                    "(about 1.2 GB) and may take a minute.</sub>"
                )

                gr.Examples(
                    examples=[["pizza.jpg"], ["dumplings.jpg"]],
                    inputs=image_input,
                    label="Try an example:",
                )

            with gr.Column(scale=1, min_width=300):  #This is the column 2
                gr.Markdown("### 🎯 Top-3 Predictions")
                top3_output = gr.Label(
                    num_top_classes=3,
                    show_label=False,
                    elem_classes=["compact-label"],
                )
                predicted_output = gr.Markdown(_READY_PREDICTED)

            with gr.Column(scale=1, min_width=300): #This is the column 3
                gr.Markdown("### 🥗 Nutrition")
                nutrition_output = gr.Markdown(_EMPTY_NUTRITION)

        classify_button.click(
            fn=predict,
            inputs=image_input,
            outputs=[
                top3_output,
                predicted_output,
                nutrition_output,
            ],
        )
        clear_button.click(
            fn=_clear_outputs,
            inputs=None,
            outputs=[
                image_input,
                top3_output,
                predicted_output,
                nutrition_output,
            ],
        )

        gr.Markdown(
            """
            ---
            **Model:** [`codealchemist01/food-image-classifier-hybrid`](https://huggingface.co/codealchemist01/food-image-classifier-hybrid)
            · ResNet50 + DeiT-Base with adaptive cross-attention fusion
            · Trained on Food-101 (101 classes)
            · Nutrition values are approximate and not medical advice.
            """
        )

    return demo


demo = build_interface()


if __name__ == "__main__":
    demo.launch(quiet=True)