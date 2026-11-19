"""Gradio app: AI-Powered Food Allergen Detection and Ingredient Risk Assistant."""

import gradio as gr

from src.detector import ALLERGENS
from src.extractor import IngredientExtractor
from src.ocr import extract_text
from src.risk import assess_risk

ICONS = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
ADVICE = {
    "HIGH": "**Do not consume.** An allergen you selected is in the ingredients.",
    "MEDIUM": ("**Use caution.** An allergen you selected may be present "
               "(trace warning or mentioned outside the ingredient list)."),
    "LOW": "No allergen you selected was detected. Always double-check the package.",
}


def load_extractor():
    """Load the fine-tuned ingredient-list model (local copy or Hugging Face)."""
    try:
        return IngredientExtractor()
    except Exception as exc:  # the app still works without the model
        print(f"Could not load the model ({exc}): scanning the whole text instead.")
        return None


EXTRACTOR = load_extractor()


def analyze(image_path, pasted_text, user_allergies):
    """Run OCR (or use pasted text), find the ingredient list, assess risk."""
    if not user_allergies:
        return "⚠️ Please select at least one allergy.", "", ""

    if pasted_text and pasted_text.strip():
        text = pasted_text.strip()
    elif image_path:
        try:
            text = extract_text(image_path)
        except Exception as exc:  # show a friendly message instead of crashing
            return f"⚠️ Could not read the image: {exc}", "", ""
    else:
        return "⚠️ Please upload an image or paste the ingredient text.", "", ""

    if not text:
        return "⚠️ No text was found. Try a clearer, closer photo.", "", ""

    result = assess_risk(text, user_allergies, EXTRACTOR)

    lines = [
        f"## {ICONS[result['overall']]} Overall risk: {result['overall']}",
        ADVICE[result["overall"]],
        "",
        "### Your allergies",
    ]
    for name, level in result["per_allergen"].items():
        lines.append(f"- {ICONS[level]} **{name}**: {level} "
                     f"({result['reasons'][name]})")

    lines.append("\n### Allergens in the ingredient list" if result["used_model"]
                 else "\n### Allergens found in the text")
    lines.append(", ".join(result["direct"]) or "None")
    if result["used_model"]:
        lines.append("\n### Mentioned elsewhere on the package")
        lines.append(", ".join(result["elsewhere"]) or "None")
    lines.append("\n### Detected in 'may contain' warnings")
    lines.append(", ".join(result["traces"]) or "None")

    if result["used_model"]:
        lines.append("\n*The ingredient list was located by the AI model.*")
    else:
        lines.append("\n*The ingredient list could not be located, so the "
                     "whole text was scanned.*")
    lines.append(
        "\n*This tool is an aid, not medical advice. OCR can make mistakes, "
        "so always read the original label.*"
    )
    return "\n".join(lines), text, result["ingredient_text"]


with gr.Blocks(title="Food Allergen Assistant") as demo:
    gr.Markdown(
        "# 🥜 Food Allergen Detection and Ingredient Risk Assistant\n"
        "Upload a photo of an ingredient label (OCR) or paste the text, "
        "choose your allergies, and get a risk report."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="filepath", label="Ingredient label photo")
            pasted = gr.Textbox(
                label="Or paste ingredient text (optional)", lines=4
            )
            allergies = gr.CheckboxGroup(
                choices=list(ALLERGENS.keys()), label="Your allergies"
            )
            button = gr.Button("Analyze", variant="primary")
        with gr.Column():
            report = gr.Markdown()
            ingredient_box = gr.Textbox(
                label="Ingredient list found by the AI model", lines=5)
            ocr_text = gr.Textbox(label="Full extracted text", lines=6)

    button.click(analyze, [image, pasted, allergies],
                 [report, ocr_text, ingredient_box])

if __name__ == "__main__":
    demo.launch()