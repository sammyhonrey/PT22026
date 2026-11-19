"""Upload the trained ingredient-list model to the Hugging Face Hub.

Run once from the rah086 folder:  python -m src.upload_model
"""

import json
from pathlib import Path

REPO_ID = "mirwaisrahimi/ingredient-list-ner"
MODEL_DIR = Path("models/ingredient-ner")
METRICS_PATH = Path("docs/ner_metrics.json")


def build_model_card(metrics):
    """Create the README (model card) shown on the Hugging Face model page."""
    return f"""---
language: multilingual
library_name: transformers
pipeline_tag: token-classification
base_model: distilbert/distilbert-base-multilingual-cased
tags:
  - token-classification
  - food
  - ocr
  - ingredients
---

# Ingredient-list detector (multilingual)

A token-classification model that finds the **ingredient list** inside the
raw OCR text of a food package. It is part of the project
*AI-Powered Food Allergen Detection and Ingredient Risk Assistant Using NLP
and OCR* (Professional Training course, student folder `rah086`).

## Task
Every word of the text gets one tag:

| id | label | meaning |
|----|-------|---------|
| 0 | O | not part of the ingredient list |
| 1 | B-ING | first word of the ingredient list |
| 2 | I-ING | inside the ingredient list |

## Training
- **Base model:** `distilbert-base-multilingual-cased`
- **Data:** Open Food Facts ingredient detection dataset
  (labelled OCR texts of product packages in several languages);
  {metrics['train_documents']} training documents
- **Setup:** {metrics['epochs']} epochs, learning rate 5e-5, batch size 16,
  texts split into chunks of 100 words, trained on a CPU in
  {metrics['training_minutes']} minutes

## Evaluation
Evaluated on {metrics['test_documents']} held-out test documents
({metrics['words_evaluated']} words). The score answers the question
"is this word part of the ingredient list?".

| Metric | Score |
|--------|-------|
| Precision | {metrics['precision']} |
| Recall | {metrics['recall']} |
| F1 | {metrics['f1']} |
| Accuracy | {metrics['accuracy']} |

## Usage
The model works on single words, so split the text into words first
(punctuation marks are separate words) and classify chunks of up to 100 words.
The project repository contains the full pipeline (`src/extractor.py`).

## Limitations
- Scores are word-level, not exact span matches.
- Accuracy drops when the OCR text is very noisy or the list is cut by the photo.
- Allergen statements such as "may contain ..." are usually outside the
  ingredient list, so they must be handled separately.
- Check the license and terms of the Open Food Facts data before reusing the model.
"""


def main():
    from huggingface_hub import HfApi

    if not (MODEL_DIR / "model.safetensors").exists():
        print(f"No trained model found in {MODEL_DIR}. Run src.train_ner first.")
        return

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    (MODEL_DIR / "README.md").write_text(build_model_card(metrics),
                                         encoding="utf-8")

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload ingredient-list NER model",
    )
    print(f"Done! Model page: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()