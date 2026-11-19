"""Model loading and prediction helpers for the NER app."""
import os
from functools import lru_cache

from transformers import pipeline

MODEL_NAME = "dslim/bert-base-NER"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNED_PATH = os.path.join(_ROOT, "models", "ner-finetuned")

# Display name -> model id or local folder. The fine-tuned model is listed only if its folder exists.
PRETRAINED = "Pre-trained BERT (dslim)"
FINETUNED = "Fine-tuned BERT (ours)"
MODELS = {PRETRAINED: MODEL_NAME}
if os.path.isdir(FINETUNED_PATH):
    MODELS[FINETUNED] = FINETUNED_PATH
DEFAULT_MODEL = PRETRAINED


@lru_cache(maxsize=None)
def _load(path):
    """Load a pipeline once and reuse it for every request."""
    return pipeline("ner", model=path, aggregation_strategy="simple")


def get_ner(key=DEFAULT_MODEL):
    return _load(MODELS[key])


def predict(text, key=DEFAULT_MODEL):
    """Return a list of entities: label, text, confidence score, character span."""
    results = get_ner(key)(text)
    return [
        {
            "label": r["entity_group"],
            "text": text[r["start"]:r["end"]],
            "score": float(r["score"]),
            "start": int(r["start"]),
            "end": int(r["end"]),
        }
        for r in results
    ]
