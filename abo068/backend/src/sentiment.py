import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "smartfeel_roberta_calibrated_final"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(MODEL_PATH / "calibration_config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

NEUTRAL_BIAS = config["neutral_bias"]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

print("SmartFeel model loaded successfully!")
print("Device:", device)
print("Neutral adjustment:", NEUTRAL_BIAS)


def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter some text.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logits[:, 1] += NEUTRAL_BIAS
        probabilities = torch.softmax(logits, dim=-1)[0]

    labels = ["Negative", "Neutral", "Positive"]
    scores = {
        labels[index]: round(probabilities[index].item() * 100, 2)
        for index in range(3)
    }
    predicted_label = max(scores, key=scores.get)

    return {
        "sentiment": predicted_label,
        "confidence": scores[predicted_label],
        "scores": scores,
    }


def analyze_batch(texts):
    if not texts:
        raise ValueError("Please enter at least one comment.")

    results = []
    for text in texts:
        if not text.strip():
            continue

        result = predict_sentiment(text)
        results.append(
            {
                "text": text,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "scores": result["scores"],
            }
        )

    if not results:
        raise ValueError("No valid comments were provided.")

    return results
