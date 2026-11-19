"""Evaluate a summarization model with ROUGE on a small sample of CNN/DailyMail.

Run:  python evaluate_model.py
Results are printed and saved to results.txt (use them in your report).
"""
import evaluate
from datasets import load_dataset
from transformers import pipeline

MODEL_ID = "sshleifer/distilbart-cnn-12-6"
N_SAMPLES = 20  # keep small so it runs fast on a laptop

data = load_dataset("cnn_dailymail", "3.0.0", split=f"test[:{N_SAMPLES}]")
summarizer = pipeline("summarization", model=MODEL_ID)

predictions = [
    summarizer(a, min_length=30, max_length=100, truncation=True)[0]["summary_text"]
    for a in data["article"]
]
references = data["highlights"]

scores = evaluate.load("rouge").compute(predictions=predictions, references=references)
lines = [f"Model: {MODEL_ID}", f"Samples: {N_SAMPLES}"]
lines += [f"{k}: {v:.4f}" for k, v in scores.items()]
print("\n".join(lines))
open("results.txt", "w").write("\n".join(lines))
