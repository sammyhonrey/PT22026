"""Gradio app: type an SMS and the model says Spam or Not spam."""

import math
import os

# Fix: stop the proxy or VPN from blocking the local app
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

import gradio as gr
import joblib

MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    raise SystemExit("model.joblib not found. Run 'python train.py' first.")

model = joblib.load(MODEL_PATH)


def spam_score(message):
    """Return a number from 0 to 1. Closer to 1 means more likely spam."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([message])[0][1])
    # Linear SVM has no probability, so we squash its score into 0..1
    raw = float(model.decision_function([message])[0])
    return 1 / (1 + math.exp(-raw))


def classify(message):
    if message is None or not message.strip():
        raise gr.Error("Please type an SMS message first.")
    score = spam_score(message)
    return {"Spam": score, "Not spam": 1 - score}


examples = [
    ["Congratulations! You won a FREE prize. Call 09061701461 now to claim."],
    ["URGENT! Your account is locked. Click this link to verify now."],
    ["Hey, are we still meeting for lunch at 12?"],
    ["Can you send me the notes from today's class?"],
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=4, label="SMS message",
                      placeholder="Type or paste an SMS here..."),
    outputs=gr.Label(num_top_classes=2, label="Result"),
    title="SMS Spam Detector",
    description="Type an SMS message. The model tells you if it looks like spam.",
    examples=examples,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()