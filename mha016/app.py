import os
import joblib
import gradio as gr
from transformers import pipeline

# Model 1: a ready-made DistilBERT model from Hugging Face
bert = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# Model 2: our own model (TF-IDF + Logistic Regression), made by train.py
MODEL_FILE = os.path.join(os.path.dirname(__file__), "review_model.joblib")
if os.path.exists(MODEL_FILE):
    vectorizer, own_model = joblib.load(MODEL_FILE)
else:
    vectorizer, own_model = None, None


def predict_bert(text):
    result = bert(text, truncation=True, max_length=512)[0]
    return f"{result['label']} (confidence: {result['score']:.0%})"


def predict_own(text):
    if own_model is None:
        return "Model file not found. Run train.py first."
    probabilities = own_model.predict_proba(vectorizer.transform([text]))[0]
    label = "POSITIVE" if probabilities[1] >= 0.5 else "NEGATIVE"
    return f"{label} (confidence: {max(probabilities):.0%})"


def classify(text):
    # Handle empty input
    if not text or not text.strip():
        message = "Please type a review first."
        return message, message
    try:
        return predict_bert(text), predict_own(text)
    except Exception as error:
        return f"Error: {error}", f"Error: {error}"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Type a customer review", lines=4),
    outputs=[
        gr.Textbox(label="Model 1: DistilBERT (ready-made)"),
        gr.Textbox(label="Model 2: TF-IDF + Logistic Regression (my own)"),
    ],
    title="Customer Review Classifier",
    description="Type a review. Two models say if it is POSITIVE or NEGATIVE.",
    examples=[
        ["I love this product, it works perfectly!"],
        ["Terrible quality. It broke after two days."],
        ["It was fine, nothing special."],
    ],
)

demo.launch()