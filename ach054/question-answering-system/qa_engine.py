"""
qa_engine.py - Question-answering model logic for ContextAI.

This file contains ONLY machine-learning code (no user-interface code), so it
can be reused by both the Gradio app (app.py) and the evaluation script
(evaluate.py).

Model: deepset/roberta-base-squad2 (RoBERTa fine-tuned on SQuAD 2.0).
Task:  extractive question answering - the model finds the answer as a span
       of text inside the context you give it.
"""

import os
from functools import lru_cache

from transformers import pipeline

# ------------------------------------------------------------
# Configuration (can be changed without touching the code below)
# ------------------------------------------------------------
# You can switch models by setting an environment variable, e.g.
#   QA_MODEL=distilbert/distilbert-base-cased-distilled-squad python app.py
DEFAULT_MODEL = os.getenv("QA_MODEL", "deepset/roberta-base-squad2")

MAX_SEQ_LEN = 512      # max tokens per window (question + context chunk)
DOC_STRIDE = 128       # overlap between windows for long contexts
MAX_QUESTION_LEN = 64  # questions longer than this are truncated
MAX_ANSWER_LEN = 40    # longest answer span (in tokens) we accept
MIN_CONFIDENCE = 0.01  # below this score we say "no confident answer"


@lru_cache(maxsize=4)
def load_model(model_name: str = DEFAULT_MODEL):
    """Load a Hugging Face question-answering pipeline (downloaded once, then cached)."""
    print(f"Loading Question Answering model: {model_name} ...")
    qa_pipeline = pipeline("question-answering", model=model_name)
    print("Model loaded successfully!")
    return qa_pipeline


def predict(question: str, context: str, model_name: str = DEFAULT_MODEL,
            allow_no_answer: bool = True) -> dict:
    """
    Find the answer to `question` inside `context`.

    Returns a dictionary:
        answer : the extracted text span ("" if the model thinks there is none)
        score  : model confidence between 0 and 1
        start  : character index where the answer starts in the context
        end    : character index where the answer ends in the context

    `allow_no_answer=True` lets SQuAD-2.0 models say "the answer is not in the
    text" instead of being forced to pick some span.
    """
    qa_pipeline = load_model(model_name)
    result = qa_pipeline(
        question=question.strip(),
        context=context.strip(),
        max_seq_len=MAX_SEQ_LEN,
        doc_stride=DOC_STRIDE,
        max_question_len=MAX_QUESTION_LEN,
        max_answer_len=MAX_ANSWER_LEN,
        handle_impossible_answer=allow_no_answer,
    )
    score = max(0.0, min(float(result.get("score", 0.0)), 1.0))
    return {
        "answer": (result.get("answer") or "").strip(),
        "score": score,
        "start": result.get("start"),
        "end": result.get("end"),
    }
