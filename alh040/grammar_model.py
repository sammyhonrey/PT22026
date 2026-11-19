"""
grammar_model.py - Loads the AI model and uses it to correct English text.

Model: vennify/t5-base-grammar-correction  (Hugging Face Model Hub)
  - Architecture: T5 ("Text-to-Text Transfer Transformer", Google, 2020),
    an encoder-decoder Transformer with about 220 million parameters.
  - The ENCODER reads the whole (incorrect) sentence and builds an
    understanding of every word in context, using self-attention.
  - The DECODER writes the corrected sentence one word-piece at a time,
    looking back at the encoder's understanding at every step.
  - It was fine-tuned on JFLEG, a dataset of sentences written by English
    learners, each corrected by 4 human annotators.
  - The model expects every input to start with the prefix "grammar: ".

The model is downloaded automatically (about 900 MB) the first time it is used
and then cached on your computer, so later runs work offline.
"""

import re

import os
_LOCAL_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "t5-grammar")
MODEL_NAME = _LOCAL_MODEL if os.path.isdir(_LOCAL_MODEL) else "vennify/t5-base-grammar-correction"
MAX_INPUT_CHARS = 3000      # protects the app from extremely long inputs
MAX_TOKENS = 128            # longest sentence (in word-pieces) the model will write

_tokenizer = None
_model = None
_device = None


def load_model():
    """Load the tokenizer and model once and keep them in memory."""
    global _tokenizer, _model, _device
    if _model is None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {MODEL_NAME} on {_device} (first time downloads ~900 MB)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(_device)
        _model.eval()
        print("Model ready.")
    return _tokenizer, _model


def split_sentences(text: str) -> list:
    """Split a paragraph into sentences so each one is corrected separately.

    The model was trained on single sentences, so it works best one at a time.
    """
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def correct_sentences(sentences: list, num_beams: int = 5, batch_size: int = 8) -> list:
    """Correct a list of sentences and return the corrected list (same order)."""
    import torch

    tokenizer, model = load_model()
    results = []
    for start in range(0, len(sentences), batch_size):
        batch = ["grammar: " + s for s in sentences[start:start + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=MAX_TOKENS).to(_device)
        with torch.no_grad():
            # Beam search: the model keeps the `num_beams` most likely sentences
            # at every step and returns the best complete one.
            outputs = model.generate(**inputs, num_beams=num_beams,
                                     max_new_tokens=MAX_TOKENS, early_stopping=True)
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return [r.strip() for r in results]


def correct_text(text: str) -> str:
    """Correct a whole paragraph (any number of sentences)."""
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"Text is too long ({len(text)} characters). "
                         f"Please enter at most {MAX_INPUT_CHARS} characters.")
    sentences = split_sentences(text)
    if not sentences:
        return ""
    return " ".join(correct_sentences(sentences))


if __name__ == "__main__":
    # Quick check from the command line:  python grammar_model.py
    example = "Yesterday i go to the school and buyed a apple. She dont like it."
    print("Original :", example)
    print("Corrected:", correct_text(example))
