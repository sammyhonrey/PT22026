"""Ingredient-list extractor: uses the fine-tuned NER model to find the
ingredient list inside the raw OCR text of a whole food package."""

import re
from pathlib import Path

MODEL_DIR = Path("models/ingredient-ner")        # local copy (after training)
HUB_REPO = "mirwaisrahimi/ingredient-list-ner"   # published model on Hugging Face
CHUNK_WORDS = 100  # must match the chunk size used in training
STRIDE = 60        # chunks overlap by CHUNK_WORDS - STRIDE words
MAX_LEN = 256      # must match the training setting

# A word is a run of characters that are not spaces or punctuation;
# every punctuation mark is its own word (same rule as the training data).
_PUNCT = r"!-/:-@\[-`{-~«»¡¿—–‘’“”„…•·"
WORD_PATTERN = re.compile(rf"[^\s{_PUNCT}]+|[{_PUNCT}]")


def split_words(text):
    """Split text into words and return a list of (word, start, end)."""
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]


def spans_from_tags(words, tags, text):
    """Cut the ingredient list(s) out of the original text.

    Args:
        words: list of (word, start, end) from split_words.
        tags: one predicted tag per word (0 = outside, 1/2 = ingredient).
        text: the original text.

    Returns:
        list of text segments, one per run of ingredient words.
    """
    segments, first, last = [], None, None
    for i, tag in enumerate(tags):
        if tag > 0:
            if first is None:
                first = i
            last = i
        elif first is not None:
            segments.append(text[words[first][1]:words[last][2]])
            first = last = None
    if first is not None:
        segments.append(text[words[first][1]:words[last][2]])
    return segments


def chunk_starts(n_words, chunk=CHUNK_WORDS, stride=STRIDE):
    """Start positions of overlapping chunks that cover every word."""
    if n_words <= chunk:
        return [0]
    starts = list(range(0, n_words - chunk + 1, stride))
    if starts[-1] + chunk < n_words:
        starts.append(n_words - chunk)
    return starts


def merge_chunk_tags(n_words, starts, chunk_tags):
    """Combine tags from overlapping chunks.

    A word near a chunk edge has little context, so for every word we keep
    the tag from the chunk in which the word is farthest from both edges.
    """
    tags = [0] * n_words
    best = [-1] * n_words
    for start, chunk in zip(starts, chunk_tags):
        size = len(chunk)
        for pos, tag in enumerate(chunk):
            centrality = min(pos, size - 1 - pos)
            if centrality > best[start + pos]:
                best[start + pos] = centrality
                tags[start + pos] = tag
    return tags


class IngredientExtractor:
    """Loads the trained model once and extracts ingredient lists."""

    def __init__(self, model_source=None):
        """Load the model from the local folder, or from Hugging Face.

        If no source is given, the local folder is used when it exists;
        otherwise the model is downloaded from the Hugging Face Hub.
        """
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        if model_source is None:
            model_source = MODEL_DIR if Path(MODEL_DIR).exists() else HUB_REPO
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_source))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_source))
        self.model.eval()

    def predict_tags(self, words):
        """Predict one tag (0/1/2) for every word."""
        torch = self._torch
        n_words = len(words)
        starts = chunk_starts(n_words)
        chunk_tags = []
        for first in range(0, len(starts), 16):  # small batches
            batch_starts = starts[first:first + 16]
            chunks = [words[s:s + CHUNK_WORDS] for s in batch_starts]
            enc = self.tokenizer(
                [[w[0] for w in chunk] for chunk in chunks],
                is_split_into_words=True, truncation=True,
                max_length=MAX_LEN, padding=True, return_tensors="pt")
            with torch.no_grad():
                guesses = self.model(**enc).logits.argmax(dim=-1)
            for b, chunk in enumerate(chunks):
                tags = [0] * len(chunk)
                previous = None
                for pos, wid in enumerate(enc.word_ids(batch_index=b)):
                    if wid is not None and wid != previous:
                        tags[wid] = int(guesses[b, pos])
                    previous = wid
                chunk_tags.append(tags)
        return merge_chunk_tags(n_words, starts, chunk_tags)

    def extract(self, text):
        """Return the ingredient list(s) found in text as a single string."""
        words = split_words(text)
        if not words:
            return ""
        segments = spans_from_tags(words, self.predict_tags(words), text)
        return " ".join(segments).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.extractor <image_path>")
        sys.exit(1)

    from src.ocr import extract_text

    ocr_text = extract_text(sys.argv[1])
    print("--- FULL OCR TEXT ---")
    print(ocr_text)
    print("\n--- INGREDIENT LIST FOUND BY THE MODEL ---")
    print(IngredientExtractor().extract(ocr_text) or "(nothing found)")