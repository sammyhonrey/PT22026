"""Model arena: compare any two NER models on the same text."""
from functools import lru_cache

import spacy

from src.inference import MODELS, predict

SPACY_NAME = "spaCy (en_core_web_sm)"
ARENA_CHOICES = list(MODELS) + [SPACY_NAME]

# spaCy uses OntoNotes labels; map them onto the 4 CoNLL labels.
# Labels with no CoNLL equivalent (DATE, MONEY, CARDINAL, ...) are ignored.
SPACY_MAP = {
    "PERSON": "PER",
    "ORG": "ORG",
    "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "NORP": "MISC", "EVENT": "MISC", "LANGUAGE": "MISC",
    "WORK_OF_ART": "MISC", "PRODUCT": "MISC", "LAW": "MISC",
}


@lru_cache(maxsize=1)
def get_spacy():
    return spacy.load("en_core_web_sm")


def predict_spacy(text):
    doc = get_spacy()(text)
    out = []
    for ent in doc.ents:
        label = SPACY_MAP.get(ent.label_)
        if label:
            out.append({"label": label, "text": ent.text,
                        "start": ent.start_char, "end": ent.end_char})
    return out


def to_segments(text, entities):
    """Convert entities into (text, label) segments for gr.HighlightedText."""
    segments, pos = [], 0
    for e in sorted(entities, key=lambda e: e["start"]):
        if e["start"] < pos:
            continue
        if e["start"] > pos:
            segments.append((text[pos:e["start"]], None))
        segments.append((text[e["start"]:e["end"]], e["label"]))
        pos = e["end"]
    if pos < len(text):
        segments.append((text[pos:], None))
    return segments


def _overlaps(a, others):
    return any(a["start"] < o["end"] and o["start"] < a["end"] for o in others)


def _entities(text, name, threshold):
    if name == SPACY_NAME:
        return predict_spacy(text)
    return [e for e in predict(text, name) if e["score"] >= threshold]


def compare(text, name_a, name_b, threshold=0.5):
    """Return entities of model A, entities of model B and a comparison table."""
    a = _entities(text, name_a, threshold)
    b = _entities(text, name_b, threshold)
    da = {(e["start"], e["end"]): e for e in a}
    db = {(e["start"], e["end"]): e for e in b}

    rows = []
    for span in sorted(set(da) | set(db)):
        ea, eb = da.get(span), db.get(span)
        if ea and eb:
            status = "Agree" if ea["label"] == eb["label"] else "Label conflict"
        elif ea:
            status = "Boundary conflict" if _overlaps(ea, b) else f"Only {name_a}"
        else:
            status = "Boundary conflict" if _overlaps(eb, a) else f"Only {name_b}"
        shown = (ea or eb)["text"]
        rows.append([shown, ea["label"] if ea else "-", eb["label"] if eb else "-", status])
    return a, b, rows
