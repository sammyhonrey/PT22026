"""Entity-level evaluation (seqeval) of NER models on CoNLL-2003 test sentences.

Usage (from the ask084 folder):  python -m src.evaluate 3453
Models: pre-trained BERT (dslim), fine-tuned BERT (only if models/ner-finetuned exists), spaCy.
"""
import os

if os.environ.get("NER_OFFLINE") == "1":  # set on networks that block Hugging Face
    os.environ["HF_HUB_OFFLINE"] = "1"

import json
import random
import sys
import time

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities

from src.arena import SPACY_MAP
from src.inference import FINETUNED, MODELS, PRETRAINED, get_ner

TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
DATA = "data/conll2003_test.jsonl"
NAME_PRE = "BERT pre-trained (dslim/bert-base-NER)"
NAME_FT = "BERT fine-tuned by us"
NAME_SPACY = "spaCy en_core_web_sm (mapped labels)"


def make_bert_tagger(ner):
    """Run a BERT pipeline on the joined sentence and map entities back to word tags."""
    def tags_for(tokens):
        text = " ".join(tokens)
        offsets, pos = [], 0
        for t in tokens:
            offsets.append((pos, pos + len(t)))
            pos += len(t) + 1
        tags = ["O"] * len(tokens)
        for ent in ner(text):
            idx = [i for i, (s, e) in enumerate(offsets) if s < ent["end"] and e > ent["start"]]
            for k, i in enumerate(idx):
                tags[i] = ("B-" if k == 0 else "I-") + ent["entity_group"]
        return tags
    return tags_for


def make_spacy_tagger():
    """Run spaCy on the gold tokenization so tags line up word by word."""
    import spacy
    from spacy.tokens import Doc
    nlp = spacy.load("en_core_web_sm")

    def tags_for(tokens):
        doc = nlp(Doc(nlp.vocab, words=tokens))
        tags = []
        for tok in doc:
            label = SPACY_MAP.get(tok.ent_type_) if tok.ent_iob_ != "O" else None
            if label is None:
                tags.append("O")
            else:
                tags.append(("I-" if tok.ent_iob_ == "I" else "B-") + label)
        return tags
    return tags_for


def fmt(tokens, tags):
    return " | ".join(f"{' '.join(tokens[s:e + 1])} [{t}]" for t, s, e in get_entities(tags)) or "(none)"


def write_dump(path, rows, gold, pred, label):
    with open(path, "w", encoding="utf-8") as f:
        shown = 0
        for r, g, p in zip(rows, gold, pred):
            if g != p and shown < 40:
                f.write("SENT: " + " ".join(r["tokens"]) + "\n")
                f.write("GOLD: " + fmt(r["tokens"], g) + "\n")
                f.write(f"{label}: " + fmt(r["tokens"], p) + "\n\n")
                shown += 1


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rows = [json.loads(line) for line in open(DATA, encoding="utf-8")]
    random.Random(42).shuffle(rows)
    rows = rows[:n]
    print(f"Evaluating on {len(rows)} test sentences")
    gold = [[TAGS[i] for i in r["ner_tags"]] for r in rows]

    models = {NAME_PRE: make_bert_tagger(get_ner(PRETRAINED))}
    if FINETUNED in MODELS:
        models[NAME_FT] = make_bert_tagger(get_ner(FINETUNED))
    models[NAME_SPACY] = make_spacy_tagger()

    preds, results = {}, {}
    for name, fn in models.items():
        start = time.time()
        preds[name] = [fn(r["tokens"]) for r in rows]
        secs = time.time() - start
        rep = classification_report(gold, preds[name], output_dict=True)
        results[name] = {
            "precision": precision_score(gold, preds[name]),
            "recall": recall_score(gold, preds[name]),
            "f1": f1_score(gold, preds[name]),
            "seconds": secs,
            "per_class": {k: rep[k] for k in ["PER", "ORG", "LOC", "MISC"]},
        }
        print(f"\n=== {name}  ({secs:.0f}s) ===")
        print(classification_report(gold, preds[name], digits=3))

    os.makedirs("docs", exist_ok=True)
    lines = [f"# Results on {len(rows)} random CoNLL-2003 test sentences (seed 42)", "",
             "| Model | Precision | Recall | F1 | Time (s) |", "|---|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['seconds']:.0f} |")
    lines += ["", "## Per-entity F1", "", "| Model | PER | ORG | LOC | MISC |", "|---|---|---|---|---|"]
    for name, r in results.items():
        f = [r["per_class"][c]["f1-score"] for c in ["PER", "ORG", "LOC", "MISC"]]
        lines.append(f"| {name} | " + " | ".join(f"{x:.3f}" for x in f) + " |")
    open("docs/results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

    write_dump("docs/error_dump.txt", rows, gold, preds[NAME_PRE], "BERT")
    if NAME_FT in preds:
        write_dump("docs/error_dump_finetuned.txt", rows, gold, preds[NAME_FT], "BERT-FT")
    print("\nSaved docs/results.md and the error dump file(s)")


if __name__ == "__main__":
    main()
