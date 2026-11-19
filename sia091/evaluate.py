# Evaluate retrieval and answer quality on the questions in eval/eval_set.json.
# Usage: python evaluate.py [path to a PDF or TXT file]

import csv
import json
import re
import sys

from src.answerer import Answerer
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.loader import load_document
from src.retriever import Retriever

DEFAULT_DOCUMENT = "data/assignment_guidelines.pdf"
EVAL_SET = "eval/eval_set.json"
RESULTS_FILE = "eval/results.csv"
CHUNK_SIZES = (40, 60, 120)
TOP_KS = (1, 3, 5)
# Same weight as the default in src/answerer.py, so the table matches the app
RETRIEVAL_WEIGHT = 5.0

# Different rules for choosing one answer from the candidates of the top chunks
STRATEGIES = {
    "retrieval_first": lambda cands: cands[0],
    "qa_prob": lambda cands: max(cands, key=lambda c: c["prob"]),
    "qa_logit": lambda cands: max(cands, key=lambda c: c["logit"]),
    "retrieval_x_prob": lambda cands: max(cands, key=lambda c: c["retrieval"] * c["prob"]),
    "logit_plus_retrieval": lambda cands: max(cands, key=lambda c: c["logit"] + RETRIEVAL_WEIGHT * c["retrieval"]),
}


def normalise(text):
    # Lower-case the text and collapse all whitespace
    return re.sub(r"\s+", " ", text).strip().lower()


def is_correct(prediction, gold):
    # Correct if the gold answer is inside the prediction,
    # or if every word of the prediction is inside the gold answer
    pred, ref = normalise(prediction), normalise(gold)
    if not pred:
        return False
    return ref in pred or set(pred.split()) <= set(ref.split())


def main():
    document = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOCUMENT
    with open(EVAL_SET, encoding="utf-8") as file:
        items = json.load(file)

    pages = load_document(document)
    embedder = Embedder()
    answerer = Answerer()
    total = len(items)

    rows = []
    for chunk_size in CHUNK_SIZES:
        retriever = Retriever(embedder)
        retriever.build(chunk_pages(pages, chunk_size, chunk_size // 4))

        for top_k in TOP_KS:
            found_evidence = 0
            correct = {name: 0 for name in STRATEGIES}

            for item in items:
                hits = retriever.search(item["question"], top_k)

                # Recall@k: is the evidence text inside any of the top chunks?
                evidence = normalise(item["evidence"])
                if any(evidence in normalise(hit["text"]) for hit in hits):
                    found_evidence += 1

                # Answer accuracy for every ranking rule
                candidates = answerer.candidates(item["question"], hits)
                for name, choose in STRATEGIES.items():
                    if is_correct(choose(candidates)["answer"], item["answer"]):
                        correct[name] += 1

            row = {
                "chunk_size": chunk_size,
                "top_k": top_k,
                "recall": round(found_evidence / total, 3),
            }
            for name in STRATEGIES:
                row[name] = round(correct[name] / total, 3)
            rows.append(row)
            print(row)

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
