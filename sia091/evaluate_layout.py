
# Strict evaluation for the layout-aware RAG pipeline.
# Usage:
# python evaluate_layout.py <document> <eval set json>

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

from src.answerer import Answerer
from src.chunker import chunk_layout_items
from src.embedder import Embedder
from src.loader_layout import load_document_layout
from src.retriever import Retriever


CHUNK_SIZE = 40
OVERLAP = 10
TOP_K = 5


def normalise_answer(text):
    """Apply SQuAD-style answer normalization."""
    text = str(text).lower()
    text = "".join(
        ch for ch in text if ch not in set(string.punctuation)
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, gold):
    """Return 1 if normalized answers match exactly."""
    return int(
        normalise_answer(prediction) == normalise_answer(gold)
    )


def token_f1(prediction, gold):
    """Calculate token-level F1 score."""
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())

    if same == 0:
        return 0.0

    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def loose_correct(prediction, gold):
    """Apply the permissive correctness rule used in evaluate.py."""
    pred = " ".join(str(prediction).lower().split())
    ref = " ".join(str(gold).lower().split())

    if not pred:
        return False

    return ref in pred or set(pred.split()) <= set(ref.split())


def main():
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python evaluate_layout.py "
            "<document> <eval set json>"
        )

    document = sys.argv[1]
    eval_path = sys.argv[2]

    if not Path(document).is_file():
        sys.exit(f"Document not found: {document}")

    if not Path(eval_path).is_file():
        sys.exit(f"Evaluation file not found: {eval_path}")

    # Load document using the layout-aware loader.
    pages = load_document_layout(document)

    if not pages:
        sys.exit("No readable pages found in document.")

    document_text = normalise_answer(
        " ".join(item["text"] for item in pages)
    )

    with open(eval_path, encoding="utf-8") as file:
        items = json.load(file)

    if not isinstance(items, list) or not items:
        sys.exit("Evaluation dataset must be a non-empty JSON list.")

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            sys.exit(f"Invalid question at item {index}.")

        if not item.get("question") or not item.get("answer"):
            sys.exit(f"Missing question or answer at item {index}.")

    # Skip questions whose gold answer is absent from the document.
    evaluated_items = []
    skipped = 0

    for item in items:
        gold = normalise_answer(item["answer"])

        if not gold or gold not in document_text:
            skipped += 1
            print(f"SKIPPED: {item['question']}")
            continue

        evaluated_items.append(item)

    asked = len(evaluated_items)

    if asked == 0:
        sys.exit(
            f"No evaluable questions found. "
            f"Skipped: {skipped}, Total: {len(items)}"
        )

    # Build retrieval index using layout-aware chunks.
    retriever = Retriever(Embedder())

    chunks = chunk_layout_items(
        pages,
        CHUNK_SIZE,
        OVERLAP
    )

    retriever.build(chunks)

    answerer = Answerer()

    loose = 0
    exact = 0
    f1_total = 0.0

    for item in evaluated_items:
        question = item["question"]
        gold = item["answer"]

        hits = retriever.search(question, TOP_K)
        prediction = answerer.answer(question, hits)["answer"]

        is_exact = exact_match(prediction, gold)
        f1 = token_f1(prediction, gold)

        loose += int(loose_correct(prediction, gold))
        exact += is_exact
        f1_total += f1

        if not is_exact:
            print(f"NOT EXACT (F1 {f1:.2f}): {question}")
            print(
                f"    gold: {gold[:80]} | "
                f"model: {prediction[:80]}"
            )

    print("\n" + "=" * 55)
    print(f"RESULT: {Path(eval_path).name}")
    print(f"Total questions: {len(items)}")
    print(f"Evaluated questions: {asked}")
    print(f"Skipped questions: {skipped}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Loose correct: {loose}/{asked} ({loose / asked:.2%})")
    print(f"Exact Match: {exact}/{asked} ({exact / asked:.2%})")
    print(f"Token F1: {f1_total / asked:.2f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
