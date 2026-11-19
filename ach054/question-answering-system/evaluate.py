"""
evaluate.py - Compare QA models on a sample of the SQuAD v2 validation set.

Usage (from inside the project folder):
    python evaluate.py                 # 300 questions, both models
    python evaluate.py --num 100       # faster test run
    python evaluate.py --num 500 --seed 7

Metrics (the standard SQuAD metrics):
    Exact Match (EM): prediction equals a correct answer after normalising text.
    F1:               word-overlap between prediction and correct answer.
For questions that have NO answer in the text (SQuAD v2), predicting an empty
answer counts as correct. Results are saved in results/.
"""

import argparse
import json
import os
import re
import string
import time
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

from qa_engine import predict

# model name -> can this model answer "no answer"?
# distilbert-...-squad was trained on SQuAD v1 only, so it ALWAYS returns a span.
MODELS = {
    "distilbert/distilbert-base-cased-distilled-squad": False,
    "deepset/roberta-base-squad2": True,
}


# ---------- standard SQuAD text normalisation and metrics ----------
def normalize(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, truth):
    return float(normalize(prediction) == normalize(truth))


def f1_score(prediction, truth):
    pred_tokens, true_tokens = normalize(prediction).split(), normalize(truth).split()
    if not pred_tokens or not true_tokens:  # both empty -> 1, otherwise 0
        return float(pred_tokens == true_tokens)
    common = Counter(pred_tokens) & Counter(true_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(true_tokens)
    return 2 * precision * recall / (precision + recall)


def score_example(prediction, gold_answers):
    """Best score over all acceptable gold answers (empty list = unanswerable)."""
    golds = gold_answers if gold_answers else [""]
    return (max(exact_match(prediction, g) for g in golds),
            max(f1_score(prediction, g) for g in golds))


# ---------- evaluation loop ----------
def evaluate_model(model_name, allow_no_answer, examples):
    em_all, f1_all, em_has, f1_has, em_no, f1_no, times, failures = [], [], [], [], [], [], [], []
    for ex in tqdm(examples, desc=model_name.split("/")[-1]):
        golds = ex["answers"]["text"]
        start = time.perf_counter()
        pred = predict(ex["question"], ex["context"], model_name, allow_no_answer)["answer"]
        times.append(time.perf_counter() - start)

        em, f1 = score_example(pred, golds)
        em_all.append(em); f1_all.append(f1)
        (em_has if golds else em_no).append(em)
        (f1_has if golds else f1_no).append(f1)
        if f1 < 0.5 and len(failures) < 10:  # keep a few failures for the report
            failures.append({"question": ex["question"], "gold": golds[:1] or ["<no answer>"],
                             "prediction": pred or "<no answer>"})

    mean = lambda xs: round(100 * sum(xs) / len(xs), 2) if xs else None
    return {
        "model": model_name,
        "questions": len(examples),
        "EM": mean(em_all), "F1": mean(f1_all),
        "HasAns_EM": mean(em_has), "HasAns_F1": mean(f1_has),
        "NoAns_EM": mean(em_no),
        "avg_seconds_per_question": round(sum(times) / len(times), 3),
        "example_failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--num", type=int, default=300, help="number of validation questions")
    parser.add_argument("--seed", type=int, default=42, help="random seed for sampling")
    args = parser.parse_args()

    data = load_dataset("rajpurkar/squad_v2", split="validation")
    data = data.shuffle(seed=args.seed).select(range(min(args.num, len(data))))
    print(f"Evaluating on {len(data)} SQuAD v2 validation questions\n")

    results = [evaluate_model(name, flag, data) for name, flag in MODELS.items()]

    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Markdown table you can paste straight into the README / report.
    lines = ["| Model | EM | F1 | HasAns F1 | NoAns EM | Sec/question |", "|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['model']} | {r['EM']} | {r['F1']} | {r['HasAns_F1']} | "
                     f"{r['NoAns_EM']} | {r['avg_seconds_per_question']} |")
    table = "\n".join(lines)
    with open("results/results_table.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    print("\nSaved: results/eval_results.json and results/results_table.md")


if __name__ == "__main__":
    main()
