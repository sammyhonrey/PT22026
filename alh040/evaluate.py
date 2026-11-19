"""
evaluate.py - Measure how well the AI corrects real learner sentences.

Dataset: JFLEG test split (748 sentences written by English learners, each with
4 corrections written by human annotators). Downloaded automatically
from the Hugging Face Hub (jhu-clsp/jfleg).

We compare two things against the human corrections:
  * "Original"  - the learner's sentence with NO correction (a baseline)
  * "AI"        - the sentence after the T5 model corrected it
If the AI is useful, its scores must be clearly higher than the baseline.

Usage:
    python evaluate.py              # all 748 sentences (~5-15 min on a laptop CPU)
    python evaluate.py --limit 100  # quick test on 100 sentences

Outputs (in results/):
    metrics.json            all numbers (also shown in the app's "How It Works" tab)
    score_comparison.png    bar chart: original vs AI
    edit_categories.png     which kinds of mistakes the AI fixed most
    examples.csv            every sentence with its AI correction and scores
"""

import argparse
import csv
import json
import os
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import grammar_model
from feedback import CATEGORY_COLORS, find_edits, similarity
from metrics import corpus_gleu, exact_match
from quiz import detokenize

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_test_set(limit=None):
    from datasets import load_dataset

    data = load_dataset("jhu-clsp/jfleg", split="test")
    sources, references = [], []
    for row in data:
        refs = [detokenize(c) for c in row["corrections"] if c.strip()]
        if refs:
            sources.append(detokenize(row["sentence"]))
            references.append(refs)
    if limit:
        sources, references = sources[:limit], references[:limit]
    return sources, references


def best_similarity(sentence, refs):
    return max(similarity(sentence, r) for r in refs)


def plot_comparison(m, path):
    labels = ["GLEU score", "Similarity to\nteacher correction", "Exact match with\na teacher"]
    before = [m["gleu_source"], m["similarity_source"], m["exact_source"]]
    after = [m["gleu_model"], m["similarity_model"], m["exact_model"]]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar([i - 0.2 for i in x], [v * 100 for v in before], 0.4, label="Original (no correction)", color="#94a3b8")
    b2 = ax.bar([i + 0.2 for i in x], [v * 100 for v in after], 0.4, label="After AI correction", color="#6366f1")
    ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title(f"JFLEG test set ({m['num_sentences']} learner sentences)")
    ax.legend(loc="upper center", ncol=2, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_categories(counts, path):
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=[CATEGORY_COLORS.get(k, "#475569") for k, _ in items])
    ax.set_xlabel("Number of edits")
    ax.set_title("Types of corrections made by the AI")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the grammar model on JFLEG.")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N sentences")
    args = parser.parse_args()

    print("Loading JFLEG test set...")
    sources, references = load_test_set(args.limit)
    print(f"{len(sources)} sentences. Correcting with the AI model...")

    grammar_model.load_model()
    start = time.time()
    corrected = []
    for i in range(0, len(sources), 32):
        corrected.extend(grammar_model.correct_sentences(sources[i:i + 32]))
        print(f"  {len(corrected)}/{len(sources)}", end="\r", flush=True)
    seconds = time.time() - start
    print()

    sim_before = [best_similarity(s, r) for s, r in zip(sources, references)]
    sim_after = [best_similarity(c, r) for c, r in zip(corrected, references)]
    n = len(sources)

    metrics = {
        "dataset": "JFLEG test split (jhu-clsp/jfleg)",
        "model": grammar_model.MODEL_NAME,
        "num_sentences": n,
        "gleu_source": corpus_gleu(sources, references),
        "gleu_model": corpus_gleu(corrected, references),
        "similarity_source": sum(sim_before) / n,
        "similarity_model": sum(sim_after) / n,
        "exact_source": sum(exact_match(s, r) for s, r in zip(sources, references)) / n,
        "exact_model": sum(exact_match(c, r) for c, r in zip(corrected, references)) / n,
        "changed_rate": sum(similarity(s, c) < 1.0 for s, c in zip(sources, corrected)) / n,
        "improved_rate": sum(a > b for a, b in zip(sim_after, sim_before)) / n,
        "worse_rate": sum(a < b for a, b in zip(sim_after, sim_before)) / n,
        "seconds_per_sentence": seconds / n,
        "device": grammar_model._device,
    }

    categories = Counter()
    for s, c in zip(sources, corrected):
        categories.update(edit.category for edit in find_edits(s, c))
    metrics["edit_categories"] = dict(categories.most_common())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "examples.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original", "ai_correction", "closest_teacher_correction",
                         "similarity_before", "similarity_after", "result"])
        for s, c, r, b, a in zip(sources, corrected, references, sim_before, sim_after):
            best_ref = max(r, key=lambda ref: similarity(c, ref))
            result = "improved" if a > b else "worse" if a < b else "same"
            writer.writerow([s, c, best_ref, f"{b:.3f}", f"{a:.3f}", result])

    plot_comparison(metrics, os.path.join(RESULTS_DIR, "score_comparison.png"))
    if categories:
        plot_categories(categories, os.path.join(RESULTS_DIR, "edit_categories.png"))

    print(f"\n{'Metric':<38}{'Original':>10}{'AI':>10}")
    for label, key in [("GLEU", "gleu"), ("Similarity to teacher", "similarity"),
                       ("Exact match with a teacher", "exact")]:
        print(f"{label:<38}{metrics[key + '_source']:>10.3f}{metrics[key + '_model']:>10.3f}")
    print(f"\nChanged {metrics['changed_rate']:.1%} of sentences | improved "
          f"{metrics['improved_rate']:.1%} | made worse {metrics['worse_rate']:.1%}")
    print(f"Results saved in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
