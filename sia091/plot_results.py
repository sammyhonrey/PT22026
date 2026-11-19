
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_FILE = "eval/controlled_model_comparison.csv"
FIGURE_DIR = Path("eval/figures")

MODELS = ["DistilBERT", "RoBERTa"]

STRATEGIES = [
    "retrieval_first",
    "qa_prob",
    "qa_logit",
    "retrieval_x_prob",
    "logit_plus_retrieval",
]

STRATEGY_LABELS = {
    "retrieval_first": "Retrieval First",
    "qa_prob": "QA Probability",
    "qa_logit": "QA Logit",
    "retrieval_x_prob": "Retrieval x Probability",
    "logit_plus_retrieval": "Logit + Retrieval",
}


def read_rows():
    path = Path(RESULTS_FILE)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    with open(path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("The results CSV is empty.")

    numeric_fields = [
        "chunk_size",
        "overlap",
        "top_k",
        "questions",
        "recall_at_k",
        "permissive_accuracy",
        "exact_match",
        "token_f1",
    ]

    for row in rows:
        for field in numeric_fields:
            row[field] = float(row[field])

    return rows


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_recall(rows):
    roberta = [
        row for row in rows
        if row["model"] == "RoBERTa"
        and row["strategy"] == "retrieval_first"
    ]

    chunk_sizes = sorted({
        row["chunk_size"] for row in roberta
    })

    plt.figure(figsize=(7, 5))

    for size in chunk_sizes:
        data = sorted(
            [r for r in roberta if r["chunk_size"] == size],
            key=lambda r: r["top_k"],
        )

        plt.plot(
            [r["top_k"] for r in data],
            [r["recall_at_k"] for r in data],
            marker="o",
            label=f"Chunk size {int(size)}",
        )

    plt.xlabel("Top-K")
    plt.ylabel("Evidence Recall@K")
    plt.title("Retrieval Performance")
    plt.xticks([1, 3, 5])
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()

    save_figure("controlled_recall_vs_topk.png")


def plot_strategies(rows):
    roberta = [
        row for row in rows
        if row["model"] == "RoBERTa"
    ]

    chunk_sizes = sorted({
        row["chunk_size"] for row in roberta
    })

    figure, axes = plt.subplots(
        1,
        len(chunk_sizes),
        figsize=(16, 5),
        sharey=True,
    )

    for axis, size in zip(axes, chunk_sizes):
        data = [
            row for row in roberta
            if row["chunk_size"] == size
        ]

        for strategy in STRATEGIES:
            strategy_rows = sorted(
                [
                    row for row in data
                    if row["strategy"] == strategy
                ],
                key=lambda row: row["top_k"],
            )

            axis.plot(
                [r["top_k"] for r in strategy_rows],
                [r["exact_match"] for r in strategy_rows],
                marker="o",
                label=STRATEGY_LABELS[strategy],
            )

        axis.set_title(f"Chunk size {int(size)}")
        axis.set_xlabel("Top-K")
        axis.set_xticks([1, 3, 5])
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Exact Match")
    axes[-1].legend(fontsize=7, loc="lower right")

    figure.suptitle(
        "Ranking Strategy Comparison (RoBERTa)"
    )

    save_figure("controlled_ranking_strategies.png")


def plot_models(rows):
    selected = [
        row for row in rows
        if row["strategy"] == "qa_logit"
    ]

    distilbert = sorted(
        [r for r in selected if r["model"] == "DistilBERT"],
        key=lambda r: (r["chunk_size"], r["top_k"]),
    )

    roberta_map = {
        (r["chunk_size"], r["top_k"]): r
        for r in selected
        if r["model"] == "RoBERTa"
    }

    labels = [
        f"{int(r['chunk_size'])}/{int(r['top_k'])}"
        for r in distilbert
    ]

    positions = list(range(len(distilbert)))
    width = 0.4

    plt.figure(figsize=(11, 5))

    plt.bar(
        [p - width / 2 for p in positions],
        [r["exact_match"] for r in distilbert],
        width,
        label="DistilBERT",
    )

    plt.bar(
        [p + width / 2 for p in positions],
        [
            roberta_map[(r["chunk_size"], r["top_k"])]["exact_match"]
            for r in distilbert
        ],
        width,
        label="RoBERTa",
    )

    plt.xticks(positions, labels, rotation=30)
    plt.xlabel("Chunk size / Top-K")
    plt.ylabel("Exact Match")
    plt.title("DistilBERT vs RoBERTa (QA Logit)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    save_figure("controlled_model_comparison.png")


def plot_best_configurations(rows):
    best_results = []

    for model in MODELS:
        model_rows = [
            row for row in rows
            if row["model"] == model
        ]

        best = max(
            model_rows,
            key=lambda row: row["exact_match"],
        )

        best_results.append(best)

        print(f"\nBest configuration: {model}")
        print("Chunk size:", int(best["chunk_size"]))
        print("Top-K:", int(best["top_k"]))
        print("Strategy:", best["strategy"])
        print("Exact Match:", best["exact_match"])
        print("Token F1:", best["token_f1"])

    positions = [0, 1]
    width = 0.35

    plt.figure(figsize=(7, 5))

    plt.bar(
        [p - width / 2 for p in positions],
        [r["exact_match"] for r in best_results],
        width,
        label="Exact Match",
    )

    plt.bar(
        [p + width / 2 for p in positions],
        [r["token_f1"] for r in best_results],
        width,
        label="Token F1",
    )

    plt.xticks(positions, MODELS)
    plt.ylabel("Score")
    plt.title("Best Configuration: Model Performance")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    save_figure("controlled_best_configurations.png")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_rows()

    print("Total experiment rows:", len(rows))

    plot_recall(rows)
    plot_strategies(rows)
    plot_models(rows)
    plot_best_configurations(rows)

    print("\nAll 4 graphs created successfully.")

    for path in sorted(FIGURE_DIR.glob("controlled_*.png")):
        print(path)


if __name__ == "__main__":
    main()
