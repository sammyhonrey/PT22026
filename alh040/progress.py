"""
progress.py - Tracks the learner's mistakes during a session.

Every time the learner corrects their own writing (Fix tab) or answers a quiz
question (Practice tab), the mistake categories are counted here. The "My
Progress" tab then shows a chart and recommends what to study next.

The progress is kept in memory for the current browser session only
(nothing is saved to disk), which keeps the app simple and private.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feedback import CATEGORY_COLORS, LESSONS


def new_progress() -> dict:
    return {"mistakes": {}, "texts_checked": 0, "quiz_scores": []}


def record_edits(progress: dict, edits: list) -> dict:
    """Count the mistake categories found in the learner's writing."""
    progress = dict(progress, mistakes=dict(progress["mistakes"]))  # copy (Gradio state)
    for edit in edits:
        progress["mistakes"][edit.category] = progress["mistakes"].get(edit.category, 0) + 1
    return progress


def record_text(progress: dict, edits: list) -> dict:
    progress = record_edits(progress, edits)
    progress["texts_checked"] += 1
    return progress


def record_quiz(progress: dict, score: int, missed_edits: list) -> dict:
    progress = record_edits(progress, missed_edits)
    progress["quiz_scores"] = progress["quiz_scores"] + [score]
    return progress


def summary_markdown(progress: dict) -> str:
    mistakes = progress["mistakes"]
    scores = progress["quiz_scores"]
    lines = ["### Your session so far",
             f"- ✏️ Texts checked: **{progress['texts_checked']}**",
             f"- 🎯 Quiz questions answered: **{len(scores)}**"]
    if scores:
        lines.append(f"- 📈 Average quiz score: **{sum(scores) / len(scores):.0f}/100** "
                     f"(best {max(scores)}, latest {scores[-1]})")
    lines.append(f"- 🔍 Mistakes found in total: **{sum(mistakes.values())}**")

    if not mistakes:
        lines.append("\nNo mistakes recorded yet - check some writing or try the quiz!")
        return "\n".join(lines)

    top = sorted(mistakes.items(), key=lambda kv: -kv[1])[:3]
    tips = ["### 📚 What to practise next"]
    for rank, (category, count) in enumerate(top, start=1):
        tips.append(f"**{rank}. {category}** ({count} time{'s' if count > 1 else ''})  \n"
                    f"{LESSONS[category]}")
    return "\n".join(lines) + "\n\n" + "\n\n".join(tips)


def mistakes_chart(progress: dict):
    """Horizontal bar chart of mistake categories (most common at the top)."""
    mistakes = progress["mistakes"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if not mistakes:
        ax.text(0.5, 0.5, "No mistakes recorded yet", ha="center", va="center",
                fontsize=13, color="#64748b")
        ax.axis("off")
        return fig

    items = sorted(mistakes.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    counts = [v for _, v in items]
    colors = [CATEGORY_COLORS.get(k, "#475569") for k in labels]
    ax.barh(labels, counts, color=colors)
    ax.set_xlabel("Number of mistakes")
    ax.set_title("Your mistakes by type")
    ax.xaxis.get_major_locator().set_params(integer=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
