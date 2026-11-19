"""
app.py - GrammarCoach: AI Grammar Fixer & Learning Tool (Gradio web app).

Run:   python app.py      then open http://127.0.0.1:7860 in your browser.

Tabs:
  1. Fix My Writing   - the AI corrects your text, shows each change and explains it
  2. Practice Quiz    - correct real learner sentences yourself and get a score
  3. My Progress      - chart of your most common mistakes + what to study next
  4. How It Works     - the model explained + evaluation results
"""

import json
import os

# Requests to the local Gradio server must never go through a system proxy or VPN;
# otherwise Gradio's start-up check of http://127.0.0.1:7860 fails with HTTP 502.
for _key in ("NO_PROXY", "no_proxy"):
    _value = os.environ.get(_key, "")
    if "127.0.0.1" not in _value:
        os.environ[_key] = ",".join(filter(None, [_value, "localhost,127.0.0.1"]))

import gradio as gr

import grammar_model
import progress as prog
import quiz
from feedback import correctness_score, diff_html, find_edits, lessons_html

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CSS = """
.gc-diff {font-size: 1.15rem; line-height: 2.1; padding: 14px 16px; border-radius: 10px;
          background: var(--block-background-fill); border: 1px solid var(--border-color-primary);}
.gc-del {background: #fee2e2; color: #b91c1c; text-decoration: line-through; padding: 1px 4px; border-radius: 4px;}
.gc-ins {background: #dcfce7; color: #15803d; text-decoration: none; font-weight: 600; padding: 1px 4px; border-radius: 4px;}
.gc-card {padding: 10px 14px; margin: 8px 0; border-radius: 10px; border: 1px solid var(--border-color-primary);
          background: var(--block-background-fill);}
.gc-good {border-color: #16a34a;}
.gc-tag {color: white; font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 999px;}
.gc-lesson {margin-top: 6px; font-size: 0.95rem; opacity: 0.9;}
.gc-score {display: flex; gap: 14px; flex-wrap: wrap;}
.gc-stat {flex: 1; min-width: 140px; padding: 12px; border-radius: 10px; text-align: center;
          border: 1px solid var(--border-color-primary); background: var(--block-background-fill);}
.gc-stat b {display: block; font-size: 1.8rem;}
.gc-question {font-size: 1.25rem; padding: 14px 16px; border-left: 5px solid #6366f1;
              background: var(--block-background-fill); border-radius: 8px;}
"""

FIX_EXAMPLES = [
    ["Yesterday i go to the school and buyed a apple."],
    ["She dont like play football in monday. Her brother have two book and they is very interesting."],
    ["Their going too the park tomorow because the wether is good."],
    ["I am interesting in learn english since three years ago."],
    ["Machine learning are a field of artificial intelligence who allow computer to learn from data."],
]

QUESTIONS, QUIZ_SOURCE = quiz.load_questions()


# ---------------------------------------------------------------------------
# Tab 1: Fix My Writing
# ---------------------------------------------------------------------------
def score_html(score: int, n_fixes: int) -> str:
    color = "#16a34a" if score >= 90 else "#ca8a04" if score >= 70 else "#dc2626"
    return (
        '<div class="gc-score">'
        f'<div class="gc-stat"><b style="color:{color}">{score}%</b>of your words were already correct</div>'
        f'<div class="gc-stat"><b>{n_fixes}</b>fix{"es" if n_fixes != 1 else ""} suggested</div>'
        '<div class="gc-stat"><b style="color:#16a34a">100%</b>after correction</div>'
        "</div>"
    )


def fix_writing(text: str, progress: dict):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Please type or paste some English text first.")
    try:
        corrected = grammar_model.correct_text(text)
    except ValueError as err:          # text too long
        raise gr.Error(str(err))
    except Exception as err:           # model could not be loaded / run
        raise gr.Error(f"The AI model could not run: {err}. "
                       "Check your internet connection the first time (see README).")

    edits = find_edits(text, corrected)
    progress = prog.record_text(progress, edits)
    return (corrected, diff_html(text, corrected),
            score_html(correctness_score(text, corrected), len(edits)),
            lessons_html(edits), progress)


# ---------------------------------------------------------------------------
# Tab 2: Practice Quiz
# ---------------------------------------------------------------------------
def question_markdown(index: int) -> str:
    return f'<div class="gc-question">📝 {QUESTIONS[index]["sentence"]}</div>'


def new_question(current: int):
    index = quiz.pick_question(QUESTIONS, exclude=current)
    # reset: question, answer box (pre-filled with the sentence), feedback, diff, hints, answers
    return (index, question_markdown(index), QUESTIONS[index]["sentence"], "", "", "", "")


def check_answer(index: int, answer: str, progress: dict):
    if not (answer or "").strip():
        raise gr.Error("Type your corrected sentence in the box first.")
    question = QUESTIONS[index]
    result = quiz.grade_answer(question, answer)
    progress = prog.record_quiz(progress, result["score"], result["remaining"])

    if result["remaining"]:
        hints = ("<h4>Still to fix (compared with a teacher's correction):</h4>"
                 + lessons_html(result["remaining"]))
    else:
        hints = lessons_html([])
    teacher = "**Teacher corrections:**\n\n" + "\n".join(f"- {c}" for c in question["corrections"])
    return (quiz.feedback_message(result), diff_html(answer, result["best_ref"]),
            hints, teacher, progress)


def show_ai_answer(index: int):
    try:
        ai = grammar_model.correct_text(QUESTIONS[index]["sentence"])
    except Exception as err:
        raise gr.Error(f"The AI model could not run: {err}")
    return f"**🤖 AI correction:** {ai}"


# ---------------------------------------------------------------------------
# Tab 3: My Progress
# ---------------------------------------------------------------------------
def show_progress(progress: dict):
    return prog.summary_markdown(progress), prog.mistakes_chart(progress)


def reset_progress():
    fresh = prog.new_progress()
    return (fresh,) + show_progress(fresh)


# ---------------------------------------------------------------------------
# Tab 4: How It Works - load evaluation results if evaluate.py has been run
# ---------------------------------------------------------------------------
def evaluation_markdown() -> str:
    path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(path):
        return ("*No evaluation results yet. Run `python evaluate.py` to test the model on "
                "the JFLEG test set - the results and charts will then appear here.*")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    return (
        f"### Evaluation on the JFLEG test set ({m['num_sentences']} learner sentences)\n\n"
        "| Metric | Original text (no correction) | After AI correction |\n"
        "|---|---|---|\n"
        f"| GLEU score (higher = closer to human corrections) | {m['gleu_source']:.3f} | **{m['gleu_model']:.3f}** |\n"
        f"| Similarity to closest teacher correction | {m['similarity_source']:.1%} | **{m['similarity_model']:.1%}** |\n"
        f"| Exactly matches a teacher correction | {m['exact_source']:.1%} | **{m['exact_model']:.1%}** |\n\n"
        f"- Sentences the AI changed: **{m['changed_rate']:.1%}**\n"
        f"- Sentences made worse (less similar to teachers than before): **{m['worse_rate']:.1%}**\n"
        f"- Average time per sentence: **{m['seconds_per_sentence']:.2f} s** on {m['device']}\n"
    )


HOW_IT_WORKS = """
## 🧠 How GrammarCoach works

**1. The AI model - T5 (Text-to-Text Transfer Transformer)**
GrammarCoach uses [`vennify/t5-base-grammar-correction`](https://huggingface.co/vennify/t5-base-grammar-correction),
a **T5-base** model (≈220 million parameters, Google 2020) fine-tuned on the **JFLEG** dataset of learner sentences.

- **Encoder** - reads your whole sentence at once. *Self-attention* lets every word look at every other word,
  so the model knows that in *"Yesterday I go"* the word *go* is linked to *yesterday*.
- **Decoder** - writes the corrected sentence one word-piece at a time, checking the encoder's understanding at every step.
- **Beam search** - instead of picking one word at a time greedily, it keeps the 5 most likely sentences
  and returns the best complete one.

Grammar correction is treated as *translation*: from "incorrect English" → "correct English".

**2. The feedback engine (rule-based)**
The app compares your text with the AI's output word by word (Python `difflib`) and uses grammar rules
(verb lists, article and preposition lists, spelling similarity) to label each change and attach a mini-lesson.

**3. Learning features**
The Practice Quiz uses real learner sentences from JFLEG's *validation* split, and scores your answer
against 4 human teacher corrections. Your mistakes are counted in **My Progress**.

**Limitations:** the AI sometimes changes correct sentences or misses mistakes; mistake labels come from
rules so they are a best guess; it only supports English and works best on single, short sentences.
"""


# ---------------------------------------------------------------------------
# Build the interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="GrammarCoach", theme=gr.themes.Soft(primary_hue="indigo"), css=CSS) as demo:
    progress_state = gr.State(prog.new_progress())
    first_question = quiz.pick_question(QUESTIONS)
    question_state = gr.State(first_question)

    gr.Markdown("# ✍️ GrammarCoach\n### AI grammar fixer that also teaches you *why*")

    with gr.Tabs():
        # --- Tab 1 -----------------------------------------------------------
        with gr.Tab("✏️ Fix My Writing"):
            with gr.Row():
                with gr.Column(scale=1):
                    text_in = gr.Textbox(label="Your English text", lines=6,
                                         placeholder="Type or paste a sentence or paragraph...")
                    fix_btn = gr.Button("Check my grammar", variant="primary")
                    gr.Examples(FIX_EXAMPLES, inputs=text_in, label="Try an example")
                with gr.Column(scale=1):
                    score_out = gr.HTML()
                    gr.Markdown("**Changes** (<span style='color:#b91c1c'>red = removed</span>, "
                                "<span style='color:#15803d'>green = added</span>)")
                    diff_out = gr.HTML()
                    corrected_out = gr.Textbox(label="Corrected text", lines=3,
                                               interactive=False, show_copy_button=True)
            gr.Markdown("### 💡 What you can learn from your mistakes")
            lessons_out = gr.HTML()

            fix_btn.click(fix_writing, inputs=[text_in, progress_state],
                          outputs=[corrected_out, diff_out, score_out, lessons_out, progress_state])

        # --- Tab 2 -----------------------------------------------------------
        with gr.Tab("🎯 Practice Quiz"):
            gr.Markdown(f"Correct this sentence written by an English learner, then check your answer. "
                        f"*(Sentences: {QUIZ_SOURCE}, {len(QUESTIONS)} available)*")
            question_out = gr.HTML(question_markdown(first_question))
            answer_in = gr.Textbox(label="Your corrected sentence", lines=2,
                                   value=QUESTIONS[first_question]["sentence"])
            with gr.Row():
                check_btn = gr.Button("Check my answer", variant="primary")
                ai_btn = gr.Button("🤖 Show AI's correction")
                next_btn = gr.Button("Next sentence ➜")
            feedback_out = gr.Markdown()
            quiz_diff_out = gr.HTML()
            hints_out = gr.HTML()
            with gr.Row():
                teacher_out = gr.Markdown()
                ai_out = gr.Markdown()

            check_btn.click(check_answer, inputs=[question_state, answer_in, progress_state],
                            outputs=[feedback_out, quiz_diff_out, hints_out, teacher_out, progress_state])
            ai_btn.click(show_ai_answer, inputs=question_state, outputs=ai_out)
            next_btn.click(new_question, inputs=question_state,
                           outputs=[question_state, question_out, answer_in, feedback_out,
                                    quiz_diff_out, hints_out, teacher_out]).then(
                lambda: "", outputs=ai_out)

        # --- Tab 3 -----------------------------------------------------------
        with gr.Tab("📊 My Progress") as progress_tab:
            with gr.Row():
                with gr.Column():
                    summary_out = gr.Markdown(prog.summary_markdown(prog.new_progress()))
                    reset_btn = gr.Button("Reset progress")
                chart_out = gr.Plot()
            progress_tab.select(show_progress, inputs=progress_state, outputs=[summary_out, chart_out])
            reset_btn.click(reset_progress, outputs=[progress_state, summary_out, chart_out])

        # --- Tab 4 -----------------------------------------------------------
        with gr.Tab("🔬 How It Works"):
            gr.Markdown(HOW_IT_WORKS)
            gr.Markdown(evaluation_markdown())
            with gr.Row():
                for name in ("score_comparison.png", "edit_categories.png"):
                    path = os.path.join(RESULTS_DIR, name)
                    if os.path.exists(path):
                        gr.Image(path, show_label=False, interactive=False)


if __name__ == "__main__":
    try:
        grammar_model.load_model()   # load once at start so the first click is fast
    except Exception as err:
        print(f"\n⚠️  Could not load the model yet: {err}\n"
              "   The first run needs internet access to download it (see README: "
              "'Running in mainland China').\n   The app will still start.\n")
    demo.launch()
