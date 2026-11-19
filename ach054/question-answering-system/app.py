"""
app.py - Gradio user interface for ContextAI (Question Answering).

Run with:  python app.py
The ML logic lives in qa_engine.py and the styling lives in styles.py.
"""

import gradio as gr

from qa_engine import DEFAULT_MODEL, MIN_CONFIDENCE, load_model, predict
from styles import CUSTOM_CSS

EMPTY_HISTORY = [["No questions asked yet.", "", ""]]

NO_ANSWER_MESSAGE = (
    "I couldn't find a confident answer in the provided text. "
    "Try rephrasing your question or adding more context."
)

EXAMPLE_CONTEXT = (
    "Artificial Intelligence is a branch of computer science. "
    "Machine learning allows computers to learn from data. "
    "Python is a popular programming language for AI development. "
    "AI is used in healthcare, education, transportation, and business."
)

# Load the model once when the app starts so the first question is fast.
load_model(DEFAULT_MODEL)


# ------------------------------------------------------------
# Application logic
# ------------------------------------------------------------
def answer_question(context, question, history):
    """Run the QA model and return everything the interface needs to update."""
    context = (context or "").strip()
    question = (question or "").strip()
    history = history or []

    # 1. Validate the inputs.
    if not context:
        return ("Please enter some context text or upload a TXT file.",
                "0%", confidence_html(0), history_table(history), history)
    if not question:
        return ("Please enter a question.",
                "0%", confidence_html(0), history_table(history), history)

    # 2. Ask the model; never let an exception crash the interface.
    try:
        result = predict(question, context)
    except Exception as error:  # noqa: BLE001 - we want to catch everything here
        print(f"System error: {error}")
        return ("An error occurred while processing your question. Please try again.",
                "0%", confidence_html(0), history_table(history), history)

    answer, score = result["answer"], result["score"]
    confidence = f"{score:.2%}"

    # 3. Handle "no answer" cases (empty span or very low confidence).
    if score < MIN_CONFIDENCE or not answer:
        answer = NO_ANSWER_MESSAGE
        score = 0.0  # don't show "no answer" certainty as answer confidence
        confidence = "No answer in text"

    # 4. Save to history and return all five outputs.
    history = (history + [[question, answer, confidence]])[-10:]  # keep latest 10
    return answer, confidence, confidence_html(score), history_table(history), history

def confidence_html(score):
    """Return a styled confidence bar for the estimated model score."""
    score = max(0.0, min(float(score), 1.0))
    percent = score * 100

    if percent >= 70:
        color = "#55A9D1"
        label = "High"
    elif percent >= 35:
        color = "#87CEEB"
        label = "Moderate"
    else:
        color = "#800020"
        label = "Low"

    return f"""
    <div class="confidence-wrap">
        <div class="confidence-top">
            <span class="confidence-caption">Model confidence estimate</span>
            <span class="confidence-value" style="color:{color}">{percent:.1f}% · {label}</span>
        </div>
        <div class="confidence-track">
            <div class="confidence-fill"
                 style="width:{percent:.1f}%; background:{color};"></div>
        </div>
        <p class="confidence-note">
            Confidence is an estimate from the model, not a guarantee of correctness.
        </p>
    </div>
    """


def history_table(history):
    if not history:
        return EMPTY_HISTORY
    return history


def load_txt_file(file_path):
    """Read an uploaded TXT file into the context box."""
    if not file_path:
        return gr.update()
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
    except UnicodeDecodeError:
        gr.Warning("Could not read this file as UTF-8 text. Please upload a UTF-8 TXT file.")
        return gr.update()
    except Exception as error:  # noqa: BLE001
        gr.Warning(f"Could not load file: {error}")
        return gr.update()

    if not content.strip():
        gr.Warning("The uploaded TXT file is empty.")
        return gr.update()
    return content


def clear_all():
    """Reset every input, output and the session history."""
    return "", "", "", "0%", confidence_html(0), EMPTY_HISTORY, [], None


# ------------------------------------------------------------
# Gradio interface
# ------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="ContextAI | Question Answering") as app:
    history_state = gr.State([])

    gr.HTML(
        """
        <div id="hero">
            <h1>🤖 ContextAI</h1>
            <div class="subtitle">Intelligent Question Answering System</div>
            <div class="description">
                Ask questions and discover answers directly from your text.
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # Context / input panel
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("## 📄 Context Input")
            gr.Markdown(
                '<div class="panel-description">Paste a paragraph or upload a TXT document.</div>'
            )

            context_input = gr.Textbox(
                label="Your Context",
                placeholder=(
                    "Paste your paragraph here...\n\n"
                    "Example: Artificial Intelligence is a branch of computer science."
                ),
                lines=10,
                elem_id="context-box",
                show_copy_button=True,
            )

            file_input = gr.File(
                label="📁 Upload TXT File",
                file_types=[".txt"],
                type="filepath",
                elem_id="upload-area",
            )
            gr.Markdown(
                '<div class="panel-description">TXT files only · UTF-8 recommended</div>'
            )

            question_input = gr.Textbox(
                label="❓ Your Question",
                placeholder="Ask a question about your text...",
                lines=2,
            )

            with gr.Row(elem_id="button-row"):
                submit_button = gr.Button(
                    "🔍 Find Answer", variant="primary", scale=3
                )
                clear_button = gr.Button(
                    "🧹 Clear All", variant="secondary", scale=1
                )

        # Answer / output panel
        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown("## 💡 AI Response")
            gr.Markdown(
                '<div class="panel-description">Your answer and estimated model confidence.</div>'
            )

            answer_output = gr.Textbox(
                label="AI Answer",
                placeholder="Your answer will appear here after you click Find Answer.",
                lines=10,
                interactive=False,
                elem_id="answer-box",
                show_copy_button=True,
            )

            confidence_output = gr.Textbox(
                label="📊 Estimated Confidence",
                value="0%",
                interactive=False,
            )

            confidence_indicator = gr.HTML(
                value=confidence_html(0),
                elem_id="confidence-panel",
            )

    gr.Markdown("## 💡 Try an Example", elem_id="examples-section")
    gr.Markdown(
        '<div class="panel-description">Select an example to automatically fill the context and question.</div>'
    )

    gr.Examples(
        examples=[
            [EXAMPLE_CONTEXT, "What is machine learning?"],
            [EXAMPLE_CONTEXT, "Which programming language is popular for AI development?"],
            [EXAMPLE_CONTEXT, "Where is AI used?"],
            [EXAMPLE_CONTEXT, "What is Artificial Intelligence?"],
        ],
        inputs=[context_input, question_input],
        label="Example Questions",
    )

    with gr.Column(elem_classes=["panel"], elem_id="history-panel"):
        gr.Markdown("## 📚 Question History")
        gr.Markdown(
            '<div class="panel-description">Your latest 10 questions and answers during this session.</div>'
        )

        history_output = gr.Dataframe(
            headers=["Question", "AI Answer", "Confidence"],
            value=EMPTY_HISTORY,
            datatype=["str", "str", "str"],
            interactive=False,
            wrap=True,
            row_count=(1, "dynamic"),
            col_count=(3, "fixed"),
            elem_id="history-table",
        )

    with gr.Accordion("ℹ️ About ContextAI | How It Works", open=False, elem_id="about-box"):
        gr.Markdown(
            """
       **ContextAI** uses RoBERTa (deepset/roberta-base-squad2), a transformer fine-tuned for
    extractive question answering.
            1. Enter or upload text that contains the information.
            2. Ask a question about that text.
            3. The model selects a likely answer span from the context.
            4. The confidence score is the model's estimated likelihood—not a
               guarantee that the answer is correct.

            **Technology:** Python · Hugging Face Transformers · Gradio
            """
        )

    gr.HTML(
        """
        <div id="footer">
            <span class="brand">🤖 ContextAI</span>
            <span> · Powered by RoBERTa Transformer</span>
            <div>Built with Python, Hugging Face and Gradio</div>
        </div>
        """
    )

    submit_button.click(
        fn=answer_question,
        inputs=[context_input, question_input, history_state],
        outputs=[
            answer_output,
            confidence_output,
            confidence_indicator,
            history_output,
            history_state,
        ],
    )

    question_input.submit(
        fn=answer_question,
        inputs=[context_input, question_input, history_state],
        outputs=[
            answer_output,
            confidence_output,
            confidence_indicator,
            history_output,
            history_state,
        ],
    )

    file_input.change(
        fn=load_txt_file,
        inputs=[file_input],
        outputs=[context_input],
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            context_input,
            question_input,
            answer_output,
            confidence_output,
            confidence_indicator,
            history_output,
            history_state,
            file_input,
        ],
    )


if __name__ == "__main__":
    # GRADIO_SERVER_NAME / GRADIO_SERVER_PORT env vars are respected automatically,
    # so the same file works locally and on Hugging Face Spaces.
    app.launch(show_error=True)
