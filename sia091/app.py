
import gradio as gr

from src.answerer import Answerer
from src.chunker import chunk_pages, chunk_layout_items
from src.embedder import Embedder
from src.loader import load_document
from src.loader_layout import load_document_layout
from src.retriever import Retriever


embedder = Embedder()
answerer = Answerer()


def process_document(file_path, chunk_size, overlap, loader_mode):

    if file_path is None:
        return None, "Please upload a PDF or TXT file first."

    try:
        if loader_mode == "Layout-aware Loader":
            pages = load_document_layout(file_path)

            chunks = chunk_layout_items(
                pages,
                int(chunk_size),
                int(overlap)
            )
        else:
            pages = load_document(file_path)

            chunks = chunk_pages(
                pages,
                int(chunk_size),
                int(overlap)
            )

        retriever = Retriever(embedder)
        retriever.build(chunks)

    except (ValueError, FileNotFoundError) as error:
        return None, f"Error: {error}"

    return (
        retriever,
        f"Ready: {len(pages)} page/item(s) split into "
        f"{len(chunks)} chunks using {loader_mode}."
    )


def format_hits(hits):

    lines = []

    for number, hit in enumerate(hits, start=1):
        preview = hit["text"][:120].replace("\n", " ")

        lines.append(
            f"{number}. Page {hit['page']} | "
            f"Score: {hit['score']:.2f} | {preview}..."
        )

    return "\n\n".join(lines)


def ask_question(question, retriever, top_k):

    if retriever is None:
        return "Please process a document first.", "", "", ""

    if not question or not question.strip():
        return "Please enter a question.", "", "", ""

    try:
        hits = retriever.search(question, int(top_k))
        result = answerer.answer(question, hits)

    except (ValueError, RuntimeError) as error:
        return f"Error: {error}", "", "", ""

    retrieved = format_hits(hits)

    # Temporary QA candidate diagnostics
    candidates = answerer.candidates(question, hits)

    candidate_lines = []
    for item in candidates:
        candidate_lines.append(
            f"Page {item['page']} | "
            f"QA Probability: {item['prob']:.4f} | "
            f"QA Logit: {item['logit']:.4f} | "
            f"Retrieval: {item['retrieval']:.4f} | "
            f"Answer: {item['answer']}"
        )

    retrieved += "\n\n--- QA CANDIDATES ---\n"
    retrieved += "\n".join(candidate_lines)


    if result["page"] is None:
        return result["answer"], "", "", retrieved

    source = (
        f"Page {result['page']} | "
        f"Confidence: {result['score']:.2f}"
    )

    return (
        result["answer"],
        source,
        result["context"],
        retrieved
    )


CSS = """
* {
    font-family: "Times New Roman", Times, serif !important;
}

body {
    background: #f5f6f8 !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 12px !important;
    background: #f5f6f8 !important;
}

/* Main layout */

#app-shell {
    gap: 20px !important;
    align-items: flex-start !important;
}

/* Sidebar white box */

#sidebar {
    background: #ffffff !important;
    color: #17243a !important;
    border: 1px solid #e1e6ee !important;
    border-radius: 12px !important;
    padding: 24px !important;
    margin: 5px !important;
    min-height: auto !important;
    box-shadow: 0 3px 14px rgba(20, 40, 70, 0.06);
}

#sidebar .block,
#sidebar .wrap,
#sidebar .form,
#sidebar .container {
    background: transparent !important;
    border-color: #e1e6ee !important;
    box-shadow: none !important;
}

#sidebar * {
    font-family: "Times New Roman", Times, serif !important;
}

/* Brand */

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 30px;
}

.brand-icon {
    width: 42px;
    height: 42px;
    display: grid;
    place-items: center;
    border-radius: 9px;
    background: #F97316 !important;
    color: #ffffff !important;
    font-size: 22px;
    font-weight: bold;
}

.brand-name {
    font-size: 21px;
    font-weight: bold;
    color: #17243a !important;
}

.brand-sub {
    color: #64748b !important;
    font-size: 12px;
    margin-top: 3px;
}

/* Sidebar headings */

.side-label {
    color: #EA580C !important;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 0.7px;
    margin: 22px 0 11px;
}

.side-note {
    color: #52627a !important;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 12px;
}

/* Sidebar text */

#sidebar label,
#sidebar p,
#sidebar span {
    color: #26364d !important;
    font-size: 14px !important;
}

/* File upload */

#sidebar [data-testid="file-upload"] {
    background: #ffffff !important;
    border: 1px dashed #cbd5e1 !important;
    border-radius: 8px !important;
    color: #26364d !important;
}

#sidebar [data-testid="file-upload"] * {
    color: #26364d !important;
}

#sidebar [data-testid="file-upload"]:hover {
    border-color: #F97316 !important;
    background: #fffaf5 !important;
}

/* Radio buttons */

#sidebar input[type="radio"] {
    accent-color: #F97316 !important;
}

#sidebar [role="radiogroup"] {
    background: #ffffff !important;
    gap: 8px !important;
}

#sidebar [role="radio"] {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 7px !important;
    padding: 9px 12px !important;
    color: #17243a !important;
}

#sidebar [role="radio"][aria-checked="true"] {
    background: #fff3e9 !important;
    border: 1px solid #F97316 !important;
}

#sidebar [role="radio"] * {
    color: #17243a !important;
}

/* Inputs */

#sidebar input,
#sidebar textarea,
#sidebar select {
    background: #ffffff !important;
    color: #17243a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
}

/* Sliders */

#sidebar input[type="range"] {
    accent-color: #F97316 !important;
}

/* Process button */

#process-button {
    background: #F97316 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    min-height: 44px !important;
    font-size: 15px !important;
    font-weight: bold !important;
}

#process-button * {
    color: #ffffff !important;
}

#process-button:hover {
    background: #EA580C !important;
}

/* Status */

#status-box textarea {
    background: #fffaf5 !important;
    color: #26364d !important;
    border: 1px solid #f3d5bf !important;
    line-height: 1.5 !important;
}

/* Main workspace */

#workspace {
    padding: 28px 4% !important;
    background: #f5f6f8 !important;
    min-width: 0;
}

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 27px;
}

.eyebrow {
    color: #EA580C !important;
    font-size: 13px;
    font-weight: bold;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

.page-title {
    color: #17243a !important;
    font-size: 31px;
    font-weight: bold;
}

.page-subtitle {
    color: #64748b !important;
    font-size: 14px;
    margin-top: 6px;
}

.status-pill {
    padding: 8px 13px;
    border: 1px solid #f3d5bf;
    background: #ffffff;
    color: #EA580C !important;
    border-radius: 20px;
    font-size: 12px;
    white-space: nowrap;
}

/* Main content boxes */

#question-panel,
#answer-panel,
#source-panel {
    background: #ffffff !important;
    border: 1px solid #e1e6ee !important;
    border-radius: 11px !important;
    padding: 21px !important;
    box-shadow: 0 3px 14px rgba(20, 40, 70, 0.035);
}

.section-heading {
    color: #17243a !important;
    font-size: 19px;
    font-weight: bold;
    margin-bottom: 5px;
}

.section-caption {
    color: #64748b !important;
    font-size: 13px;
    margin-bottom: 15px;
}

/* Question and answer fields */

#question-panel textarea,
#answer-panel textarea,
#source-panel textarea {
    background: #ffffff !important;
    color: #17243a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 7px !important;
    line-height: 1.7 !important;
    font-size: 15px !important;
}

#question-panel label,
#answer-panel label,
#source-panel label {
    color: #344761 !important;
    font-size: 14px !important;
    font-weight: bold !important;
}

/* Generate answer button */

#ask-button {
    background: #F97316 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    min-height: 45px !important;
    font-size: 15px !important;
    font-weight: bold !important;
}

#ask-button * {
    color: #ffffff !important;
}

#ask-button:hover {
    background: #EA580C !important;
}

/* Retrieval details */

#workspace .accordion {
    background: #ffffff !important;
    border: 1px solid #e1e6ee !important;
    border-radius: 9px !important;
}

#workspace .accordion summary {
    color: #17243a !important;
    font-size: 14px !important;
}

/* Footer */

.footer {
    color: #64748b !important;
    text-align: center;
    font-size: 12px;
    margin-top: 28px;
}

footer {
    display: none !important;
}

/* Responsive design */

@media (max-width: 850px) {
    #sidebar {
        margin: 4px !important;
        padding: 18px !important;
    }

    #workspace {
        padding: 20px 12px !important;
    }

    .page-title {
        font-size: 25px;
    }

    .topbar {
        flex-wrap: wrap;
    }
}
"""


with gr.Blocks(
    title="DocuMind | Document AI",
    theme=gr.themes.Base(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css=CSS
) as demo:

    retriever_state = gr.State(None)

    with gr.Row(
        elem_id="app-shell",
        equal_height=False
    ):

        # Sidebar

        with gr.Column(
            scale=1,
            min_width=280,
            elem_id="sidebar"
        ):

            gr.HTML(
                """
                <div class="brand">
                    <div class="brand-icon">D</div>
                    <div>
                        <div class="brand-name">DocuMind</div>
                        <div class="brand-sub">
                            INTELLIGENT DOCUMENT AI
                        </div>
                    </div>
                </div>
                """
            )

            gr.HTML(
                '<div class="side-label">WORKSPACE</div>'
            )

            gr.Markdown(
                """
                **01 - Document Setup**

                <div class="side-note">
                Upload a document and configure how the system
                extracts and indexes its content.
                </div>
                """
            )

            gr.HTML(
                '<div class="side-label">UPLOAD DOCUMENT</div>'
            )

            file_input = gr.File(
                label="PDF or TXT",
                file_types=[".pdf", ".txt"],
                type="filepath"
            )

            gr.HTML(
                '<div class="side-label">EXTRACTION METHOD</div>'
            )

            loader_mode = gr.Radio(
                choices=[
                    "Standard Loader",
                    "Layout-aware Loader"
                ],
                value="Standard Loader",
                label="PDF extraction"
            )

            gr.HTML(
                '<div class="side-label">CHUNK CONFIGURATION</div>'
            )

            chunk_size = gr.Slider(
                minimum=20,
                maximum=300,
                value=40,
                step=10,
                label="Chunk size (words)"
            )

            overlap = gr.Slider(
                minimum=0,
                maximum=100,
                value=10,
                step=5,
                label="Overlap (words)"
            )

            process_button = gr.Button(
                "Process Document",
                elem_id="process-button"
            )

            gr.HTML(
                '<div class="side-label">SYSTEM STATUS</div>'
            )

            status = gr.Textbox(
                label="Processing status",
                value="Waiting for document...",
                interactive=False,
                lines=3,
                elem_id="status-box"
            )

            gr.HTML(
                """
                <div style="
                    border-top:1px solid #e1e6ee;
                    margin-top:25px;
                    padding-top:15px;
                    color:#64748b;
                    font-size:13px;
                    line-height:1.8;">
                    DOCUMENT INTELLIGENCE<br>
                    AI-powered retrieval workspace
                </div>
                """
            )

        # Main workspace

        with gr.Column(
            scale=3,
            min_width=400,
            elem_id="workspace"
        ):

            gr.HTML(
                """
                <div class="topbar">
                    <div>
                        <div class="eyebrow">
                            AI DOCUMENT WORKSPACE
                        </div>
                        <div class="page-title">
                            Document Q&amp;A
                        </div>
                        <div class="page-subtitle">
                            Ask questions. Find answers.
                            Explore your documents.
                        </div>
                    </div>
                    <div class="status-pill">
                        AI SYSTEM
                    </div>
                </div>
                """
            )

            with gr.Group(elem_id="question-panel"):

                gr.HTML(
                    """
                    <div class="section-heading">
                        Ask your document
                    </div>
                    <div class="section-caption">
                        Enter a question and retrieve information
                        directly from your uploaded document.
                    </div>
                    """
                )

                question = gr.Textbox(
                    label="Your question",
                     
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=10,
                    step=1,
                    label="Retrieval depth: number of chunks"
                )

                ask_button = gr.Button(
                    "Generate Answer",
                    elem_id="ask-button"
                )

            with gr.Group(elem_id="answer-panel"):

                gr.HTML(
                    """
                    <div class="section-heading">
                        Answer
                    </div>
                    <div class="section-caption">
                        Information extracted from the document.
                    </div>
                    """
                )

                answer = gr.Textbox(
                    label="Generated response",
                    lines=6,
                    interactive=False
                )

            with gr.Group(elem_id="source-panel"):

                gr.HTML(
                    """
                    <div class="section-heading">
                        Sources and Evidence
                    </div>
                    <div class="section-caption">
                        Review the original document content
                        supporting the answer.
                    </div>
                    """
                )

                source = gr.Textbox(
                    label="Source reference",
                    interactive=False,
                    lines=1
                )

                context = gr.Textbox(
                    label="Original source text",
                    lines=6,
                    interactive=False
                )

            with gr.Accordion(
                "Retrieval Details",
                open=False
            ):

                gr.Markdown(
                    "Retrieved document chunks and similarity scores."
                )

                retrieved = gr.Textbox(
                    label="Retrieved chunks",
                    lines=10,
                    interactive=False
                )

            gr.HTML(
                """
                <div class="footer">
                    DOCUMIND | INTELLIGENT DOCUMENT QUESTION ANSWERING
                </div>
                """
            )

    process_button.click(
        fn=process_document,
        inputs=[
            file_input,
            chunk_size,
            overlap,
            loader_mode
        ],
        outputs=[
            retriever_state,
            status
        ]
    )

    ask_button.click(
        fn=ask_question,
        inputs=[
            question,
            retriever_state,
            top_k
        ],
        outputs=[
            answer,
            source,
            context,
            retrieved
        ]
    )


if __name__ == "__main__":
    demo.launch()
