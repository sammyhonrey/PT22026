"""Text Summarization App (Gradio + Hugging Face Transformers).

Run:  python app.py
"""
import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Models the user can choose from (all from the Hugging Face Model Hub)
MODELS = {
    "DistilBART (fast, good quality)": "sshleifer/distilbart-cnn-12-6",
    "T5-small (very small, fastest)": "t5-small",
    "BART-large-CNN (best quality, slow)": "facebook/bart-large-cnn",
}

_cache = {}  # keeps loaded models in memory so we load each only once


def load_model(model_id: str):
    """Load tokenizer + model (and reuse them if already loaded)."""
    if model_id not in _cache:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        _cache[model_id] = (tokenizer, model)
    return _cache[model_id]


def generate_summary(text: str, model_id: str, min_words: int, max_words: int) -> str:
    """Run the model and return the summary text."""
    tokenizer, model = load_model(model_id)
    # T5 models need a task prefix; BART models do not
    if "t5" in model_id.lower():
        text = "summarize: " + text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    # words -> tokens (roughly 1 word = 1.3 tokens)
    ids = model.generate(
        **inputs,
        min_length=int(min_words * 1.3),
        max_length=int(max_words * 1.3),
        num_beams=4,
        do_sample=False,
        early_stopping=True,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def summarize(text: str, model_name: str, min_words: int, max_words: int):
    """Return a summary and some simple statistics."""
    # --- input checks (error handling) ---
    if not text or not text.strip():
        return "Please paste some text first.", ""
    n_words = len(text.split())
    if n_words < 30:
        return "The text is too short to summarize (write at least 30 words).", ""
    if min_words >= max_words:
        return "Minimum length must be smaller than maximum length.", ""

    try:
        summary = generate_summary(text, MODELS[model_name], min_words, max_words)
    except Exception as e:  # e.g. no internet, out of memory
        return f"Error while summarizing: {e}", ""

    s_words = len(summary.split())
    stats = (f"Original: {n_words} words | Summary: {s_words} words | "
             f"Compression: {100 - round(100 * s_words / n_words)}%")
    return summary, stats


with gr.Blocks(title="Text Summarizer") as demo:
    gr.Markdown("# Text Summarization\nPaste a long text and get a short summary.")
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(label="Input text", lines=12,
                                 placeholder="Paste an article here...")
            model_in = gr.Dropdown(list(MODELS), value=list(MODELS)[0], label="Model")
            min_in = gr.Slider(10, 100, value=30, step=5, label="Minimum summary length (words)")
            max_in = gr.Slider(30, 250, value=100, step=10, label="Maximum summary length (words)")
            btn = gr.Button("Summarize", variant="primary")
        with gr.Column():
            out = gr.Textbox(label="Summary", lines=8)
            info = gr.Textbox(label="Statistics")
    btn.click(summarize, [text_in, model_in, min_in, max_in], [out, info])

if __name__ == "__main__":
    demo.launch(share=True)
