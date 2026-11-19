"""
Optional Gradio interface for VoiceForge.

This file is an alternative front-end for the same application core used by
``app.py``.  It requires Gradio, which is **not** part of the base
requirements because the project must run without it:

    pip install gradio
    python app_gradio.py

Note: this interface could not be executed in the development environment
because Gradio was not installed and the machine had no network access to
install it.  ``app.py`` is the verified interface.  The code below uses the
stable Gradio 4/5 Blocks API, but treat it as untested until you run it.
"""

from __future__ import annotations

import argparse

try:
    import gradio as gr
except ModuleNotFoundError:  # pragma: no cover - depends on the machine
    raise SystemExit(
        "Gradio is not installed.\n"
        "Install it with:  pip install gradio\n"
        "Then run:         python app_gradio.py\n"
        "The dependency-free interface 'python app.py' works without it."
    )

from voiceforge.app_core import AppError, VoiceForgeApp


def build_interface(app: VoiceForgeApp | None = None):
    """Create the Gradio Blocks interface."""

    app = app or VoiceForgeApp()

    def generate_speech(text: str):
        try:
            result = app.synthesize(text)
        except AppError as error:
            # gr.Error shows the message as a red pop-up in the interface.
            raise gr.Error(str(error)) from error

        message = f"Generated {result.duration_seconds} s of audio."

        if result.padded:
            message += " The model output was very short and was padded."
        if result.truncated:
            message += " The output reached the maximum length and was cut."

        return result.path, message

    with gr.Blocks(title="VoiceForge") as interface:
        gr.Markdown(
            "# VoiceForge\n"
            "A custom English text-to-speech system built from scratch with "
            "PyTorch and trained on LJSpeech. No pretrained TTS model is used."
        )

        gr.Markdown(
            "> **Experimental model.** The student-built acoustic model is "
            "small and trained on a limited data subset. The audio is buzzy "
            "and is **not expected to be fully intelligible**."
        )

        text_input = gr.Textbox(
            label="English text",
            placeholder="Hello, welcome to VoiceForge.",
            lines=3,
            max_lines=8,
        )

        generate_button = gr.Button("Generate Speech", variant="primary")

        audio_output = gr.Audio(
            label="Generated speech",
            type="filepath",
        )

        status_output = gr.Textbox(
            label="Status",
            interactive=False,
        )

        generate_button.click(
            generate_speech,
            inputs=text_input,
            outputs=[audio_output, status_output],
        )

        text_input.submit(
            generate_speech,
            inputs=text_input,
            outputs=[audio_output, status_output],
        )

    return interface


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the VoiceForge Gradio app.")
    parser.add_argument("--checkpoint", default="checkpoints/voiceforge_attn.pt")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    app = VoiceForgeApp(checkpoint_path=args.checkpoint)

    try:
        app.load()
    except AppError as error:
        print(f"ERROR: {error}")
        return 1

    build_interface(app).launch(
        server_port=args.port,
        share=args.share,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
