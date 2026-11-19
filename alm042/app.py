"""Launch the VoiceForge speech application.

Examples
--------
Start the interface with the pretrained Kokoro engine (default when
available)::

    python app.py

Force the pretrained engine, or use only the custom model::

    python app.py --engine kokoro
    python app.py --engine voiceforge --checkpoint checkpoints/voiceforge_attn.pt

Pick a different Kokoro voice and open the browser::

    python app.py --voice am_michael --open

The interface runs entirely locally: no audio or text leaves the machine.
"""

from __future__ import annotations

import argparse
import os
import sys

from voiceforge.app_core import (
    DEFAULT_CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    ENGINE_AUTO,
    ENGINE_KOKORO,
    ENGINE_VOICEFORGE,
    AppError,
    VoiceForgeApp,
)
from voiceforge.kokoro_backend import (
    DEFAULT_VOICE,
    KOKORO_VOICES,
    configure_cache,
)
from voiceforge.web_app import run

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the VoiceForge text-to-speech application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--engine",
        default=ENGINE_AUTO,
        choices=[ENGINE_AUTO, ENGINE_KOKORO, ENGINE_VOICEFORGE],
        help="Speech engine. 'auto' prefers Kokoro when installed.",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        choices=list(KOKORO_VOICES),
        help="Kokoro voice (used when the Kokoro engine is active).",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="Compute device. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=32,
        help="Griffin-Lim iterations (quality versus speed).",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the interface in the default browser.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not 1 <= args.port <= 65535:
        print("ERROR: --port must be between 1 and 65535", file=sys.stderr)
        return 1

    # Some Windows accounts cannot write to the default Hugging Face cache,
    # which breaks the Kokoro weight download. Select a writable cache
    # before any Hugging Face import happens.
    explicit_hf_home = bool(os.environ.get("HF_HOME"))
    selected_cache = configure_cache()

    if selected_cache and not explicit_hf_home:
        print(
            f"Hugging Face cache: {selected_cache} "
            "(the default cache location is not writable)"
        )

    app = VoiceForgeApp(
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        griffin_lim_iterations=args.n_iter,
        engine=args.engine,
        kokoro_voice=args.voice,
    )

    # Load the selected engine once at startup so a missing package, missing
    # weights or a broken checkpoint is reported immediately instead of on
    # the first button click.
    engine = app.resolve_engine()

    try:
        if engine == ENGINE_KOKORO:
            backend = app.load_kokoro()
            print(f"Engine: Kokoro-82M (voice {app.kokoro_voice}, "
                  f"{backend.SAMPLE_RATE} Hz)")
        else:
            app.load()
            print(f"Engine: VoiceForge custom model")
            print(f"Checkpoint: {args.checkpoint}")
            print(f"Device: {app.device_description()}")
    except AppError as error:
        if args.engine == ENGINE_AUTO and engine == ENGINE_KOKORO:
            print(f"WARNING: {error}")
            print("Falling back to the VoiceForge custom model.")
            app.set_engine(ENGINE_VOICEFORGE)

            try:
                app.load()
            except AppError as fallback_error:
                print(f"ERROR: {fallback_error}", file=sys.stderr)
                return 1

            print(f"Checkpoint: {args.checkpoint}")
            print(f"Device: {app.device_description()}")
        else:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

    run(
        app,
        host=args.host,
        port=args.port,
        open_browser=args.open,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
