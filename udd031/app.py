#!/usr/bin/env python
"""Entry point for the Multimodal Mental State Analyser.

    python app.py                 # serve on http://127.0.0.1:7860
    python app.py --port 8000     # different port
    python app.py --share         # temporary public Gradio tunnel

The heavy dependencies (torch, transformers) are imported only when a clip is
analysed, so the page loads instantly and a missing weight download surfaces as a
readable message instead of a stack trace at import time.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="app.py",
        description="Multimodal Mental State Analyser — emotion and stress from speech and its transcript.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=7860, help="port (default: %(default)s)")
    parser.add_argument("--share", action="store_true", help="expose a temporary public Gradio link")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("mmsa.app")

    try:
        from mmsa.config import load_config
        from mmsa.ui import build_demo
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}\n"
            "Install the project requirements first:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001 - configuration errors must be legible
        print(f"Configuration could not be loaded: {exc}", file=sys.stderr)
        return 2

    log.info("building interface for %s", cfg)
    demo = build_demo(cfg)
    log.info("serving on http://%s:%d  (Ctrl+C to stop)", args.host, args.port)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_api=False,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
