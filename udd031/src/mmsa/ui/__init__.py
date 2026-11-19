"""Interface layer: theme, fragments and page assembly.

Importing this package does **not** import Gradio. ``app_ui`` pulls it in lazily so
that the plotting and narration modules stay importable in a bare numerical
environment (the report build, for example, never needs a browser framework).
"""

from __future__ import annotations

from .theme import TOKENS, build_theme, load_css

__all__ = ["TOKENS", "build_theme", "load_css", "build_demo"]


def build_demo(*args: object, **kwargs: object) -> object:
    """Lazy re-export of :func:`mmsa.ui.app_ui.build_demo`."""
    from .app_ui import build_demo as _build

    return _build(*args, **kwargs)
