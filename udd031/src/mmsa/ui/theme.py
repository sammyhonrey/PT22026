"""Gradio theme and design tokens.

The visual language is "holographic clinic": deep navy glass panels, one cyan/violet
accent pair, monospace numerals, generous spacing. The goal is that the application
looks like a 2026 research instrument rather than a default Gradio demo, because the
interface is the first thing an examiner sees.

Gradio's theme API has changed across minor versions, so the token overrides are
applied defensively: if any keyword is rejected the plain base theme is returned
and the custom CSS (which carries most of the appearance) still applies.
"""

from __future__ import annotations

import gradio as gr

__all__ = ["TOKENS", "build_theme", "load_css"]

TOKENS: dict[str, str] = {
    "bg_deep": "#070B18",
    "bg_mid": "#0B1226",
    "bg_panel": "#101A38",
    "cyan": "#22D3EE",
    "violet": "#A78BFA",
    "magenta": "#F472B6",
    "amber": "#FBBF24",
    "green": "#34D399",
    "rose": "#FB7185",
    "text": "#E8EEF9",
    "muted": "#93A4C3",
    "border": "rgba(255,255,255,0.10)",
    "panel_fill": "rgba(255,255,255,0.045)",
}


def build_theme() -> gr.themes.Base:
    """Return the project theme, falling back to the stock base on API drift."""
    try:
        theme = gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.violet,
            neutral_hue=gr.themes.colors.slate,
            font=["Inter", "Segoe UI", "system-ui", "sans-serif"],
            font_mono=["JetBrains Mono", "Consolas", "monospace"],
            radius_size=gr.themes.sizes.radius_lg,
        )
        return theme.set(
            body_background_fill=f"linear-gradient(160deg, {TOKENS['bg_deep']} 0%, "
            f"{TOKENS['bg_mid']} 45%, {TOKENS['bg_panel']} 100%)",
            body_text_color=TOKENS["text"],
            block_background_fill=TOKENS["panel_fill"],
            block_border_color=TOKENS["border"],
            block_radius="18px",
            block_shadow="0 18px 50px rgba(0,0,0,0.45)",
            button_primary_background_fill=f"linear-gradient(135deg, {TOKENS['cyan']}, {TOKENS['violet']})",
            button_primary_text_color="#050912",
            input_background_fill="rgba(255,255,255,0.04)",
            input_border_color=TOKENS["border"],
        )
    except Exception:  # noqa: BLE001 - theme API drift must not break the app
        return gr.themes.Base()


def load_css(path: str) -> str:
    """Read the stylesheet from disk, returning an empty string if missing."""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""
