"""Local Streamlit component for bounded, reactive chart layout identity.

The frame reports only its own width through the supported component protocol.
It never reads parent DOM, uploads, model state, or data and makes no remote call.
"""

from pathlib import Path

from streamlit.components.v1 import declare_component

# A tiny local component (see assets/layout_observer) that tells Python
# how wide the page is. The app uses this to make charts fit small screens.
_observe = declare_component("ecowatt_layout_width", path=str(Path(__file__).resolve().parents[1] / "assets/layout_observer"))


def layout_width() -> int:
    """Return the page width in pixels, or 0 if it is not known yet."""
    width = _observe(key="ecowatt_layout_width", default=0, tab_index=-1, height=0)
    # Only accept a normal whole number. Anything strange becomes 0.
    return width if type(width) is int and 0 <= width <= 16000 else 0
