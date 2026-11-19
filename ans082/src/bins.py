"""Bin advice for SmartSort AI: from a predicted class to the right bin, a tip and a colour.

The mapping follows China's national four-category household-waste scheme (recyclable, kitchen
waste, hazardous waste, other waste). Local rules can differ between cities, so the tips are
general advice only.

This file does not import torch, so the app and the tests can use it without loading the model.
"""

RECYCLABLE = "Recyclable"
KITCHEN = "Kitchen waste"
HAZARDOUS = "Hazardous waste"
OTHER = "Other waste"

# Display colours for the four bins (blue, green, red, grey).
BIN_COLORS = {
    RECYCLABLE: "#2E6FD8",
    KITCHEN: "#3A9D4A",
    HAZARDOUS: "#D64541",
    OTHER: "#6B7280",
}

_ZH = {RECYCLABLE: "可回收物", KITCHEN: "厨余垃圾", HAZARDOUS: "有害垃圾", OTHER: "其他垃圾"}


def _entry(bin_name: str, tip: str) -> dict:
    return {"bin": bin_name, "zh": _ZH[bin_name], "color": BIN_COLORS[bin_name], "tip": tip}


# One entry for each of the 10 model classes (same names as data.CLASSES).
BINS = {
    "plastic": _entry(RECYCLABLE, "Empty the bottle, remove the cap, flatten it."),
    "paper": _entry(RECYCLABLE, "Keep it dry and clean; remove plastic parts."),
    "cardboard": _entry(RECYCLABLE, "Flatten the box; remove tape if possible."),
    "metal": _entry(RECYCLABLE, "Rinse cans; do not crush sharp edges."),
    "glass": _entry(RECYCLABLE, "Rinse; wrap broken glass safely before disposal."),
    "clothes": _entry(RECYCLABLE, "Clean clothes can go to a textile collection point."),
    "shoes": _entry(RECYCLABLE, "Pair them up; donate if still wearable."),
    "biological": _entry(KITCHEN, "Drain liquids; remove packaging before disposal."),
    "battery": _entry(HAZARDOUS, "Never put in normal trash; use a battery collection point."),
    "trash": _entry(OTHER, "Put in the general waste bin."),
}

# Confidence below this value means "not sure". It was chosen on the validation split only
# (E3: 91.7% of images answered, 99.5% of them correct). See docs/experiment_notes.md.
DEFAULT_THRESHOLD = 0.90

_FALLBACK = _entry(OTHER, "Put in the general waste bin.")


def advice(label: str) -> dict:
    """Return the bin, Chinese bin name, colour and tip for a class name (a copy of the entry)."""
    entry = dict(BINS.get(label, _FALLBACK))
    entry["label"] = label
    entry["known"] = label in BINS
    return entry


def bin_of(label: str) -> str:
    """Just the English bin name for a class."""
    return advice(label)["bin"]


def decide(label: str, confidence: float, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Advice plus a flag that says whether the model is sure enough to give it."""
    result = advice(label)
    result["confidence"] = float(confidence)
    result["threshold"] = float(threshold)
    result["confident"] = float(confidence) >= float(threshold)
    return result
