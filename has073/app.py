import json
import os

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import shap
import logging

from features import FEATURE_NAMES, extract_features, normalize_url
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LOCAL_MODEL_PATHS = ("models/best_model.joblib", "best_model.joblib")
HF_MODEL_REPO = "Ratul16961/phishing-url-random-forest"
HF_MODEL_FILE = "best_model.joblib"
CONFIG_PATH = "config.json"  
MAX_URL_LENGTH = 2000

READABLE = {
    "url_length": "URL length",
    "hostname_length": "hostname length",
    "path_length": "path length",
    "num_dots": "number of dots",
    "num_hyphens": "number of hyphens",
    "num_underscores": "number of underscores",
    "num_digits": "number of digits",
    "num_special_chars": "number of special characters",
    "num_subdomains": "number of subdomains",
    "has_at_symbol": "contains '@'",
    "has_ip_address": "IP address as host",
    "suspicious_word_count": "suspicious keywords",
    "has_double_slash_in_path": "'//' inside the path",
    "num_query_params": "query parameters",
}


def load_model():
    """Use a local model if present, otherwise download it from the Hugging Face Hub."""
    for path in LOCAL_MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path)
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILE)
        return joblib.load(downloaded)
    except Exception as exc:
        logger.warning("Could not load the model from the Hugging Face Hub: %s", exc)
        return None


model = load_model()

try:
    explainer = shap.TreeExplainer(model) if model is not None else None
except Exception as exc:
    logger.warning("SHAP explainer is not available: %s", exc)
    explainer = None


def load_threshold() -> float:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return float(json.load(f).get("threshold", 0.5))
    return 0.5


THRESHOLD = load_threshold()


def validate(url: str):
    """Return an error message if the input is not usable, otherwise None."""
    if not url or not url.strip():
        return "Please enter a URL."
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        return f"The URL is too long (maximum {MAX_URL_LENGTH} characters)."
    if any(c.isspace() for c in url):
        return "A URL cannot contain spaces."
    if "." not in normalize_url(url):
        return "This does not look like a URL (no domain found)."
    return None


def red_flags(values: dict) -> list:
    flags = []
    if values["has_at_symbol"]:
        flags.append("contains an '@' symbol")
    if values["has_ip_address"]:
        flags.append("uses an IP address instead of a domain name")
    if values["suspicious_word_count"] > 0:
        flags.append(f"contains {int(values['suspicious_word_count'])} suspicious keyword(s)")
    if values["num_hyphens"] >= 3:
        flags.append(f"has {int(values['num_hyphens'])} hyphens")
    if values["num_subdomains"] >= 3:
        flags.append(f"has {int(values['num_subdomains'])} subdomains")
    if values["url_length"] > 75:
        flags.append(f"is long ({int(values['url_length'])} characters)")
    if values["num_digits"] >= 8:
        flags.append(f"has many digits ({int(values['num_digits'])})")
    return flags


def top_contributions(feats: pd.DataFrame, k: int = 3) -> list:
    """Top-k features by SHAP value (positive = pushes toward phishing)."""
    if explainer is None:
        return []
    try:
        sv = explainer.shap_values(feats)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        row = sv[0]
    except Exception:
        return []
    order = np.argsort(-np.abs(row))[:k]
    return [(FEATURE_NAMES[i], float(row[i]), float(feats.iloc[0, i])) for i in order]


def predict(url: str):
    if model is None:
        return "Model not found. Run train.py first.", {}, ""
    error = validate(url)
    if error:
        return "Invalid input", {}, f"**{error}**"

    feats = pd.DataFrame([extract_features(url)], columns=FEATURE_NAMES)
    prob = float(model.predict_proba(feats)[0, 1])
    label = "Phishing" if prob >= THRESHOLD else "Safe"

    lines = ["### Why this result?"]
    contributions = top_contributions(feats)
    if contributions:
        lines.append("**Strongest signals used by the model:**")
        for name, shap_value, value in contributions:
            direction = "towards phishing" if shap_value > 0 else "towards safe"
            lines.append(
                f"- {READABLE[name]} = {value:g} pushes {direction} ({shap_value:+.2f})"
            )
    flags = red_flags(dict(zip(FEATURE_NAMES, feats.iloc[0])))
    if flags:
        lines.append("**Red flags:** this URL " + "; ".join(flags) + ".")
    elif label == "Safe":
        lines.append("No obvious red flags were found in the URL text.")
    lines.append(
        "\n*The scheme (http/https) is ignored. This is an educational demo "
        "based on URL text only, not a security tool.*"
    )

    return label, {"Safe": 1 - prob, "Phishing": prob}, "\n".join(lines)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter a URL", placeholder="http://example.com/login"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Confidence"),
        gr.Markdown(),
    ],
    title="Phishing URL Detector",
    description="Paste a URL to check whether it looks like phishing.",
    examples=[
        ["https://www.wikipedia.org"],
        ["http://paypa1-secure-login.xyz/verify@account"],
    ],
)

if __name__ == "__main__":
    demo.launch()