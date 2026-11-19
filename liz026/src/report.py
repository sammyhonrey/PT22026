"""Generates a downloadable HTML report for each analyzed X-ray."""
import base64
import datetime as dt
import html
import io
import re

from PIL import Image

from src import config


def _img_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "Anonymous"


def build_report(patient_name: str, label: str, prob_pneumonia: float,
                 threshold: float, xray: Image.Image, heatmap: Image.Image,
                 warnings: list[str]) -> str:
    """Write a self-contained HTML report (images embedded) and return its path."""
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now()
    name = patient_name.strip() or "Anonymous"
    confidence = prob_pneumonia if label == "Pneumonia" else 1 - prob_pneumonia
    color = "#c0392b" if label == "Pneumonia" else "#1e8449"
    warn_html = "".join(f"<li>{html.escape(w)}</li>" for w in warnings)

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>X-Ray Report - {html.escape(name)}</title>
<style>
 body{{font-family:Arial,sans-serif;max-width:820px;margin:30px auto;color:#222}}
 h1{{border-bottom:2px solid #333;padding-bottom:6px}}
 table{{border-collapse:collapse;margin:12px 0}} td{{padding:6px 14px;border:1px solid #ccc}}
 .result{{font-size:22px;font-weight:bold;color:{color}}}
 .imgs{{display:flex;gap:16px}} .imgs figure{{margin:0;text-align:center}}
 .imgs img{{width:360px;border:1px solid #999}}
 .note{{font-size:12px;color:#666;margin-top:24px}}
</style></head><body>
<h1>Chest X-Ray Analysis Report</h1>
<table>
 <tr><td>Patient</td><td>{html.escape(name)}</td></tr>
 <tr><td>Date / time</td><td>{now:%Y-%m-%d %H:%M:%S}</td></tr>
 <tr><td>Result</td><td class="result">{label}</td></tr>
 <tr><td>Confidence</td><td>{confidence*100:.2f}%</td></tr>
 <tr><td>P(Pneumonia)</td><td>{prob_pneumonia:.4f} (decision threshold {threshold:.2f})</td></tr>
 <tr><td>Model</td><td>MobileNetV2 transfer-learning CNN, 224x224 input</td></tr>
</table>
{f'<h3>Warnings</h3><ul>{warn_html}</ul>' if warnings else ''}
<div class="imgs">
 <figure><img src="{_img_to_data_uri(xray)}"><figcaption>Input X-ray (224x224)</figcaption></figure>
 <figure><img src="{_img_to_data_uri(heatmap)}"><figcaption>Grad-CAM: regions that influenced the decision</figcaption></figure>
</div>
<p class="note">This result was produced automatically by a deep-learning model for
educational purposes only. It is not a medical diagnosis. Consult a qualified
radiologist or physician.</p>
</body></html>"""

    path = config.REPORT_DIR / f"Report_{_safe_filename(name)}_{now:%Y%m%d_%H%M%S}.html"
    path.write_text(doc, encoding="utf-8")
    return str(path)
