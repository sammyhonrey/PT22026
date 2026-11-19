#!/usr/bin/env python3
"""Check declared opaque palette pairs; this is not rendered accessibility QA."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def luminance(colour: str) -> float:
    channels = [int(colour.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
    return sum(value * weight for value, weight in zip(linear, [.2126, .7152, .0722]))


def contrast(foreground: str, background: str) -> float:
    lighter, darker = sorted([luminance(foreground), luminance(background)], reverse=True)
    return (lighter + .05) / (darker + .05)


def check_palettes(path: Path) -> dict:
    content = path.read_bytes()
    rows = []
    for name, colours in json.loads(content).items():
        surfaces = ["bg", "panel", "sidebar", "control", "raised", "hover"]
        pairs = [(text, surface, 4.5) for text in ["text", "muted", "subtle"] for surface in surfaces]
        pairs += [("ink", "primary", 4.5)]
        pairs += [(accent, surface, 3) for accent in ["primary", "secondary"] for surface in surfaces]
        for foreground, background, minimum in pairs:
            ratio = contrast(colours[foreground], colours[background])
            rows.append({"palette": name, "foreground": foreground, "background": background,
                         "ratio": ratio, "minimum": minimum, "passed": ratio >= minimum})
    return {"scope": "Declared opaque palette pairs only; excludes compositing, rendered widgets, charts, keyboard, zoom and mobile layout. Not full WCAG conformance.",
            "reference": "https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html",
            "palette_sha256": hashlib.sha256(content).hexdigest(), "checks": rows,
            "passed": all(row["passed"] for row in rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/ui_contrast_verification.json")
    args = parser.parse_args()
    result = check_palettes(ROOT / "config/palettes.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{'PASS' if result['passed'] else 'FAIL'}: {len(result['checks'])} declared opaque colour pairs. Rendered visual/accessibility QA remains separate.")
    raise SystemExit(0 if result["passed"] else 1)
