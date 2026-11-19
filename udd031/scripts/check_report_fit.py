#!/usr/bin/env python
"""Check that every section in report.html fits one A4 page.

Each page has a fixed height and hidden overflow, so over-long content is clipped
without an error. This measures the real content extent per section under print
emulation and flags anything that spills.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "report" / "report.html"
SHOT_DIR = ROOT / "report" / "_fit"

#: A4 at 96 CSS px per inch: 297mm = 1122.52px.
A4_PX = 1122.5
#: Sub-pixel rounding is not overflow.
TOLERANCE_PX = 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", action="store_true", help="save PNGs of overflowing sections")
    args = parser.parse_args(argv)

    if not HTML.exists():
        print(f"missing {HTML}")
        return 1

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 794, "height": int(A4_PX)})
        page.goto(HTML.resolve().as_uri(), wait_until="networkidle")
        page.emulate_media(media="print")
        page.wait_for_timeout(500)

        rows = page.evaluate(
            """() => Array.from(document.querySelectorAll('section.page')).map((s, i) => {
                 const box = s.getBoundingClientRect();
                 let bottom = box.top;
                 for (const el of s.querySelectorAll('*')) {
                   const r = el.getBoundingClientRect();
                   if (r.height > 0 && r.bottom > bottom) bottom = r.bottom;
                 }
                 return {
                   i: i + 1,
                   id: s.id || '(no id)',
                   box: Math.round(box.height),
                   content: Math.round(bottom - box.top),
                   own: Math.round(s.scrollHeight)
                 };
               })"""
        )

        overflow = []
        print(f"{'#':>3}  {'content':>7}  {'box':>7}  {'scroll':>7}  id")
        for row in rows:
            over = row["content"] > A4_PX + TOLERANCE_PX
            flag = "  <-- OVERFLOW" if over else ""
            print(f"{row['i']:>3}  {row['content']:>7}  {row['box']:>7}  {row['own']:>7}  {row['id']}{flag}")
            if over:
                overflow.append(row)

        if args.shots:
            SHOT_DIR.mkdir(parents=True, exist_ok=True)
            for row in overflow:
                el = page.query_selector_all("section.page")[row["i"] - 1]
                target = SHOT_DIR / f"page{row['i']:02d}_{row['id'].replace('/', '-')}.png"
                el.screenshot(path=str(target))
                print(f"  shot -> {target.relative_to(ROOT)}")

        browser.close()

    print(f"\n{len(overflow)} of {len(rows)} sections exceed one A4 page ({A4_PX:.0f}px)")
    for row in overflow:
        print(f"  {row['id']}: {row['content'] - A4_PX:.0f}px too tall")
    return 0 if not overflow else 2


if __name__ == "__main__":
    sys.exit(main())
