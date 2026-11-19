#!/usr/bin/env python
"""Rewrite the contents page numbers to match the rendered report.

Measures where each section lands under print emulation and writes the number back
into ``report/_parts/00_open.html``. ``--check`` exits 1 when the table is stale.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "report" / "report.html"
TOC_PATH = ROOT / "report" / "_parts" / "00_open.html"

#: A4 at 96 CSS px per inch. Each ``section.page`` is 297mm tall.
A4_PX = 1122.5
#: Chromium rounds 297mm up by a fraction of a pixel; treat that as one page.
SLACK_PX = 4.0

ROW_RE = re.compile(
    r'<span class="toc__n">(?P<n>[^<]*)</span>'
    r'<span class="toc__t">(?P<t>[^<]*)</span>'
    r'<span class="toc__p">(?P<p>\d+)</span>'
)

#: Entries whose heading text is not the contents label, mapped to a section id.
ID_ENTRIES = {
    "Abstract": "abstract",
    "Introduction": "s1",
    "Related work": "s2",
    "System design and methodology": "s3-divider",
    "Implementation": "s4",
    "Results and discussion": "s5",
    "Conclusion and future work": "s6",
    "References": "references",
    "Rubric mapping": "appendix-a",
    "Declaration of AI assistance": "declaration",
}

_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*|[AB])(?:\.\s*|\s+)")

#: Ignored when matching a short contents label against a longer printed heading.
_STOPWORDS = frozenset(
    "a an and the of to in with for this that by on at as is are be from".split()
)


def _strip_numbering(text: str) -> str:
    return _NUMBERING.sub("", text.strip()).strip()


def _pages_spanned(height: float) -> int:
    if height <= A4_PX + SLACK_PX:
        return 1
    return math.ceil(height / A4_PX)


def measure() -> dict:
    """Return section heights, section offsets and heading offsets from Chromium."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 794, "height": int(A4_PX)})
        page.goto(HTML_PATH.resolve().as_uri(), wait_until="networkidle")
        page.emulate_media(media="print")
        page.wait_for_timeout(500)
        data = page.evaluate(
            """() => {
                 const secs = Array.from(document.querySelectorAll('section.page'));
                 const sections = secs.map(s => ({
                   id: s.id || '',
                   h: s.scrollHeight,
                   top: s.getBoundingClientRect().top + window.scrollY
                 }));
                 const heads = Array.from(document.querySelectorAll('h1.sec, h2.sub, h3.minor'))
                   .map(h => ({
                     text: h.textContent.replace(/\\s+/g, ' ').trim(),
                     top: h.getBoundingClientRect().top + window.scrollY
                   }));
                 return { sections, heads };
               }"""
        )
        browser.close()
    return data


def build_index(data: dict) -> tuple[list[dict], list[dict]]:
    """Attach a first-page number to every section and heading."""
    starts: list[int] = []
    page_no = 1
    for section in data["sections"]:
        section["first_page"] = page_no
        section["pages"] = _pages_spanned(section["h"])
        starts.append(section["top"])
        page_no += section["pages"]

    for head in data["heads"]:
        index = 0
        for i, top in enumerate(starts):
            if head["top"] >= top:
                index = i
        offset = head["top"] - starts[index]
        head["page"] = data["sections"][index]["first_page"] + int(offset // A4_PX)
    return data["sections"], data["heads"]


def resolve(label: str, sections: list[dict], heads: list[dict]) -> int | None:
    """Find the printed page for one contents label."""
    if label in ID_ENTRIES:
        wanted = ID_ENTRIES[label]
        for section in sections:
            if section["id"] == wanted:
                return section["first_page"]
        return None

    target = _strip_numbering(label).lower()
    for head in heads:
        text = _strip_numbering(head["text"]).lower()
        if text == target or text.startswith(target):
            return head["page"]

    # Contents labels are deliberately shorter than the printed headings
    # ("Scope, ethics and limits of claim" vs "... and the limits of the claim"),
    # so fall back to a unique prefix match on the first few significant words.
    tokens = [token for token in re.split(r"[^a-z0-9]+", target) if token]
    significant = [token for token in tokens if token not in _STOPWORDS]
    if len(significant) < 2:
        return None

    hits = []
    for head in heads:
        words = [
            token for token in re.split(r"[^a-z0-9]+", _strip_numbering(head["text"]).lower()) if token
        ]
        content = [word for word in words if word not in _STOPWORDS]
        shared = sum(1 for token in significant if token in content)
        if content[:2] == significant[:2] and shared / len(significant) >= 0.6:
            hits.append(head["page"])
    return hits[0] if len(set(hits)) == 1 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="report drift without rewriting")
    parser.add_argument("--dump-heads", action="store_true", help="print every measured heading and exit")
    args = parser.parse_args(argv)

    for path in (HTML_PATH, TOC_PATH):
        if not path.exists():
            print(f"missing {path}")
            return 1

    sections, heads = build_index(measure())
    if args.dump_heads:
        for head in heads:
            print(f"p{head['page']:>3}  {head['text']}")
        return 0

    toc_source = TOC_PATH.read_text(encoding="utf-8")
    rows = list(ROW_RE.finditer(toc_source))
    if not rows:
        print("no contents rows found; nothing to do")
        return 1

    total = sum(section["pages"] for section in sections)
    print(f"{len(sections)} sections -> {total} printed pages")
    for section in sections:
        if section["pages"] > 1:
            print(f"  WARN section {section['id'] or '(no id)'} spans {section['pages']} pages")

    updates: list[tuple[str, str, int, int]] = []
    missing: list[str] = []
    for row in rows:
        label = row.group("t")
        old = int(row.group("p"))
        new = resolve(label, sections, heads)
        if new is None:
            missing.append(label)
            continue
        updates.append((row.group(0), row.group("n"), old, new))

    stale = [(n, old, new) for _span, n, old, new in updates if old != new]
    for number, old, new in stale:
        print(f"  {number:>6}  {old:>3} -> {new}")

    if missing:
        print(f"\n  WARN no heading matched: {', '.join(missing)}")

    if args.check:
        if stale or missing:
            print(f"\ncontents table is stale ({len(stale)} changed of {len(updates)} resolved)")
            return 1
        print("\ncontents table is in sync")
        return 0

    rewritten = toc_source
    for span, _number, old, new in updates:
        if old == new:
            continue
        replacement = span.replace(f'<span class="toc__p">{old}</span>', f'<span class="toc__p">{new}</span>')
        rewritten = rewritten.replace(span, replacement, 1)
    TOC_PATH.write_text(rewritten, encoding="utf-8")
    print(f"\nrewrote {len(stale)} page numbers in {TOC_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
