#!/usr/bin/env python
"""Render the report HTML to an editable Word document.

Chromium can print to PDF but Word cannot read paged CSS, so the same semantic
HTML is rebuilt with real Word constructs: heading styles, shaded table headers,
captioned figures and callout boxes.

    python scripts/html_to_docx.py report/report.html out.docx --title "..."
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]

NAVY = "13234B"
CYAN = "0E7490"
VIOLET = "5B21B6"
GREY = "5B6B8C"
LIGHT = "F4F6FB"
CALLOUT_BG = "FFF8E7"
CODE_BG = "F1F3F7"

BODY_FONT = "Georgia"
HEAD_FONT = "Calibri"
MONO_FONT = "Consolas"


# Low-level Word helpers


def _rgb(hex_value: str) -> RGBColor:
    return RGBColor.from_string(hex_value.replace("#", "").upper())


def _shade(element, fill: str) -> None:
    """Apply a solid background to a table cell or paragraph."""
    target = element._tc.get_or_add_tcPr() if hasattr(element, "_tc") else element._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill.replace("#", "").upper())
    target.append(shd)


def _cell_margins(table, top: int = 60, start: int = 90) -> None:
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for edge, value in (("top", top), ("start", start), ("bottom", top), ("end", start)):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    tbl_pr.append(mar)


def _table_borders(table, colour: str = "D8DEEA") -> None:
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = OxmlElement(f"w:{edge}")
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), "4")
        node.set(qn("w:color"), colour)
        borders.append(node)
    tbl_pr.append(borders)


def _left_border(paragraph, colour: str, size: int = 18) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), str(size))
    left.set(qn("w:space"), "10")
    left.set(qn("w:color"), colour)
    borders.append(left)
    p_pr.append(borders)


def _page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(end)


# Inline formatting


def add_text(paragraph, node, *, size: float | None = None, bold: bool = False,
             italic: bool = False, mono: bool = False, colour: str | None = None) -> None:
    """Copy an HTML inline subtree into a Word paragraph, keeping emphasis."""
    if isinstance(node, NavigableString):
        text = str(node)
        if not text.strip() and text != " ":
            return
        run = paragraph.add_run(re.sub(r"\s+", " ", text))
        run.bold = bold or None
        run.italic = italic or None
        if mono:
            run.font.name = MONO_FONT
            run.font.size = Pt((size or 10.5) - 0.5)
        elif size:
            run.font.size = Pt(size)
        if colour:
            run.font.color.rgb = _rgb(colour)
        return

    if not isinstance(node, Tag):
        return
    name = node.name.lower()
    classes = node.get("class") or []
    if name == "br":
        paragraph.add_run().add_break()
        return
    if name in {"script", "style", "svg", "caption"}:
        return

    is_bold = bold or name in {"b", "strong"} or "card__title" in classes
    is_italic = italic or name in {"i", "em", "cite"}
    is_mono = mono or name in {"code", "kbd", "samp"} or "mono" in classes
    for child in node.children:
        add_text(paragraph, child, size=size, bold=is_bold, italic=is_italic,
                 mono=is_mono, colour=colour)


# SVG rasterisation


def rasterise_svgs(html_path: Path, out_dir: Path) -> list[Path]:
    """Screenshot every inline SVG in document order, at 2x for print clarity."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  note: playwright absent, inline diagrams will be skipped")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1000, "height": 900}, device_scale_factor=2)
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.emulate_media(media="screen")
        page.wait_for_timeout(400)
        svgs = page.locator("svg")
        for index in range(svgs.count()):
            target = out_dir / f"svg_{index:02d}.png"
            try:
                svgs.nth(index).screenshot(path=str(target), animations="disabled")
                saved.append(target)
            except Exception as exc:  # noqa: BLE001 - a broken glyph must not stop the build
                print(f"  warn: could not rasterise svg {index}: {type(exc).__name__}")
        browser.close()
    return saved


# Document building


class Builder:
    def __init__(self, doc: Document, base_dir: Path, svgs: list[Path]) -> None:
        self.doc = doc
        self.base = base_dir
        self.svgs = svgs
        self.svg_cursor = 0
        self.figure_no = 0
        self.first_page = True

    # -- helpers ---------------------------------------------------------- #

    def _next_svg(self) -> Path | None:
        if self.svg_cursor < len(self.svgs):
            path = self.svgs[self.svg_cursor]
            self.svg_cursor += 1
            return path
        return None

    def heading(self, tag: Tag, level: int) -> None:
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True))
        paragraph = self.doc.add_paragraph()
        paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 10)
        paragraph.paragraph_format.space_after = Pt(5)
        paragraph.paragraph_format.keep_with_next = True
        run = paragraph.add_run(text)
        run.font.name = HEAD_FONT
        run.bold = True
        run.font.size = Pt({1: 20, 2: 13.5, 3: 11.5}.get(level, 11))
        run.font.color.rgb = _rgb(NAVY if level == 1 else CYAN if level == 2 else VIOLET)
        if level == 1:
            _left_border(paragraph, CYAN, size=24)
            p_pr = paragraph._p.get_or_add_pPr()
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:fill"), "F7FAFD")
            p_pr.append(shd)

    def paragraph(self, tag: Tag) -> None:
        block = self.doc.add_paragraph()
        block.paragraph_format.space_after = Pt(6)
        block.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_text(block, tag)

    def eyebrow(self, text: str) -> None:
        block = self.doc.add_paragraph()
        block.paragraph_format.space_after = Pt(2)
        run = block.add_run(text.upper())
        run.font.name = HEAD_FONT
        run.font.size = Pt(7.5)
        run.bold = True
        run.font.color.rgb = _rgb(GREY)
        run.font.all_caps = True

    def list_block(self, tag: Tag, ordered: bool) -> None:
        style = "List Number" if ordered else "List Bullet"
        for item in tag.find_all("li", recursive=False):
            block = self.doc.add_paragraph(style=style)
            block.paragraph_format.space_after = Pt(3)
            add_text(block, item, size=10)

    def table(self, tag: Tag) -> None:
        caption = tag.find("caption")
        if caption:
            cap = self.doc.add_paragraph()
            cap.paragraph_format.space_before = Pt(6)
            cap.paragraph_format.space_after = Pt(3)
            cap.paragraph_format.keep_with_next = True
            add_text(cap, caption, size=8.5, bold=True, colour=GREY)

        rows = tag.find_all("tr")
        if not rows:
            return
        header_cells = rows[0].find_all(["th", "td"])
        columns = max(len(r.find_all(["th", "td"])) for r in rows)
        table = self.doc.add_table(rows=0, cols=columns)
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        _table_borders(table)
        _cell_margins(table)

        for row_index, row in enumerate(rows):
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            word_row = table.add_row()
            is_header = row_index == 0 and (cells[0].name == "th" or row.find("th") is not None)
            if is_header:
                # Repeat the header on every page a long table spans.
                tr_pr = word_row._tr.get_or_add_trPr()
                repeat = OxmlElement("w:tblHeader")
                repeat.set(qn("w:val"), "true")
                tr_pr.append(repeat)
            for col_index, node in enumerate(cells):
                if col_index >= columns:
                    break
                cell = word_row.cells[col_index]
                cell.text = ""
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(1)
                if "num" in (node.get("class") or []) and not is_header:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                add_text(paragraph, node, size=8.5, bold=is_header)
                if is_header:
                    _shade(cell, NAVY)
                    for run in paragraph.runs:
                        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                        run.bold = True
                elif row_index % 2 == 0:
                    _shade(cell, LIGHT)

    def card(self, tag: Tag) -> None:
        label = tag.find(class_="card__label")
        title = tag.find(class_="card__title")
        table = self.doc.add_table(rows=1, cols=1)
        table.autofit = True
        _table_borders(table, "E2E8F4")
        cell = table.rows[0].cells[0]
        cell.text = ""
        _shade(cell, LIGHT)
        first = cell.paragraphs[0]
        if label:
            run = first.add_run(re.sub(r"\s+", " ", label.get_text(" ", strip=True)))
            run.font.name = HEAD_FONT
            run.bold = True
            run.font.size = Pt(7.5)
            run.font.color.rgb = _rgb(CYAN)
        if title:
            block = cell.add_paragraph()
            run = block.add_run(title.get_text(" ", strip=True))
            run.font.name = HEAD_FONT
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = _rgb(NAVY)
        for child in tag.children:
            if isinstance(child, Tag) and child.name in {"p", "ul", "ol"}:
                if child.name == "p":
                    block = cell.add_paragraph()
                    block.paragraph_format.space_after = Pt(2)
                    add_text(block, child, size=9)
                else:
                    for item in child.find_all("li", recursive=False):
                        block = cell.add_paragraph(style="List Bullet")
                        block.paragraph_format.space_after = Pt(1)
                        add_text(block, item, size=8.5)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def callout(self, tag: Tag) -> None:
        table = self.doc.add_table(rows=1, cols=1)
        table.autofit = True
        _table_borders(table, "EBD9A8")
        cell = table.rows[0].cells[0]
        cell.text = ""
        _shade(cell, CALLOUT_BG)
        label = tag.find(class_="callout__label")
        first = cell.paragraphs[0]
        if label:
            run = first.add_run(re.sub(r"\s+", " ", label.get_text(" ", strip=True)).upper())
            run.font.name = HEAD_FONT
            run.bold = True
            run.font.size = Pt(7.5)
            run.font.color.rgb = _rgb("8A6A1F")
        for child in tag.children:
            if not isinstance(child, Tag):
                continue
            if child.name == "p" or child.name == "div":
                block = cell.add_paragraph() if cell.paragraphs[0].text else cell.paragraphs[0]
                block.paragraph_format.space_after = Pt(2)
                add_text(block, child, size=9)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def kpi_row(self, tag: Tag) -> None:
        items = tag.find_all(class_="kpi")
        if not items:
            return
        table = self.doc.add_table(rows=1, cols=len(items))
        table.autofit = True
        _table_borders(table, "DCE6F5")
        for index, item in enumerate(items):
            cell = table.rows[0].cells[index]
            cell.text = ""
            _shade(cell, "F5F9FE")
            value = item.find(class_="kpi__v")
            label = item.find(class_="kpi__l")
            if value:
                run = cell.paragraphs[0].add_run(value.get_text(strip=True))
                run.font.name = HEAD_FONT
                run.bold = True
                run.font.size = Pt(20)
                run.font.color.rgb = _rgb(CYAN)
            if label:
                block = cell.add_paragraph()
                run = block.add_run(label.get_text(" ", strip=True))
                run.font.size = Pt(8)
                run.font.color.rgb = _rgb(GREY)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(2)

    def figure(self, tag: Tag) -> None:
        image = None
        svg = tag.find("svg")
        img_tag = tag.find("img")
        if svg is not None:
            image = self._next_svg()
        elif img_tag is not None and img_tag.get("src"):
            candidate = (self.base / img_tag["src"]).resolve()
            if candidate.exists():
                image = candidate
        if image is not None:
            try:
                self.doc.add_picture(str(image), width=Inches(6.2))
                self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            except Exception as exc:  # noqa: BLE001
                print(f"  warn: figure skipped ({type(exc).__name__})")
        caption = tag.find(class_="diagram__cap") or tag.find(class_="figure__cap")
        if caption:
            block = self.doc.add_paragraph()
            block.alignment = WD_ALIGN_PARAGRAPH.CENTER
            block.paragraph_format.space_before = Pt(3)
            block.paragraph_format.space_after = Pt(10)
            add_text(block, caption, size=8.5, colour=GREY)

    def toc(self, tag: Tag) -> None:
        rows = tag.find_all(class_="toc__row")
        if not rows:
            return
        table = self.doc.add_table(rows=0, cols=2)
        table.autofit = True
        for row in rows:
            number = row.find(class_="toc__n")
            title = row.find(class_="toc__t")
            page = row.find(class_="toc__p")
            cells = table.add_row().cells
            left = cells[0].paragraphs[0]
            number_text = number.get_text(strip=True) if number else ""
            label = title.get_text(" ", strip=True) if title else ""
            run = left.add_run(f"{number_text}  " if number_text and number_text != "—" else "")
            run.font.name = HEAD_FONT
            run.bold = True
            run.font.color.rgb = _rgb(CYAN)
            left.add_run(label)
            if page is not None:
                right = cells[1].paragraphs[0]
                right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                right.add_run(page.get_text(strip=True))

    def formula(self, tag: Tag) -> None:
        block = self.doc.add_paragraph()
        block.alignment = WD_ALIGN_PARAGRAPH.CENTER
        block.paragraph_format.space_before = Pt(4)
        block.paragraph_format.space_after = Pt(4)
        text = tag.get_text(" ", strip=True)
        run = block.add_run(text)
        run.font.name = MONO_FONT
        run.font.size = Pt(9.5)
        _shade(block, CODE_BG)

    def divider(self, section: Tag) -> None:
        number = section.find(class_="divider__num")
        title = section.find(class_="divider__title")
        self.doc.add_paragraph()
        if number:
            run = self.doc.add_paragraph().add_run(number.get_text(strip=True))
            run.font.name = HEAD_FONT
            run.font.size = Pt(56)
            run.bold = True
            run.font.color.rgb = _rgb("C9D6EC")
        if title:
            block = self.doc.add_paragraph()
            run = block.add_run(title.get_text(" ", strip=True))
            run.font.name = HEAD_FONT
            run.font.size = Pt(24)
            run.bold = True
            run.font.color.rgb = _rgb(NAVY)
        for paragraph in section.find_all("p"):
            self.paragraph(paragraph)
        items = section.find_all(class_="divider__item")
        if items:
            table = self.doc.add_table(rows=0, cols=2)
            table.autofit = True
            for item in items:
                spans = item.find_all("span")
                cells = table.add_row().cells
                if spans:
                    cells[0].text = spans[0].get_text(strip=True)
                    cells[1].text = " ".join(s.get_text(" ", strip=True) for s in spans[1:])
                for cell in cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(9)

    # -- block dispatch ---------------------------------------------------- #

    def block(self, node: Tag) -> None:
        name = node.name.lower()
        classes = node.get("class") or []

        if name == "section":
            if "page" in classes:
                if not self.first_page:
                    self.doc.add_page_break()
                self.first_page = False
            if "divider" in classes:
                self.divider(node)
                return
            for child in node.children:
                if isinstance(child, Tag):
                    self.block(child)
            return

        if name in {"h1", "h2", "h3"}:
            self.heading(node, int(name[1]))
            return
        if name == "p":
            if "eyebrow" in classes:
                self.eyebrow(node.get_text(" ", strip=True))
                return
            if "ref" in classes:
                block = self.doc.add_paragraph()
                block.paragraph_format.space_after = Pt(3)
                block.paragraph_format.left_indent = Cm(0.7)
                block.paragraph_format.first_line_indent = Cm(-0.7)
                add_text(block, node, size=8.8)
                return
            self.paragraph(node)
            return
        if name in {"ul", "ol"}:
            self.list_block(node, ordered=name == "ol")
            return
        if name == "table":
            self.table(node)
            return
        if name == "figure":
            self.figure(node)
            return
        if name in {"div", "blockquote"}:
            if "card" in " ".join(classes):
                self.card(node)
                return
            if "callout" in " ".join(classes):
                self.callout(node)
                return
            if "kpi-row" in " ".join(classes):
                self.kpi_row(node)
                return
            if "toc" in classes:
                self.toc(node)
                return
            if "formula" in classes:
                self.formula(node)
                return
            if "pills" in classes:
                block = self.doc.add_paragraph()
                block.paragraph_format.space_after = Pt(8)
                for index, pill in enumerate(node.find_all(class_="pill")):
                    if index:
                        block.add_run("   ·   ").font.color.rgb = _rgb(GREY)
                    run = block.add_run(pill.get_text(" ", strip=True))
                    run.font.size = Pt(8.5)
                    run.font.color.rgb = _rgb(GREY)
                return
            if "cover__rule" in classes or "rule" in classes:
                block = self.doc.add_paragraph()
                _left_border(block, CYAN, size=6)
                return
            for child in node.children:
                if isinstance(child, Tag):
                    self.block(child)
                elif isinstance(child, NavigableString) and str(child).strip():
                    block = self.doc.add_paragraph()
                    add_text(block, child)
            return
        if name == "br":
            self.doc.add_paragraph()
            return
        # Anything else: recurse so content is never silently dropped.
        for child in node.children:
            if isinstance(child, Tag):
                self.block(child)


# Cover page


def build_cover(builder: Builder, section: Tag, title: str, subtitle: str) -> None:
    doc = builder.doc
    banner = doc.add_table(rows=1, cols=1)
    banner.autofit = True
    cell = banner.rows[0].cells[0]
    cell.text = ""
    _shade(cell, NAVY)
    _cell_margins(banner, top=260, start=180)
    first = cell.paragraphs[0]
    run = first.add_run(title)
    run.font.name = HEAD_FONT
    run.font.size = Pt(30)
    run.bold = True
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    block = cell.add_paragraph()
    run = block.add_run(subtitle)
    run.font.name = BODY_FONT
    run.font.size = Pt(13)
    run.font.color.rgb = _rgb("BBD3EE")

    for tag in section.find_all("p"):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        block = doc.add_paragraph()
        block.alignment = WD_ALIGN_PARAGRAPH.CENTER
        block.paragraph_format.space_before = Pt(14)
        add_text(block, tag, size=10.5, italic=True, colour=GREY)
        break

    badges = [b.get_text(" ", strip=True) for b in section.select(".badge")]
    if badges:
        block = doc.add_paragraph()
        block.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = block.add_run("   ·   ".join(badges))
        run.font.name = HEAD_FONT
        run.font.size = Pt(9.5)
        run.bold = True
        run.font.color.rgb = _rgb(CYAN)

    image = builder._next_svg()
    if image is not None:
        doc.add_picture(str(image), width=Inches(6.0))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    meta = section.select_one(".cover__meta")
    if meta:
        pairs = []
        for item in meta.find_all("div"):
            label = item.find("span")
            value = item.find("strong")
            if label and value:
                pairs.append((label.get_text(strip=True), value.get_text(" ", strip=True)))
        table = doc.add_table(rows=0, cols=2)
        table.autofit = True
        _table_borders(table, "DCE6F5")
        for label, value in pairs:
            cells = table.add_row().cells
            cells[0].text = ""
            run = cells[0].paragraphs[0].add_run(label)
            run.font.size = Pt(8.5)
            run.font.color.rgb = _rgb(GREY)
            cells[1].text = ""
            run = cells[1].paragraphs[0].add_run(value)
            run.font.size = Pt(9.5)
            run.bold = True
            run.font.color.rgb = _rgb(NAVY)
    # No page break here: the first body section opens a page itself, and adding
    # one too would leave a blank page between the cover and chapter one.


# Main


def configure(doc: Document, running_head: str) -> None:
    section = doc.sections[0]
    section.page_height = Cm(29.7)
    section.page_width = Cm(21.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.9)
    section.left_margin = Cm(2.1)
    section.right_margin = Cm(2.1)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12

    header = section.header.paragraphs[0]
    header.text = running_head
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for run in header.runs:
        run.font.name = HEAD_FONT
        run.font.size = Pt(7.5)
        run.font.color.rgb = _rgb(GREY)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _page_number(footer)
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = _rgb(GREY)


def stamp_document(doc: Document, title: str, subtitle: str, author: str) -> None:
    """Write the file properties a reader sees in Word's Info panel.

    python-docx starts from a blank template whose author is the library name and whose
    creation date is 2013, which looks like an artefact rather than a document. The
    properties below are the ones Word would otherwise fill in from the person editing.
    """
    now = datetime.now(timezone.utc)
    props = doc.core_properties
    props.author = author
    props.last_modified_by = author
    props.title = title
    props.subject = subtitle or title
    props.comments = ""
    props.keywords = "speech emotion recognition; multimodal fusion; stress; RAVDESS"
    props.category = "Professional Training YN3012170034 - Midterm assignment"
    props.created = now
    props.modified = now
    props.revision = 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html", type=Path)
    parser.add_argument("docx", type=Path)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--running-head", default="")
    parser.add_argument("--author", default="UDDIN MEJBAH")
    args = parser.parse_args(argv)

    if not args.html.exists():
        print(f"source not found: {args.html}")
        return 1

    soup = BeautifulSoup(args.html.read_text(encoding="utf-8"), "lxml")
    body = soup.body or soup

    with tempfile.TemporaryDirectory() as tmp:
        svgs = rasterise_svgs(args.html, Path(tmp))
        doc = Document()
        configure(doc, args.running_head or args.title)
        stamp_document(doc, args.title, args.subtitle, args.author)
        builder = Builder(doc, args.html.parent, svgs)

        cover = body.find("section", id="cover")
        if cover is not None:
            build_cover(builder, cover, args.title, args.subtitle or "")
            builder.first_page = False

        for node in body.children:
            if isinstance(node, Tag):
                if cover is not None and node is cover:
                    continue
                builder.block(node)

        args.docx.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(args.docx))

    size_mb = args.docx.stat().st_size / (1024 * 1024)
    paragraphs = sum(1 for _ in Document(str(args.docx)).paragraphs)
    print(f"docx : {args.docx}  ({size_mb:.2f} MB, {paragraphs} paragraphs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
