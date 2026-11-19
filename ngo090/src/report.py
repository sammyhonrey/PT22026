"""The case report: one investigation, exported as PDF or CSV.

Both shapes say the same thing -- the case as it stands, the recommendation the
rules produced (when one is on screen), the evidence behind it, and the case's
own trail. Nothing here writes to a store: an export is a read, so pressing
"Download" can no more move a case than reading it can.

The content is assembled deterministically from the record, the same way
RiskPilot's answers are: the same case always produces the same report (the
"generated at" line aside, which is passed in by the caller).
"""
from __future__ import annotations

import csv
import io
from xml.sax.saxutils import escape as _xml_escape

from src.agent import Investigation
from src.cases import Case, CaseEvent, DECISION_LABELS

REPORT_TOOL = "reportlab"        # the only extra dependency, for the PDF shape


# --------------------------------------------------------------------------- #
# The shared skeleton: (heading, [(item, detail)]) pairs
# --------------------------------------------------------------------------- #
def _money(value: float) -> str:
    return f"{float(value):,.2f}"


def _would_set(recommendation) -> str:
    """The change the recommendation would make, as one line."""
    would = []
    if recommendation.proposed_status:
        would.append("resolution " + recommendation.proposed_status.replace("_", " "))
    if recommendation.proposed_state:
        would.append("state " + recommendation.proposed_state)
    if recommendation.proposed_priority:
        would.append("priority " + recommendation.proposed_priority)
    return ", ".join(would) if would else "none -- the case already matches"


def report_sections(case: Case, events: tuple[CaseEvent, ...] | list = (),
                    investigation: Investigation | None = None
                    ) -> list[tuple[str, list[tuple[str, str]]]]:
    """The report's content, shared by the PDF and the CSV.

    `events` is the case's history; it is rendered oldest first -- a report
    reads like a story, a history popover reads like a feed.
    """
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    sections.append(("Case", [
        ("Case", case.case_id),
        ("Transaction", f"#{int(case.txn_id)} · "
                        f"{(case.tx_type or 'unknown type').replace('_', ' ').lower()}"),
        ("Amount", _money(case.amount)),
        ("Risk at open", f"{case.risk:.1f} / 100"),
        ("Policy decision", DECISION_LABELS.get(case.decision, case.decision.title())),
        ("Resolution", case.status_label),
        ("State", case.state_label),
        ("Priority", case.priority_label),
        ("Source", case.source),
        ("Opened", case.created_at),
        ("Last update", case.updated_at),
    ]))
    if case.explanation:
        sections[-1][1].append(("Opening note", case.explanation))

    if investigation is not None:
        rec = investigation.recommendation
        sections.append(("Recommendation", [
            ("Verdict", rec.label),
            ("Rule", rec.rule),
            ("Risk now", f"{investigation.risk:.1f} / 100 · policy says "
                         + DECISION_LABELS.get(investigation.policy_decision,
                                               investigation.policy_decision.title())),
            ("Proposed change", _would_set(rec)),
            ("Needs an override", "yes" if rec.requires_override else "no"),
            ("Prepared by", investigation.agent),
            ("Summary", investigation.narrative),
        ]))
        if investigation.evidence:
            sections.append(("Evidence", [
                (evidence.source, evidence.text)
                for evidence in investigation.evidence
            ]))
        if investigation.gaps:
            sections.append(("What was not checked", [("--", gap) for gap in investigation.gaps]))

    sections.append(("Notes", [("--", case.notes.strip() or "Nothing recorded yet.")]))
    trail = [(event.created_at, event.detail) for event in reversed(list(events))] \
        or [("--", "No edit has touched this case since it was opened.")]
    sections.append(("History", trail))
    return sections


# --------------------------------------------------------------------------- #
# The CSV shape
# --------------------------------------------------------------------------- #
def case_report_csv(case: Case, events=(), investigation: Investigation | None = None) -> str:
    """The whole report as one table: Section, Item, Detail."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Section", "Item", "Detail"])
    for heading, rows in report_sections(case, events, investigation):
        for item, detail in rows:
            writer.writerow([heading, item, detail])
    return buf.getvalue()


def csv_file_name(case: Case) -> str:
    return f"FRAUDINTEL_case_{case.case_id}.csv"


# --------------------------------------------------------------------------- #
# The PDF shape (reportlab)
# --------------------------------------------------------------------------- #
def pdf_file_name(case: Case) -> str:
    return f"FRAUDINTEL_case_{case.case_id}.pdf"


def case_report_pdf(case: Case, events=(), investigation: Investigation | None = None,
                    generated_at: str = "") -> bytes:
    """The same content, laid out for print. Deterministic for equal inputs:
    the caller passes `generated_at` (and reportlab's invariant mode pins the
    document dates), so two exports of an unchanged case are byte-identical."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    heading = colors.HexColor("#1d3a6b")
    soft = colors.HexColor("#eef2f9")
    line = colors.HexColor("#c9d4e8")

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=17,
                                 textColor=heading, spaceAfter=1)
    sub_style = ParagraphStyle("ReportSub", parent=styles["Normal"], fontSize=9.5,
                               textColor=colors.HexColor("#5a6b85"), spaceAfter=10)
    section_style = ParagraphStyle("ReportSection", parent=styles["Heading2"], fontSize=12,
                                   textColor=heading, spaceBefore=12, spaceAfter=4)
    cell_style = ParagraphStyle("ReportCell", parent=styles["Normal"], fontSize=9,
                                leading=12.5)
    item_style = ParagraphStyle("ReportItem", parent=cell_style, textColor=colors.HexColor("#41506b"))

    def _para(text: str, style) -> Paragraph:
        return Paragraph(_xml_escape(str(text)).replace("\n", "<br/>"), style)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm, title=f"Case report {case.case_id}",
        author="FRAUDINTEL", invariant=1)
    story = [Paragraph("FRAUDINTEL", title_style),
             Paragraph(f"Case report · {case.case_id}"
                       + (f" · generated {generated_at}" if generated_at else ""), sub_style)]

    for heading_text, rows in report_sections(case, events, investigation):
        story.append(Paragraph(heading_text, section_style))
        data = [[_para(item, item_style), _para(detail, cell_style)] for item, detail in rows]
        table = Table(data, colWidths=[42 * mm, 112 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), soft),
            ("GRID", (0, 0), (-1, -1), 0.4, line),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Exported from FRAUDINTEL. This report is a read: the record it describes is "
        "append-only in the case store, and only an approved proposal or an analyst's "
        "save can change it.", sub_style))
    doc.build(story)
    return buf.getvalue()
