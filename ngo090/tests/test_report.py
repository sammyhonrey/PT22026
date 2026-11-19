"""The case report export (src/report.py): one case, PDF and CSV.

An export is a read: the tests drive the same sections the download buttons
serve, and check the two shapes agree on content and that the PDF bytes are
deterministic for an unchanged case.
"""
import csv
from io import StringIO
from types import SimpleNamespace

import pytest

from src.cases import CaseStore
from src.report import (case_report_csv, case_report_pdf, csv_file_name,
                        pdf_file_name, report_sections)


@pytest.fixture
def case(tmp_path):
    store = CaseStore(tmp_path / "cases.db")
    return store.create(txn_id=986, risk=99.0, decision="review", amount=477_066.97,
                        tx_type="CASH_OUT", source="live",
                        explanation="Receiver balance 0 (typical 558,465)")


def _investigation():
    """A hand-built reading, the shape InvestigationAgent.investigate returns."""
    return SimpleNamespace(
        recommendation=SimpleNamespace(label="Escalate", rule="R3", proposed_status=None,
                                       proposed_state=None, proposed_priority="urgent",
                                       requires_override=False),
        risk=97.4, policy_decision="review", agent="rules:R3",
        narrative="The alert is part of a cluster of three look-alikes.",
        evidence=(SimpleNamespace(source="get_case", text="Case opened at risk 99.0."),
                  SimpleNamespace(source="find_related_transactions",
                                  text="2 related transactions scored at or above 90.")),
        gaps=("Related activity was not searched: no corpus available.",))


def _sections_map(case, events=(), investigation=None):
    return {heading: rows for heading, rows in report_sections(case, events, investigation)}


def test_the_case_facts_are_there(case):
    sections = _sections_map(case)
    rows = dict(sections["Case"])
    assert rows["Case"] == case.case_id
    assert rows["Amount"] == "477,066.97"
    assert rows["Resolution"] == "Investigating"
    assert rows["State"] == "Open"
    assert rows["Priority"] == "Urgent"         # risk 99 + review -> urgent by default
    assert rows["Policy decision"] == "Sent to review"
    assert rows["Opening note"] == case.explanation


def test_the_recommendation_is_included_only_when_there_is_a_reading(case):
    assert "Recommendation" not in _sections_map(case)

    sections = _sections_map(case, investigation=_investigation())
    rows = dict(sections["Recommendation"])
    assert rows["Verdict"] == "Escalate"
    assert rows["Rule"] == "R3"
    assert rows["Proposed change"] == "priority urgent"
    assert rows["Needs an override"] == "no"
    assert rows["Prepared by"] == "rules:R3"


def test_evidence_and_gaps_ride_along_with_a_reading(case):
    sections = _sections_map(case, investigation=_investigation())
    assert [detail for _, detail in sections["Evidence"]] == \
        ["Case opened at risk 99.0.",
         "2 related transactions scored at or above 90."]
    assert sections["What was not checked"] == [
        ("--", "Related activity was not searched: no corpus available.")]


def test_notes_say_nothing_recorded_when_empty(case):
    assert dict(_sections_map(case)["Notes"])["--"] == "Nothing recorded yet."


def test_the_history_reads_oldest_first(tmp_path):
    from src.tools import Toolkit
    store = CaseStore(tmp_path / "cases.db")
    made = store.create(txn_id=1, risk=95.0, decision="review", amount=10.0)
    Toolkit(cases=store).update_case(made.case_id, priority="urgent")
    events = store.events(made.number)              # newest first, as the popover shows
    details = [detail for _, detail in _sections_map(made, events=events)["History"]]
    assert details[0].startswith("Opened")          # oldest first, as a report reads
    assert "urgent" in details[-1].lower()


def test_the_csv_parses_and_carries_the_same_content(case):
    rows = list(csv.reader(StringIO(case_report_csv(case, investigation=_investigation()))))
    assert rows[0] == ["Section", "Item", "Detail"]
    assert ["Case", "Resolution", "Investigating"] in rows
    assert ["Recommendation", "Verdict", "Escalate"] in rows
    assert any(row[0] == "Evidence" for row in rows)


def test_the_file_names_carry_the_case_id(case):
    assert pdf_file_name(case) == f"FRAUDINTEL_case_{case.case_id}.pdf"
    assert csv_file_name(case) == f"FRAUDINTEL_case_{case.case_id}.csv"


def test_the_pdf_is_real_and_deterministic(case):
    pytest.importorskip("reportlab")
    first = case_report_pdf(case, investigation=_investigation(),
                            generated_at="15 Nov 2026 13:40")
    again = case_report_pdf(case, investigation=_investigation(),
                            generated_at="15 Nov 2026 13:40")
    assert first.startswith(b"%PDF") and len(first) > 2_000
    assert first == again          # an unchanged case exports byte-identical reports
