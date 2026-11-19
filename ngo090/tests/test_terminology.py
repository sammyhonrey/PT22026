"""One name per concept, everywhere.

The app carries three separate things and used to label them inconsistently
("outcome" vs "true label", "verdict" vs "predicted", "review" vs "send to
review"). These tests pin the vocabulary so a new page cannot reintroduce a
synonym:

    Verdict   what the model says          Fraud / Suspicious / Not fraud / Not covered
    Decision  what the policy does          Approve / Review / Block
    Outcome   what actually happened        Fraud / Legitimate / unknown (-)

They are fast: no engine, no app boot, only the render helpers.
"""
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
sys.path.insert(0, str(APP))
import ui  # noqa: E402

# Words that mean one of the three nouns but are not the chosen name for it.
BANNED = ("true label", "true-label", "actual label", "ground truth",
          "predicted:", "prediction:", "send to review", "false alarm")
DECISIONS = {"Approve", "Review", "Block"}
VERDICTS = {"Fraud", "Suspicious", "Not fraud", "Not covered"}
OUTCOMES = {"Fraud", "Legitimate", "-"}


def body(html) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(html)))


def headers(html) -> list[str]:
    return re.findall(r"<th>([^<]+)</th>", str(html))


def test_each_decision_has_exactly_one_word():
    assert {text for text, _ in ui.DECISION_TEXT.values()} == DECISIONS


def test_the_alerts_table_names_and_orders_the_three_nouns():
    rows = [{"hour": "16:00", "type": "TRANSFER", "amount": 800_000.0, "risk": 99.7,
             "reason": "unusually large", "action": "block", "fraud": True},
            {"hour": "16:01", "type": "CASH_OUT", "amount": 900_000.0, "risk": 96.2,
             "reason": "empty receiver", "action": "review", "fraud": False}]
    html = ui.alerts_table(rows)
    assert headers(html) == ["Time", "Type", "Amount", "Risk", "Decision", "Outcome", "Main reason"]
    text = body(html)
    assert "Block" in text and "Review" in text       # same words as the policy pill
    assert "Legitimate" in text and "Fraud" in text   # an alert that was not fraud
    assert "False alarm" not in text


def test_the_live_feed_calls_the_truth_outcome():
    rows = pd.DataFrame([{"id": 1, "minute": 3, "type": "TRANSFER", "amount": 500_000.0,
                          "risk": 99.9, "decision": "block", "applicable": True, "isFraud": 1}])
    html = ui.feed_html(rows, show_truth=True)
    assert headers(html) == ["Time", "Type", "Amount", "Risk", "Decision", "Outcome"]


def test_the_scored_file_puts_verdict_before_decision_before_outcome():
    from src.verdict import BandStat

    bands = [BandStat(lo=0.0, hi=99.5, frauds=0, total=1_000),
             BandStat(lo=99.5, hi=100.01, frauds=950, total=1_000)]
    rows = pd.DataFrame([
        {"row": 1, "type": "TRANSFER", "amount": 250_000.0, "risk": 99.9, "reason": "amount",
         "decision": "block", "applicable": True, "isFraud": 1},
        {"row": 2, "type": "PAYMENT", "amount": 120.0, "risk": 0.0, "reason": "",
         "decision": "approve", "applicable": False, "isFraud": 0},
    ])
    html = ui.scored_table_html(rows, bands=bands, n_total=39_102)
    assert headers(html) == ["Row", "Type", "Amount", "Risk", "Verdict", "Decision", "Outcome", "Main reason"]
    text = body(html)
    assert "Not covered" in text                       # a type the engine has no examples for
    assert "Fraud" in text and "Legitimate" in text    # verdict and outcome, side by side
    assert "what actually happened" in text            # the footnote defines the three nouns
    for word in BANNED:
        assert word not in text.lower()


def test_the_single_card_speaks_the_same_language():
    from src.verdict import BandStat

    band = BandStat(lo=99.5, hi=100.01, frauds=1_131, total=1_191)
    html = ui.verdict_html(band, n_total=39_102)
    text = body(html)
    assert "Verdict: Fraud" in text
    assert band.label in VERDICTS


@pytest.mark.parametrize("path", sorted(APP.glob("*.py")), ids=lambda p: p.name)
def test_no_page_reintroduces_a_synonym(path):
    """A static guard: the source of a page must not contain a banned label."""
    source = path.read_text(encoding="utf-8").lower()
    found = [word for word in BANNED if word in source]
    assert not found, f"{path.name} uses {found}; the names are Verdict / Decision / Outcome"
