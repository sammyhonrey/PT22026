"""The score-to-fraud-statement mapping (src/verdict.py).

These are the rules a reader sees on the testing page, so they are pinned here:
band edges, label thresholds and the wording of a rate.
"""
from src.verdict import BANDS, BandStat, band_for, rate_text


def _band(lo, hi, frauds, total):
    return BandStat(lo=lo, hi=hi, frauds=frauds, total=total)


def test_a_band_is_called_fraud_only_when_most_of_it_was_fraud():
    assert _band(99.5, 100.01, 95, 100).label == "Fraud"
    assert _band(99.0, 99.5, 50, 100).label == "Fraud"          # exactly half still counts
    assert _band(99.0, 99.5, 49, 100).label == "Suspicious"
    assert _band(95.0, 97.0, 10, 100).label == "Suspicious"     # exactly a tenth is suspicious
    assert _band(95.0, 97.0, 9, 100).label == "Not fraud"


def test_labels_map_to_chip_tones():
    assert _band(99.5, 100.01, 95, 100).tone == "fraud"
    assert _band(95.0, 97.0, 20, 100).tone == "suspicious"
    assert _band(0.0, 50.0, 1, 1000).tone == "clean"


def test_an_empty_band_is_not_called_fraud():
    empty = _band(50.0, 90.0, 0, 0)
    assert empty.rate == 0.0 and empty.label == "Not fraud"


def test_rates_are_written_as_counts_a_person_can_read():
    assert rate_text(0.95) == "95 in 100"
    assert rate_text(0.62) == "62 in 100"
    assert rate_text(0.004) == "4 in 1,000"
    assert rate_text(0.0) == "none"
    assert rate_text(1.0) == "almost all"


def test_every_score_finds_a_band_including_the_edges():
    stats = [_band(lo, hi, 0, 10) for lo, hi in BANDS]
    assert band_for(0.0, stats).span == "0-50"
    assert band_for(60.7, stats).span == "50-90"
    assert band_for(99.5, stats).span == "99.5-100"
    assert band_for(100.0, stats).span == "99.5-100"            # 100 belongs to the last band
    assert band_for(-3.0, stats) is None and band_for(101.0, stats) is None


def test_band_spans_round_the_open_edge_for_reading():
    assert _band(99.5, 100.01, 1, 2).span == "99.5-100"
    assert _band(0.0, 50.0, 1, 2).span == "0-50"
