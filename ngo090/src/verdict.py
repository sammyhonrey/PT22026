"""Turn a risk percentile into the fraud statement a reader can act on.

The engine's score is a percentile against normal traffic, not a probability, so
on its own it never answers "is this fraud?". This module measures, on the labelled
replay window, how often transactions in each score band actually were fraud, and
maps a score to its band. Every number comes from labelled data, so the label can
be checked against the file instead of being a modelling assumption.

    stats = band_stats(engine, run.replay)
    band = band_for(99.6, stats)
    band.label   # "Fraud" -- 89 in 100 transactions at this score were fraud
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.processing import engine_frame

# Bands: wide where the score says little, narrow where it decides.
BANDS: tuple[tuple[float, float], ...] = (
    (0.0, 50.0), (50.0, 90.0), (90.0, 95.0), (95.0, 97.0), (97.0, 99.0), (99.0, 99.5), (99.5, 100.01),
)
FRAUD_AT = 0.50          # at or above this share of frauds, the band is called fraud
SUSPICIOUS_AT = 0.10     # above this share, the band is called suspicious


@dataclass(frozen=True)
class BandStat:
    """How often a score band was fraud on the labelled window."""
    lo: float
    hi: float
    frauds: int
    total: int

    @property
    def rate(self) -> float:
        return self.frauds / self.total if self.total else 0.0

    @property
    def label(self) -> str:
        if self.rate >= FRAUD_AT:
            return "Fraud"
        return "Suspicious" if self.rate >= SUSPICIOUS_AT else "Not fraud"

    @property
    def tone(self) -> str:
        return {"Fraud": "fraud", "Suspicious": "suspicious"}.get(self.label, "clean")

    @property
    def span(self) -> str:
        return f"{self.lo:g}-{min(self.hi, 100.0):g}"


def rate_text(rate: float) -> str:
    """'89 in 100' or '3 in 1,000' -- a share a person can read without percentages."""
    if rate <= 0:
        return "none"
    if rate < 0.01:
        return f"{max(1, round(rate * 1000)):,} in 1,000"
    if rate >= 0.995:
        return "almost all"
    return f"{round(rate * 100):.0f} in 100"


def band_stats(engine, labelled: pd.DataFrame) -> list[BandStat]:
    """Score the replay window once and count frauds per score band."""
    risk = np.asarray(engine.score(engine_frame(labelled)).risk)
    fraud = labelled["isFraud"].to_numpy().astype(bool)
    out = []
    for lo, hi in BANDS:
        inside = (risk >= lo) & (risk < hi)
        out.append(BandStat(lo=lo, hi=hi, frauds=int(fraud[inside].sum()), total=int(inside.sum())))
    return out


def band_for(risk: float, stats: list[BandStat]) -> BandStat | None:
    """The band that contains this score; 100 belongs to the last band, outside 0-100 is None."""
    for band in stats:
        if band.lo <= risk < band.hi:
            return band
    if stats and risk == 100.0 and stats[-1].hi >= 100:
        return stats[-1]
    return None
