"""The comparison layer: scenarios, the diff, the noise floor (src/scenario.py).

No Streamlit and no trained models: a stand-in engine scores the way the real
one behaves -- an emptied sender into a new account is risky, a small transfer
between funded accounts is not -- so the arithmetic of a comparison can be
tested on its own.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.decision import CostConfig, Policy
from src.scenario import (
    MEASURES, REPLAY, SYNTHETIC, Corpus, Scenario, axes, compare, corpus_frame, run_scenario,
    shift_step,
)

BALANCED = Policy(block_threshold=99.5, review_min_score=90.0, daily_capacity=150)
STRICT = Policy(block_threshold=99.0, review_min_score=85.0, daily_capacity=300)
# The same rule with a review cut-off above what the stand-in engine gives a
# drain: it flags nothing, which is how a policy difference shows up here.
BLIND = Policy(block_threshold=99.5, review_min_score=95.0, daily_capacity=150)
# Capacity large enough that it never decides the outcome.
SIEVE = Policy(block_threshold=99.5, review_min_score=90.0, daily_capacity=10_000)


class FakeEngine:
    """The rule the models learned: the sender's balance is emptied."""

    def score(self, df: pd.DataFrame) -> SimpleNamespace:
        amount = df["amount"].to_numpy(dtype=float)
        sender = df["oldbalanceOrg"].to_numpy(dtype=float)
        risk = np.where(sender <= amount * 1.05, 92.0, 20.0)
        return SimpleNamespace(risk=risk, features=None)


def pool_rows(steps: int = 72, per_step: int = 12, fraud_share: float = 0.1) -> pd.DataFrame:
    """A tiny replay pool: some drains (fraud) and some ordinary transfers."""
    rng = np.random.default_rng(3)
    rows = []
    for step in range(1, steps + 1):
        for _ in range(per_step):
            fraud = rng.random() < fraud_share
            amount = float(rng.uniform(100_000, 900_000)) if fraud else float(rng.uniform(1_000, 40_000))
            sender = amount * (1.01 if fraud else float(rng.uniform(2.0, 9.0)))
            rows.append({"step": step, "type": "TRANSFER" if fraud else "PAYMENT",
                         "amount": amount, "oldbalanceOrg": sender,
                         "oldbalanceDest": 0.0 if fraud else 500_000.0,
                         "isFraud": int(fraud)})
    return pd.DataFrame(rows)


@pytest.fixture
def pool() -> pd.DataFrame:
    return pool_rows()


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


def run(name: str, policy: Policy, corpus: Corpus, engine, pool, costs=CostConfig()) -> object:
    return run_scenario(Scenario(name, policy, corpus, costs), engine, pool)


# --------------------------------------------------------------------------- #
# One scenario
# --------------------------------------------------------------------------- #
def test_a_scenario_reports_the_six_measures(engine, pool):
    result = run("Balanced", BALANCED, Corpus(first_day=1, last_day=3), engine, pool)
    assert set(result.metrics) >= {key for key, *_ in MEASURES}
    assert result.metrics["transactions"] == len(result.df)
    assert result.metrics["fraud_caught"] + result.metrics["fraud_missed"] == result.metrics["fraud_total"]


def test_the_policy_decides_which_fraud_is_stopped(engine, pool):
    corpus = Corpus(first_day=1, last_day=3)
    sieving = run("Balanced", SIEVE, corpus, engine, pool)
    blind = run("Blind", BLIND, corpus, engine, pool)
    assert sieving.metrics["fraud_caught"] > blind.metrics["fraud_caught"]
    assert blind.metrics["fraud_caught"] == 0
    assert sieving.metrics["reviewed"] >= blind.metrics["reviewed"]


def test_a_sample_is_drawn_once_and_repeats(engine, pool):
    one = corpus_frame(Corpus(first_day=1, last_day=3, sample=4242), pool)
    again = corpus_frame(Corpus(first_day=1, last_day=3, sample=4242), pool)
    other = corpus_frame(Corpus(first_day=1, last_day=3, sample=777), pool)
    assert one.equals(again)
    assert len(one) < len(pool)
    assert not one.equals(other)


def test_the_replay_window_is_cut_to_its_days(engine, pool):
    frame = corpus_frame(Corpus(first_day=2, last_day=2), pool)
    assert int(frame["step"].min()) == 25 and int(frame["step"].max()) == 48


def test_the_shift_moves_the_scores_not_the_money(engine, pool):
    corpus = Corpus(first_day=1, last_day=3)
    plain = run("Normal", BALANCED, corpus, engine, pool)
    shifted = run("Shift", BALANCED, Corpus(first_day=1, last_day=3, shift=True), engine, pool)
    assert not np.array_equal(plain.risk, shifted.risk)                 # the model saw another world
    assert shifted.metrics["fraud_amount_total"] == plain.metrics["fraud_amount_total"]
    assert shift_step(plain.df) == 37                                   # from the middle of 1..72


def test_the_shift_makes_the_same_policy_fire_more(engine, pool):
    """Amounts ten times larger stop looking like an emptied balance: the rule goes quiet."""
    plain = run("Normal", BALANCED, Corpus(first_day=1, last_day=3), engine, pool)
    shifted = run("Shift", BALANCED, Corpus(first_day=1, last_day=3, shift=True), engine, pool)
    assert shifted.metrics["fraud_caught"] != plain.metrics["fraud_caught"]


def test_a_corpus_refuses_impossible_settings():
    with pytest.raises(ValueError):
        Corpus(first_day=5, last_day=3)
    with pytest.raises(ValueError):
        Corpus(kind="psychic")
    with pytest.raises(ValueError):
        Corpus(kind=SYNTHETIC, hours=0)


def test_policy_line_says_it_in_words():
    line = Scenario("A", BALANCED).policy_line()
    assert "blocks at risk 99.5" in line and "150 reviews a day" in line
    never = Scenario("B", Policy(block_threshold=float("inf"))).policy_line()
    assert never.startswith("never blocks")


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #
def test_the_decomposition_adds_up_to_the_net_difference(engine, pool):
    corpus = Corpus(first_day=1, last_day=3)
    cmp = compare(run("Balanced", BALANCED, corpus, engine, pool),
                  run("Strict", STRICT, corpus, engine, pool))
    parts = cmp.decomposition
    assert parts["more caught"] + parts["more reviews"] + parts["more blocked customers"] \
        == pytest.approx(parts["net"], abs=1e-6)
    assert parts["net"] == pytest.approx(cmp.deltas["net_benefit"], abs=1e-6)


def test_two_policies_on_the_same_traffic_are_a_paired_comparison(engine, pool):
    corpus = Corpus(first_day=1, last_day=3)
    cmp = compare(run("Balanced", BALANCED, corpus, engine, pool),
                  run("Strict", STRICT, corpus, engine, pool))
    assert cmp.paired
    assert axes(cmp.left.scenario, cmp.right.scenario) == {"policy"}
    assert cmp.warnings == []
    assert cmp.interval is not None and cmp.interval[0] <= cmp.interval[1]


def test_the_same_scenario_twice_is_a_tie_inside_the_noise(engine, pool):
    corpus = Corpus(first_day=1, last_day=3)
    one = run("Balanced", BALANCED, corpus, engine, pool)
    cmp = compare(one, one)
    assert all(abs(value) < 1e-9 for value in cmp.deltas.values())
    assert cmp.interval == (0.0, 0.0)
    assert "inside the noise" in cmp.verdict


def test_another_sample_is_not_a_like_for_like_comparison(engine, pool):
    cmp = compare(run("A", BALANCED, Corpus(first_day=1, last_day=3, sample=11), engine, pool),
                  run("B", STRICT, Corpus(first_day=1, last_day=3, sample=22), engine, pool))
    assert not cmp.paired
    assert cmp.interval is None
    assert any("same transactions" in warning for warning in cmp.warnings)
    assert "not a like-for-like" in cmp.verdict


def test_two_axes_moving_is_called_out(engine, pool):
    cmp = compare(
        run("A", BALANCED, Corpus(first_day=1, last_day=3), engine, pool),
        run("B", STRICT, Corpus(first_day=1, last_day=3, shift=True), engine, pool, CostConfig(review_cost=2_000)))
    assert axes(cmp.left.scenario, cmp.right.scenario) == {"traffic", "policy", "costs"}
    assert any("cannot be pinned" in warning for warning in cmp.warnings)


def test_the_verdict_names_the_winner_and_the_price(engine, pool):
    corpus = Corpus(first_day=1, last_day=3, sample=1)
    cmp = compare(run("Balanced", SIEVE, corpus, engine, pool),
                  run("Blind", BLIND, corpus, engine, pool))
    assert cmp.verdict.startswith("Balanced is ahead")
    assert "fraud cases stopped" in cmp.verdict


def test_a_small_window_gets_a_noise_warning(engine, pool):
    tiny = Corpus(first_day=1, last_day=1, sample=0)
    cmp = compare(run("A", BALANCED, tiny, engine, pool), run("B", STRICT, tiny, engine, pool))
    small = min(cmp.left.metrics["fraud_total"], cmp.right.metrics["fraud_total"])
    assert (small < 50) == any("fewer than 50" in warning for warning in cmp.warnings)


# --------------------------------------------------------------------------- #
# The pattern axis (synthetic traffic)
# --------------------------------------------------------------------------- #
def test_a_synthetic_corpus_labels_every_fraud_case(engine, pool):
    result = run("Known", BALANCED, Corpus(kind=SYNTHETIC, hours=24, seed=5), engine, pool)
    rows = result.pattern_rows()
    assert rows and sum(row["cases"] for row in rows) == result.metrics["fraud_total"]
    assert rows[0]["pattern"] == "PaySim pattern"


def test_a_replay_window_has_no_pattern_breakdown(engine, pool):
    result = run("Balanced", BALANCED, Corpus(first_day=1, last_day=3), engine, pool)
    assert result.pattern_rows() == []


def test_the_new_pattern_slips_through_the_same_policy(engine, pool):
    """The engine was taught "the sender is emptied"; small transfers are not that."""
    base = dict(kind=SYNTHETIC, hours=48, seed=9, fraud_rate=0.05)
    known = run("Known pattern", SIEVE, Corpus(**base, new_pattern=False), engine, pool)
    new = run("New pattern", SIEVE, Corpus(**base, new_pattern=True), engine, pool)
    cmp = compare(known, new)
    assert cmp.paired and axes(known.scenario, new.scenario) == {"traffic"}
    assert new.metrics["fraud_caught"] < known.metrics["fraud_caught"]
    blind = [row for row in new.pattern_rows() if row["pattern"].startswith("Small transfers")]
    assert blind and blind[0]["rate"] == 0.0
    assert "invisible to the rules" in cmp.verdict


def test_the_pattern_comparison_leaves_the_fraud_count_alone(engine, pool):
    base = dict(kind=SYNTHETIC, hours=24, seed=4, fraud_rate=0.05)
    known = run("Known", BALANCED, Corpus(**base, new_pattern=False), engine, pool)
    new = run("New", BALANCED, Corpus(**base, new_pattern=True), engine, pool)
    assert known.metrics["fraud_total"] == new.metrics["fraud_total"]
