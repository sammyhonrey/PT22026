"""Two scenarios, the same engine, one screen: what actually changes.

A scenario is four things: the traffic, the scores, the policy, the costs.
Comparing two of them only means something if **one** of those differs, and
if both sides saw the **same** transactions. This module enforces both.

What is compared
----------------
    policy axis    Balanced vs Strict, on identical transactions and scores.
    world axis     normal traffic vs a market shift: the same transactions,
                   but the model inputs change halfway (amounts 10x, receivers
                   new), so the same model is asked about a different world.
    pattern axis   known fraud vs a new fraud shape (the generator's "small
                   transfers"), drawn from the same seed: same traffic, same
                   fraud rows, half of them reshaped.

The six numbers an analyst argues about come from `decision.evaluate`:
fraud caught, false alarms, review workload, amount saved, missed fraud and
net benefit. Nothing here recomputes money: the comparison is a subtraction.

Paired or it is noise
---------------------
`Corpus.pair_id` names the traffic a scenario draws. Two scenarios with the
same pair_id saw the same rows in the same order, which is what makes the
bootstrap on the net-benefit difference (`net_delta_interval`) meaningful.
When the pair_ids differ, the comparison still renders -- but it says so,
because the difference then includes sampling noise rather than policy.

Money stays on the original amounts. A market shift changes what the model
sees (and therefore its scores), not what the transactions were worth: a
report that multiplied every amount by ten would "save" ten times as much
for the same decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.artifacts import score_rows
from src.decision import APPROVE, CostConfig, Policy, decide, evaluate, row_outcomes
from src.generator import GeneratorConfig, TransactionGenerator
from src.replay import SAMPLE_FRACTION, select_window, window_bounds

REPLAY, SYNTHETIC = "replay", "synthetic"

# (key, label, unit, better) -- unit is "count" or "money"; better is which
# direction is an improvement. The page colours the difference with `better`.
MEASURES = (
    ("fraud_caught", "Fraud caught", "count", "higher"),
    ("false_alarms", "False alarms", "count", "lower"),
    ("reviewed", "Review workload", "count", "lower"),
    ("fraud_amount_saved", "Amount saved", "money", "higher"),
    ("fraud_missed", "Missed fraud", "count", "lower"),
    ("net_benefit", "Net benefit", "money", "higher"),
)


# --------------------------------------------------------------------------- #
# What gets scored
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Corpus:
    """The traffic of a scenario, and how its scores are produced.

    `pair_id` deliberately leaves out the axis being compared: a market shift
    or a new fraud pattern is drawn from the same rows, so those variants are
    still a paired comparison. What it does include -- the days, the sample
    number, the synthetic seed -- is what decides *which* rows those are.
    """

    kind: str = REPLAY
    first_day: int = 1
    last_day: int = 3
    sample: int = 0                 # replay: 0 = every transaction, >0 = that random sample
    shift: bool = False             # model inputs change from the middle on
    hours: int = 72                 # synthetic only
    volume_per_hour: int = 300
    fraud_rate: float = 0.03
    new_pattern: bool = False
    seed: int = 1                   # synthetic only

    def __post_init__(self) -> None:
        if self.kind not in (REPLAY, SYNTHETIC):
            raise ValueError(f"corpus kind must be {REPLAY!r} or {SYNTHETIC!r}, got {self.kind!r}")
        if self.kind == REPLAY and self.first_day > self.last_day:
            raise ValueError("the first replay day must not be after the last one.")
        if self.kind == SYNTHETIC and self.hours < 1:
            raise ValueError("a synthetic corpus needs at least one hour.")
        if self.kind == SYNTHETIC and not 0.0 <= self.fraud_rate <= 1.0:
            raise ValueError("fraud_rate must be between 0 and 1.")

    @property
    def pair_id(self) -> tuple:
        """Identity of the transactions drawn, ignoring the axis under test."""
        if self.kind == REPLAY:
            return (REPLAY, self.first_day, self.last_day, self.sample)
        return (SYNTHETIC, self.seed, self.hours, self.volume_per_hour, round(self.fraud_rate, 6))

    def describe(self) -> str:
        if self.kind == REPLAY:
            sample = "every transaction" if not self.sample else f"sample #{self.sample:06d}"
            return f"Replay · days {self.first_day}–{self.last_day} · {sample}"
        return f"Synthetic · {self.hours} h · {self.fraud_rate:.1%} fraud"

    def variant(self) -> str:
        """The one thing about this traffic that is not its own description."""
        if self.shift:
            return "market shifts halfway through"
        if self.new_pattern:
            return "half the fraud changes shape"
        return ""


@dataclass(frozen=True)
class Scenario:
    """A name, a policy and the traffic it is applied to."""

    name: str
    policy: Policy
    corpus: Corpus = Corpus()
    costs: CostConfig = CostConfig()

    def describe(self) -> str:
        return self.corpus.describe()

    def policy_line(self) -> str:
        return policy_line(self.policy)


def policy_line(policy: Policy) -> str:
    """Plain words for a policy, the same everywhere it is shown."""
    block = "never blocks" if policy.block_threshold == float("inf") else \
        f"blocks at risk {policy.block_threshold:g}"
    reviews = f"{policy.daily_capacity} reviews a day" if policy.daily_capacity else "no reviews"
    return f"{block} · {reviews} · reviews above {policy.review_min_score:g}"


# --------------------------------------------------------------------------- #
# Running one scenario
# --------------------------------------------------------------------------- #
def corpus_frame(corpus: Corpus, pool: pd.DataFrame) -> pd.DataFrame:
    """The transactions a scenario is scored on, in step order.

    A replay window is cut from the run's replay pool and, if a sample number
    is given, thinned exactly the way the replay page thins it (same seed,
    same fraction), so "sample #203674" means the same rows on both pages.
    A synthetic corpus is generated fresh from its seed.
    """
    if corpus.kind == SYNTHETIC:
        config = GeneratorConfig(volume_per_hour=corpus.volume_per_hour,
                                 fraud_rate=corpus.fraud_rate,
                                 new_pattern=corpus.new_pattern)
        frame = TransactionGenerator(corpus.seed).generate(0, corpus.hours * 60, config)
    else:
        first, last = int(pool["step"].min()), int(pool["step"].max())
        start, end = window_bounds(first, last, corpus.first_day, corpus.last_day)
        frame = select_window(pool, start, end)
        if corpus.sample:
            keep = np.random.default_rng(corpus.sample).random(len(frame)) < SAMPLE_FRACTION
            if not keep.any():
                keep[0] = True
            frame = frame[keep].reset_index(drop=True)
    if frame.empty:
        raise ValueError("the chosen corpus contains no transactions.")
    return frame.sort_values("step", kind="stable").reset_index(drop=True)


def shift_step(frame: pd.DataFrame) -> int:
    """The hour from which the model inputs change (the middle of the window)."""
    first, last = int(frame["step"].min()), int(frame["step"].max())
    return first + (last - first + 1) // 2


@dataclass
class ScenarioResult:
    """Everything one scenario produced, ready to diff against another."""

    scenario: Scenario
    df: pd.DataFrame
    risk: np.ndarray
    action: pd.Series
    outcomes: pd.DataFrame
    metrics: dict

    @property
    def name(self) -> str:
        return self.scenario.name

    @property
    def caught_rate(self) -> float:
        return self.metrics["recall"]

    @property
    def flag_rate(self) -> float:
        flagged = self.metrics["reviewed"] + self.metrics["blocked"]
        return flagged / self.metrics["transactions"] if self.metrics["transactions"] else 0.0

    @property
    def signature(self) -> tuple:
        return (len(self.df), int(self.df["isFraud"].sum()),
                round(float(self.df["amount"].sum()), 2))

    def cumulative_net(self) -> pd.Series:
        """Running net benefit per simulated hour, for the overlay chart."""
        return self.outcomes.groupby("step")["net_benefit"].sum().cumsum()

    def pattern_rows(self) -> list[dict]:
        """Catch rate per fraud pattern -- only when the traffic carries labels.

        The replay file has no pattern column, so the policy axis has nothing
        to break down. A synthetic corpus does, and that breakdown is the whole
        point of comparing a known pattern with a new one.
        """
        if "pattern" not in self.df.columns:
            return []
        fraud = self.df["isFraud"].to_numpy() == 1
        if not fraud.any():
            return []
        flagged = self.action.to_numpy() != APPROVE
        labels = self.df.loc[fraud, "pattern"].replace("", "Unlabelled")
        frame = pd.DataFrame({"pattern": labels.to_numpy(), "caught": flagged[fraud]})
        out = []
        for pattern, group in frame.groupby("pattern", sort=False):
            total = int(len(group))
            caught = int(group["caught"].sum())
            out.append({"pattern": str(pattern), "cases": total, "caught": caught,
                        "rate": caught / total if total else 0.0})
        return sorted(out, key=lambda r: -r["cases"])


def run_scenario(scenario: Scenario, engine, pool: pd.DataFrame) -> ScenarioResult:
    """Score the traffic, apply the policy, and measure the result."""
    frame = corpus_frame(scenario.corpus, pool)
    shift_from = shift_step(frame) if scenario.corpus.shift else None
    # score_rows shifts a copy of the frame for the model, so the money below
    # is still computed on what the transactions were actually worth.
    risk = np.asarray(score_rows(engine, frame, shift_from).risk, dtype=float)
    action = decide(frame, risk, scenario.policy)
    return ScenarioResult(
        scenario=scenario, df=frame, risk=risk, action=action,
        outcomes=row_outcomes(frame, action, scenario.costs),
        metrics=evaluate(frame, action, scenario.costs),
    )


# --------------------------------------------------------------------------- #
# Comparing two of them
# --------------------------------------------------------------------------- #
@dataclass
class Comparison:
    left: ScenarioResult
    right: ScenarioResult
    deltas: dict
    decomposition: dict
    changed: list[str]
    warnings: list[str]
    interval: tuple[float, float] | None
    paired: bool
    verdict: str


def _cost_changes(a: CostConfig, b: CostConfig) -> list[str]:
    if a == b:
        return []
    parts = []
    if a.review_cost != b.review_cost:
        parts.append(f"review cost {a.review_cost:,.0f}")
    if a.friction_cost != b.friction_cost:
        parts.append(f"friction cost {a.friction_cost:,.0f}")
    if a.review_catch_rate != b.review_catch_rate:
        parts.append(f"catch rate {a.review_catch_rate:.0%}")
    return [f"the cost settings ({', '.join(parts)})"]


def _corpus_changes(a: Corpus, b: Corpus) -> list[str]:
    if a.pair_id != b.pair_id:
        return ["the transactions themselves"]
    if a.shift != b.shift:
        return ["the traffic (a market shift on one side)"]
    if a.new_pattern != b.new_pattern:
        return ["the fraud pattern"]
    return []


def _policy_changes(a: Policy, b: Policy) -> list[str]:
    out = []
    if a.block_threshold != b.block_threshold:
        out.append(f"the auto-block threshold ({a.block_threshold:g} → {b.block_threshold:g})")
    if a.daily_capacity != b.daily_capacity:
        out.append(f"the review capacity ({a.daily_capacity} → {b.daily_capacity} a day)")
    if a.review_min_score != b.review_min_score:
        out.append(f"the review cut-off ({a.review_min_score:g} → {b.review_min_score:g})")
    return out


def changes(a: Scenario, b: Scenario) -> list[str]:
    """Everything that differs between the two scenarios, in plain words."""
    return _corpus_changes(a.corpus, b.corpus) + _policy_changes(a.policy, b.policy) \
        + _cost_changes(a.costs, b.costs)


def axes(a: Scenario, b: Scenario) -> set[str]:
    """Which of the three axes moved: traffic, policy, costs.

    Several settings can move inside one axis (Balanced and Strict differ in
    three knobs at once) and that is still a policy comparison. Two *axes*
    moving is different: no single cause can be named any more.
    """
    out = set()
    if a.corpus.pair_id != b.corpus.pair_id or a.corpus.shift != b.corpus.shift \
            or a.corpus.new_pattern != b.corpus.new_pattern:
        out.add("traffic")
    if a.policy != b.policy:
        out.add("policy")
    if a.costs != b.costs:
        out.add("costs")
    return out


def net_delta_interval(left: ScenarioResult, right: ScenarioResult,
                       n_boot: int = 400, seed: int = 7) -> tuple[float, float] | None:
    """A 90% bootstrap range for the net-benefit difference (paired rows).

    Resampling transactions answers "would this difference survive another
    draw of the same traffic?" -- which is the question a comparison has to
    answer before anyone quotes its headline number. None when the two sides
    did not see the same rows: then the interval would describe the wrong
    thing.
    """
    if left.scenario.corpus.pair_id != right.scenario.corpus.pair_id:
        return None
    if len(left.outcomes) != len(right.outcomes):
        return None
    delta = right.outcomes["net_benefit"].to_numpy() - left.outcomes["net_benefit"].to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    totals = delta[draws].sum(axis=1)
    return float(np.quantile(totals, 0.05)), float(np.quantile(totals, 0.95))


def _money(x: float) -> str:
    """Compact money for the verdict line (same shape as the dashboard)."""
    sign, x = ("-" if x < 0 else ""), abs(x)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= limit:
            return f"{sign}{x / limit:,.2f}{suffix}"
    return f"{sign}{x:,.0f}"


def _blind_spots(result: ScenarioResult) -> str:
    """A sentence about a fraud shape the policy mostly misses, if there is one."""
    weak = [row for row in result.pattern_rows() if row["cases"] >= 10 and row["rate"] < 0.10]
    if not weak:
        return ""
    row = weak[0]
    return (f" One shape is nearly invisible to the rules: {row['caught']} of "
            f"{row['cases']} {row['pattern'].lower()} caught ({row['rate']:.0%}).")


def _verdict(left: ScenarioResult, right: ScenarioResult, deltas: dict,
             interval: tuple[float, float] | None, paired: bool, moved: set[str]) -> str:
    """One or two sentences a person can repeat in a meeting."""
    if not paired:
        return (f"{left.name} and {right.name} did not see the same transactions, so this is two "
                "runs side by side, not a like-for-like comparison.")
    d_net = deltas["net_benefit"]
    if interval is not None and interval[0] <= 0.0 <= interval[1]:
        head = (f"On net benefit the two are inside the noise: the {_money(abs(d_net))} gap "
                f"between {left.name} and {right.name} could come from another sample of the "
                "same traffic.")
    else:
        win = right if d_net >= 0 else left
        head = f"{win.name} is ahead by {_money(abs(d_net))} on net benefit."

    if "traffic" in moved:
        body = (f"The difference comes from the traffic, not from the policy: {right.name} catches "
                f"{right.metrics['fraud_caught']:,} of {right.metrics['fraud_total']:,} fraud cases "
                f"against {left.metrics['fraud_caught']:,} of {left.metrics['fraud_total']:,}, "
                f"with {right.metrics['false_alarms']:,} false alarms against "
                f"{left.metrics['false_alarms']:,}.")
    else:
        sign = 1.0 if d_net >= 0 else -1.0
        caught = int(round(sign * (right.metrics["fraud_caught"] - left.metrics["fraud_caught"])))
        saved = sign * (right.metrics["fraud_amount_saved"] - left.metrics["fraud_amount_saved"])
        reviews = int(round(sign * (right.metrics["reviewed"] - left.metrics["reviewed"])))
        blocked = int(round(sign * (right.metrics["blocked"] - left.metrics["blocked"])))
        reviews_text = ("the same review workload" if reviews == 0 else
                        f"{abs(reviews):,} {'more' if reviews > 0 else 'fewer'} reviews")
        body = (f"{abs(caught):,} {'more' if caught > 0 else 'fewer'} fraud cases stopped "
                f"({_money(abs(saved))} {'more' if saved > 0 else 'less'} in value), for "
                f"{reviews_text} and {abs(blocked):,} "
                f"{'more' if blocked > 0 else 'fewer'} blocked customers.")
    return head + " " + body + _blind_spots(right)


def compare(left: ScenarioResult, right: ScenarioResult, n_boot: int = 400) -> Comparison:
    """Diff two finished scenarios: the six measures, and where the net went."""
    deltas = {key: right.metrics[key] - left.metrics[key] for key, *_ in MEASURES}
    d_saved = right.metrics["fraud_amount_saved"] - left.metrics["fraud_amount_saved"]
    d_review = right.metrics["review_cost_total"] - left.metrics["review_cost_total"]
    d_friction = right.metrics["friction_cost_total"] - left.metrics["friction_cost_total"]
    decomposition = {
        "more caught": d_saved,
        "more reviews": -d_review,
        "more blocked customers": -d_friction,
        "net": d_saved - d_review - d_friction,
    }

    paired = left.scenario.corpus.pair_id == right.scenario.corpus.pair_id
    changed = changes(left.scenario, right.scenario)
    moved = axes(left.scenario, right.scenario)
    warnings = []
    if not paired:
        warnings.append("The two sides did not see the same transactions, so part of the "
                        "difference is sampling noise, not the change you are testing.")
    if len(moved) > 1:
        warnings.append("Traffic, policy and costs all moved at once, so the difference "
                        "cannot be pinned on any single one.")
    if left.metrics["fraud_total"] < 50 or right.metrics["fraud_total"] < 50:
        warnings.append("This window holds fewer than 50 fraud cases, so read the counts "
                        "as a direction rather than a measurement.")

    interval = net_delta_interval(left, right, n_boot=n_boot) if paired else None
    return Comparison(
        left=left, right=right, deltas=deltas, decomposition=decomposition,
        changed=changed, warnings=warnings, interval=interval, paired=paired,
        verdict=_verdict(left, right, deltas, interval, paired, moved),
    )
