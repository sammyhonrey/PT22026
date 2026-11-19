"""Mode 4, Scenario comparison: two policies or two worlds, one set of measures.

The page is a read: it scores a frozen set of transactions twice and subtracts
the results. Nothing here changes the policy in the sidebar, opens a case or
writes an audit line -- if a comparison convinces the analyst, they can copy
the winner's settings into the Decision policy themselves.

Every pair the page offers changes exactly one axis:

    policy     Balanced vs Strict, on identical transactions and scores.
    traffic    normal traffic vs a market shift: same transactions, the model
               asked about a world where amounts are 10x and receivers are new.
    pattern    known fraud vs a new fraud shape, from the same seed.

The numbers live in src/scenario.py (pure Python, tested); this file only
collects the settings and draws them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace

import streamlit as st

import ui
from common import get_engine, render_costs_popover
from src.decision import PRESETS, Policy
from src.monitor import hour_label
from src.replay import day_of_step, default_days
from src.scenario import (
    MEASURES, REPLAY, SYNTHETIC, Corpus, Scenario, Comparison, compare, run_scenario,
)

PREF_KEYS = ("pairing", "cmp_days", "cmp_sample", "cmp_hours", "cmp_fraud", "cmp_seed",
             "a_preset", "a_block", "a_cap", "a_review",
             "b_preset", "b_block", "b_cap", "b_review")

BALANCED = PRESETS["Balanced"]
STRICT_SAME_CAPACITY = replace(PRESETS["Strict"], daily_capacity=BALANCED.daily_capacity)


@dataclass(frozen=True)
class Pairing:
    """A ready-made comparison: what each side is, and which axis moves."""

    left: str
    right: str
    left_policy: Policy
    right_policy: Policy
    source: str = REPLAY
    shift_right: bool = False
    new_pattern_right: bool = False
    note: str = ""


PAIRINGS = {
    "Balanced vs Strict": Pairing(
        "Balanced", "Strict", BALANCED, PRESETS["Strict"],
        note="Strict blocks at a lower risk and staffs twice the reviews, so it changes "
             "three settings at once: the threshold, the review cut-off and the capacity."),
    "Balanced vs Strict, same capacity": Pairing(
        "Balanced", "Strict, same capacity", BALANCED, STRICT_SAME_CAPACITY,
        note="The staffing is identical on both sides, so this isolates the threshold and "
             "the review cut-off."),
    "Normal traffic vs market shift": Pairing(
        "Normal traffic", "Market shift", BALANCED, BALANCED, shift_right=True,
        note="One policy, one set of transactions, two worlds: from the middle of the window "
             "amounts are 10x larger and receivers are new accounts. Money is still counted on "
             "what the transactions were worth."),
    "Known fraud vs new fraud pattern": Pairing(
        "Known pattern", "New pattern", BALANCED, BALANCED, source=SYNTHETIC, new_pattern_right=True,
        note="One policy, one set of transactions, two fraud shapes: half of the fraud switches "
             "from emptied accounts to small transfers between established ones."),
}
CUSTOM = "Custom (pick both sides)"


# --------------------------------------------------------------------------- #
# Settings that must survive a visit to another page
# --------------------------------------------------------------------------- #
def _restore_prefs() -> None:
    for key, value in st.session_state.get("_compare_prefs", {}).items():
        st.session_state.setdefault(key, value)


def _save_prefs() -> None:
    st.session_state["_compare_prefs"] = {k: st.session_state[k] for k in PREF_KEYS
                                          if k in st.session_state}


def _policy_keys(side: str, policy: Policy, preset: str) -> dict:
    return {f"{side}_preset": preset,
            f"{side}_block": 100.0 if policy.block_threshold == float("inf") else float(policy.block_threshold),
            f"{side}_cap": int(policy.daily_capacity),
            f"{side}_review": int(policy.review_min_score)}


def _load_pairing() -> None:
    """The pairing changed: put its two policies into the editors."""
    spec = PAIRINGS.get(st.session_state.get("pairing", ""))
    if spec is None:                                  # Custom: leave the editors alone
        return
    for side, policy, preset in (("a", spec.left_policy, spec.left),
                                 ("b", spec.right_policy, spec.right)):
        st.session_state.update(_policy_keys(side, policy, _preset_name(preset, policy)))


def _preset_name(label: str, policy: Policy) -> str:
    for name, preset in PRESETS.items():
        if preset == policy:
            return name
    return "Custom"


def _apply_preset(side: str) -> None:
    name = st.session_state.get(f"{side}_preset", "Custom")
    preset = PRESETS.get(name)
    if preset is not None:
        for key, value in _policy_keys(side, preset, name).items():
            st.session_state[key] = value


def _mark_custom(side: str) -> None:
    st.session_state[f"{side}_preset"] = "Custom"


def _new_sample() -> None:
    st.session_state["cmp_sample"] = random.randrange(1, 1_000_000)


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
def _policy_editor(side: str, title: str) -> Policy:
    """Three sliders per side, under a preset that fills them."""
    st.markdown(f'<div class="fa-root"><div class="fa-sub2">{title}</div></div>', unsafe_allow_html=True)
    st.selectbox("Preset", [*PRESETS, "Custom"], key=f"{side}_preset",
                 on_change=_apply_preset, args=(side,),
                 help="Fills the three settings below. Moving a slider switches to Custom.")
    st.slider("Auto-block above risk score", 90.0, 100.0, 99.5, 0.1, key=f"{side}_block",
              on_change=_mark_custom, args=(side,), help="100 = never block automatically.")
    st.slider("Daily review capacity", 0, 1000, 150, 10, key=f"{side}_cap",
              on_change=_mark_custom, args=(side,))
    st.slider("Only review above risk score", 80, 99, 90, 1, key=f"{side}_review",
              on_change=_mark_custom, args=(side,))
    block = float(st.session_state[f"{side}_block"])
    return Policy(block_threshold=float("inf") if block >= 100 else block,
                  review_min_score=float(st.session_state[f"{side}_review"]),
                  daily_capacity=int(st.session_state[f"{side}_cap"]), capacity_mode="rolling")


def _corpus_controls(spec: Pairing, slot, day_lo: int, day_hi: int) -> tuple[Corpus, Corpus]:
    """The traffic of both sides, from the popover in the controls row (`slot`)."""
    if spec.source == SYNTHETIC:
        with slot.popover("Synthetic traffic"):
            hours = st.slider("Hours of traffic", 12, 168, 48, 12, key="cmp_hours",
                              help="Generated fresh from a fixed seed, so both sides see the same rows.")
            rate = st.slider("Share of fraud (%)", 0.5, 10.0, 3.0, 0.5, key="cmp_fraud")
            seed = st.number_input("Seed", 1, 999_999, 11, 1, key="cmp_seed",
                                   help="The same seed always draws the same traffic.")
        base = dict(kind=SYNTHETIC, hours=int(hours), volume_per_hour=300,
                    fraud_rate=float(rate) / 100.0, seed=int(seed))
        return Corpus(**base, new_pattern=False), Corpus(**base, new_pattern=spec.new_pattern_right)

    with slot.popover("Transactions"):
        if day_lo < day_hi:
            days = st.slider("Replay days", day_lo, day_hi, default_days(day_lo, day_hi), key="cmp_days",
                             help="The window both sides are scored on.")
        else:
            days = (day_lo, day_hi)
        sample = st.number_input("Sample number (0 = every transaction)", 0, 999_999, 0, 1,
                                 key="cmp_sample",
                                 help="A number replays the same random 80% of the window every "
                                      "time -- on this page and in Historical replay. Comparisons "
                                      "are frozen so both sides see the same rows.")
        st.button("Draw a new sample", on_click=_new_sample,
                  help="A new sample is a new set of transactions: the same comparison on other "
                       "traffic, which is how you see whether the difference holds.")
        st.caption("Differences between two samples of the same traffic are the noise floor of "
                   "every number on this page.")
    base = dict(kind=REPLAY, first_day=int(days[0]), last_day=int(days[1]), sample=int(sample))
    return Corpus(**base), Corpus(**base, shift=spec.shift_right)


# --------------------------------------------------------------------------- #
# Drawing a result
# --------------------------------------------------------------------------- #
def _signed_money(x: float) -> str:
    return ("+" if x > 0 else "") + ui.fmt_money(x)


def _tone(delta: float, better: str) -> str:
    if abs(delta) < 1e-9:
        return "flat"
    good = delta > 0 if better == "higher" else delta < 0
    return "up" if good else "down"


def _cards(cmp: Comparison) -> list[dict]:
    cards = []
    for side, result, eyebrow in (("a", cmp.left, "A"), ("b", cmp.right, "B")):
        m = result.metrics
        stats = [
            ("Fraud caught", f"{m['fraud_caught']:,} of {m['fraud_total']:,}"),
            ("Caught rate", f"{m['recall']:.0%}"),
            ("Alerts raised", f"{m['reviewed'] + m['blocked']:,}"),
            ("Net benefit", ui.fmt_money(m["net_benefit"])),
        ]
        corpus = result.scenario.describe()
        variant = result.scenario.corpus.variant()
        cards.append({
            "eyebrow": f"{eyebrow} · {result.name}",
            "name": result.name,
            "corpus": corpus + (f" · {variant}" if variant else ""),
            "policy": result.scenario.policy_line(),
            "stats": stats,
        })
    return cards


def _table_rows(cmp: Comparison) -> list[dict]:
    rows = []
    for key, label, unit, better in MEASURES:
        left, right, delta = cmp.left.metrics[key], cmp.right.metrics[key], cmp.deltas[key]
        fmt = ui.fmt_money if unit == "money" else ui.fmt_count
        diff = _signed_money(delta) if unit == "money" else f"{delta:+,.0f}"
        rows.append({"label": label, "left": fmt(left), "right": fmt(right), "diff": diff,
                     "tone": _tone(delta, better), "strong": key == "net_benefit"})
    return rows


def _decomposition(cmp: Comparison) -> tuple[list[dict], str]:
    saved = cmp.decomposition["more caught"]
    reviews = cmp.decomposition["more reviews"]
    friction = cmp.decomposition["more blocked customers"]
    items = [
        {"label": "Fraud stopped", "value": saved, "text": _signed_money(saved), "tone": _tone(saved, "higher")},
        {"label": "Review cost", "value": reviews, "text": _signed_money(reviews), "tone": _tone(reviews, "higher")},
        {"label": "Blocked customers", "value": friction, "text": _signed_money(friction),
         "tone": _tone(friction, "higher")},
    ]
    return items, _signed_money(cmp.decomposition["net"])


def _overlay(cmp: Comparison) -> str:
    """Net benefit over the simulated hours, both scenarios on one axis."""
    left, right = cmp.left.cumulative_net(), cmp.right.cumulative_net()
    steps = sorted(set(left.index) | set(right.index))
    if not steps:
        return ""
    labels = [hour_label(int(s)) for s in steps]
    series = [
        {"name": cmp.left.name, "color": ui.TEAL, "values": left.reindex(steps).ffill().fillna(0.0).tolist()},
        {"name": cmp.right.name, "color": ui.BLUE, "values": right.reindex(steps).ffill().fillna(0.0).tolist()},
    ]
    legend = "".join(f'<i style="background:{s["color"]}"></i>{s["name"]}&nbsp;&nbsp;' for s in series)
    return ui.card("Net benefit over the simulated hours",
                   ui.area_chart(labels, series, "cmp", height=240), "cash", legend=legend)


def _pattern_rows(cmp: Comparison) -> list[dict]:
    left = {row["pattern"]: row for row in cmp.left.pattern_rows()}
    right = {row["pattern"]: row for row in cmp.right.pattern_rows()}
    if not left and not right:
        return []
    rows = []

    def cell(row) -> str:
        return "no cases" if row is None else f"{row['caught']} of {row['cases']} ({row['rate']:.0%})"

    for pattern in list(dict.fromkeys([*left, *right])):
        cases = f"{left[pattern]['cases'] if pattern in left else 0} / " \
                f"{right[pattern]['cases'] if pattern in right else 0}"
        rows.append({"pattern": pattern, "cases": cases,
                     "left": cell(left.get(pattern)), "right": cell(right.get(pattern))})
    return rows


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
def render(header, tag: str, run) -> None:
    _restore_prefs()
    pool = run.replay
    day_lo, day_hi = day_of_step(int(pool["step"].min())), day_of_step(int(pool["step"].max()))
    st.session_state.setdefault("pairing", "Balanced vs Strict")
    if "a_block" not in st.session_state:          # first visit: the pairing fills both sides
        _load_pairing()
    pending = st.session_state.pop("_copy_to_b", None)
    if pending:
        st.session_state.update(pending)

    header.markdown(ui.header_html("compare", "Two scenarios · one engine · one set of measures",
                                  "Scenario comparison"), unsafe_allow_html=True)

    # ---- controls: one toolbar, everything on the same baseline ------------ #
    bar = st.columns([2.4, 1.3, 1.1], vertical_alignment="bottom")
    pairing = bar[0].selectbox("Comparison", [*PAIRINGS, CUSTOM], key="pairing",
                               on_change=_load_pairing)
    if pairing in PAIRINGS:
        spec = PAIRINGS[pairing]
    else:
        choice = bar[1].selectbox("Traffic", ["Replay window", "Synthetic"], key="cmp_source")
        source = SYNTHETIC if choice == "Synthetic" else REPLAY
        spec = Pairing("A", "B", BALANCED, STRICT_SAME_CAPACITY, source=source)
    with bar[2]:
        costs = render_costs_popover()
    corpus_a, corpus_b = _corpus_controls(spec, bar[1], day_lo, day_hi)

    with st.container(key="cmp_sides"):
        with st.expander("The two sides", expanded=False):
            cols = st.columns(2)
            with cols[0]:
                policy_a = _policy_editor("a", "A")
            with cols[1]:
                policy_b = _policy_editor("b", "B")
            if pairing != CUSTOM and st.button("Copy A's settings to B", key="copy_a_to_b"):
                # Writing a widget key after its widget exists is refused by Streamlit:
                # hand the values to the next run instead.
                st.session_state["_copy_to_b"] = _policy_keys("b", policy_a, "Custom")
                st.rerun()

    scenario_a = Scenario(spec.left if pairing != CUSTOM else "A", policy_a, corpus_a, costs)
    scenario_b = Scenario(spec.right if pairing != CUSTOM else "B", policy_b, corpus_b, costs)

    left = _run(tag, scenario_a, get_engine(tag), pool)
    right = _run(tag, scenario_b, get_engine(tag), pool)
    cmp = compare(left, right)

    # ---- what happened before, in one line -------------------------------- #
    note = spec.note if pairing in PAIRINGS else ""
    if note:
        st.caption(note)
    if not cmp.changed:
        st.info("The two sides are identical — change one setting on either side to compare.")

    for warning in cmp.warnings:
        st.warning(warning, icon=":material/warning:")

    # ---- the measures ----------------------------------------------------- #
    st.markdown(ui.compare_cards_html(_cards(cmp)), unsafe_allow_html=True)
    items, net = _decomposition(cmp)
    st.markdown(ui.two_cards(
        ui.card("Side by side", ui.comparison_table_html(
            _table_rows(cmp), cmp.left.name, cmp.right.name), "bars"),
        ui.card("Where the difference comes from",
                ui.decomposition_html(items, net), "cash")), unsafe_allow_html=True)

    st.markdown(_overlay(cmp), unsafe_allow_html=True)

    patterns = _pattern_rows(cmp)
    if patterns:
        st.markdown(ui.card("Catch rate by fraud shape",
                            ui.pattern_table_html(patterns, cmp.left.name, cmp.right.name),
                            "pie"), unsafe_allow_html=True)
    st.caption("A synthetic corpus knows which fraud shape each case belongs to. The replay "
               "file does not, so this table only appears when the traffic is generated.")

    _save_prefs()


@st.cache_data(show_spinner="Scoring both scenarios...")
def _run(tag: str, scenario: Scenario, _engine, _pool):
    """Score one scenario (cached: changing the other side does not rescore this one)."""
    return run_scenario(scenario, _engine, _pool)
