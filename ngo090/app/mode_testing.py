"""Mode 3, Transaction testing: score one transaction by hand, or a whole file."""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

import ui
from common import get_band_stats, render_costs_popover, render_policy_popover
from src.decision import Policy
from src.processing import MAX_ROWS, ProcessedBatch, process_frame, summarize_batch, template_csv
from src.tools import Toolkit
from src.verdict import band_for

TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
EXAMPLES = {
    "normal": {"tx_type": "TRANSFER", "tx_amount": 8_000.0, "tx_sender": 120_000.0, "tx_receiver": 60_000.0, "tx_hour": 14},
    "empty": {"tx_type": "TRANSFER", "tx_amount": 250_000.0, "tx_sender": 250_000.0, "tx_receiver": 0.0, "tx_hour": 3},
}
VIEWS = ("Single transaction", "Upload a file")
VIEW_KEY = "testing_view"        # which of the two views is open; kept in the session, not in the browser
FILE_BYTES = "uploaded_bytes"    # the chosen file, copied out of the widget (see _keep_upload)
FILE_NAME = "uploaded_name"
SINGLE_DEFAULTS = {"tx_type": "TRANSFER", "tx_amount": 250_000.0, "tx_sender": 250_000.0,
                   "tx_receiver": 0.0, "tx_hour": 3}
TX_KEPT = "tx_kept"              # the typed transaction, copied out of the widgets (see _render_single)


def _load_example(name: str) -> None:
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


def _keep_upload() -> None:
    """Copy the chosen file out of the uploader widget.

    Streamlit drops a widget's value when that widget is not rendered in a run, so a
    page switch would forget the file. A plain session key is not a widget, so the
    bytes survive navigation and the table is rebuilt from the cache for free.

    The callback also fires when the file is removed with the uploader's X: that is a
    deliberate "forget it", so the copy goes too -- otherwise the page falls back to
    the copy and shows the Clear row as if the file were still loaded.
    """
    file = st.session_state.get("upload")
    if file is None:
        st.session_state.pop(FILE_BYTES, None)
        st.session_state.pop(FILE_NAME, None)
        return
    st.session_state[FILE_BYTES] = file.getvalue()
    st.session_state[FILE_NAME] = file.name


def _clear_upload() -> None:
    st.session_state.pop("upload", None)
    st.session_state.pop(FILE_BYTES, None)
    st.session_state.pop(FILE_NAME, None)


def _wire_scored_table() -> None:
    """The scored table's verdict filter and pager are plain HTML buttons; this runs
    their script. Paging in the browser keeps every click instant: no rerun, no blink."""
    try:
        import streamlit.components.v1 as components
        components.html(ui.scored_table_script(), height=0)
    except Exception:                               # a cosmetic touch must never break the page
        pass


def _keep_view() -> None:
    """A segmented control deselects when its active option is clicked again; hold the view."""
    if st.session_state.get(VIEW_KEY) is None:
        st.session_state[VIEW_KEY] = st.session_state.get("last_view", VIEWS[0])
    st.session_state["last_view"] = st.session_state[VIEW_KEY]


@st.cache_data(show_spinner="Scoring the file...")
def process_upload(tag: str, content: bytes, block: float, review_min: float, capacity: int) -> ProcessedBatch:
    """Score an uploaded file (cached, so changing a cost slider does not rescore it)."""
    from common import get_engine
    df = pd.read_csv(io.BytesIO(content))
    policy = Policy(block_threshold=block, review_min_score=review_min, daily_capacity=capacity)
    return process_frame(get_engine(tag), df, policy)


def _seed_single() -> None:
    """Put the typed transaction back into the widgets.

    Streamlit drops a widget's value when that widget is not rendered in a run, so
    switching to the upload view (or to another page) leaves the five keys unset and
    the form would come back with its defaults. The kept copy is a plain dict, not a
    widget, so it survives; setdefault only fills what the runtime dropped.
    """
    kept = st.session_state.get(TX_KEPT) or {}
    for key, default in SINGLE_DEFAULTS.items():
        st.session_state.setdefault(key, kept.get(key, default))


def _remember_single() -> None:
    """Copy what is on screen now (defaults, an example or typing) out of the widgets."""
    st.session_state[TX_KEPT] = {key: st.session_state[key] for key in SINGLE_DEFAULTS}


def _render_single(tag: str, engine, policy: Policy) -> None:
    st.markdown(ui.mode_head("Single transaction", "Type in a transaction and see its risk, the verdict, the decision and the reasons."),
                unsafe_allow_html=True)
    _seed_single()

    c = st.columns(5)
    c[0].selectbox("Type", TYPES, key="tx_type")
    c[1].number_input("Amount", min_value=0.0, step=1_000.0, format="%.2f", key="tx_amount")
    c[2].number_input("Sender balance before", min_value=0.0, step=1_000.0, format="%.2f", key="tx_sender")
    c[3].number_input("Receiver balance before", min_value=0.0, step=1_000.0, format="%.2f", key="tx_receiver")
    c[4].slider("Hour of day", 0, 23, key="tx_hour")
    # The next run may not render these widgets at all, and then only this copy is left.
    _remember_single()

    b = st.columns([1, 1.6, 3.4])
    b[0].button("Example: normal transfer", on_click=_load_example, args=("normal",))
    b[1].button("Example: large transfer to an empty account", on_click=_load_example, args=("empty",))
    # Money leaving an account that cannot cover it. Not CASH_IN: there the money comes in, so a low
    # balance is normal. This is a hint only, never a validation: the engine scores what is typed.
    if st.session_state.tx_type != "CASH_IN" and st.session_state.tx_amount > st.session_state.tx_sender:
        st.markdown(ui.unfunded_tx_hint_html(st.session_state.tx_amount, st.session_state.tx_sender),
                    unsafe_allow_html=True)
    if "oldbalanceOrg" in engine.drop:
        st.caption("This engine does not use the sender balance (it is removed in the recommended setting), "
                   "so changing it has no effect on the score.")

    tx = {"type": st.session_state.tx_type, "amount": st.session_state.tx_amount,
          "oldbalanceOrg": st.session_state.tx_sender, "oldbalanceDest": st.session_state.tx_receiver,
          "hour": st.session_state.tx_hour}
    try:
        # The same tool an agent or a script calls, not a UI-only shortcut.
        prediction = Toolkit(engine, policy=policy).explain_transaction(tx)
    except ValueError as exc:
        st.error(str(exc))
        return
    # The score is a percentile; the band says how often that percentile was fraud.
    stats = get_band_stats(tag) if prediction.applicable else []
    band = band_for(prediction.risk, stats) if stats else None
    st.markdown(ui.prediction_html(prediction, policy, band, sum(s.total for s in stats)),
                unsafe_allow_html=True)


def _render_file(tag: str, policy: Policy, costs) -> None:
    st.markdown(ui.mode_head("Upload a file", f"Score up to {MAX_ROWS:,} transactions at once from a CSV file."),
                unsafe_allow_html=True)
    st.markdown(
        "Required columns: `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`. Optional: `hour` (0-23) or `step`, "
        "and `isFraud` (0 or 1) if you know the outcomes, which adds detection results and the money impact. "
        "Header names are not case sensitive and other columns are ignored.")
    st.download_button("Download a template file", data=template_csv(), file_name="transactions_template.csv",
                       mime="text/csv")
    uploaded = st.file_uploader("CSV file", type=["csv"], key="upload", label_visibility="collapsed",
                                on_change=_keep_upload)
    # The widget is gone after leaving the page; the copy in the session is not.
    content = uploaded.getvalue() if uploaded is not None else st.session_state.get(FILE_BYTES)
    name = uploaded.name if uploaded is not None else st.session_state.get(FILE_NAME)
    if content is None:
        st.info("Choose a file to score it. Review capacity is not applied to uploaded files, "
                "because it needs transactions spread over time.")
        return
    if uploaded is None:
        kept = st.columns([7, 1], vertical_alignment="center")
        kept[0].caption(f"Showing **{name}**, the file chosen earlier on this page.")
        kept[1].button("Clear", on_click=_clear_upload, key="clear_upload")

    try:
        batch = process_upload(tag, content, float(policy.block_threshold),
                               float(policy.review_min_score), int(policy.daily_capacity))
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("This file could not be read as a CSV. Check that it has a header row and comma separators.")
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    if batch.rows.empty:
        st.error("No valid rows were found in the file.")
    else:
        stats = get_band_stats(tag)
        st.markdown(ui.batch_summary_html(summarize_batch(batch, costs)), unsafe_allow_html=True)
        st.markdown(ui.scored_table_html(batch.rows, bands=stats, n_total=sum(s.total for s in stats)),
                    unsafe_allow_html=True)
        _wire_scored_table()
        out = batch.rows.rename(columns={"reason": "main_reason"}).drop(columns=["applicable"])
        out["verdict"] = [(band_for(float(r.risk), stats).label if getattr(r, "applicable", True) else "not covered")
                          for r in batch.rows.itertuples()]
        st.download_button("Download the scored file", data=out.to_csv(index=False),
                           file_name="scored_transactions.csv", mime="text/csv", key="scored_download")
    if not batch.rejected.empty:
        with st.expander(f"{len(batch.rejected):,} rows were rejected"):
            st.dataframe(batch.rejected.head(200), hide_index=True)


def render(header, tag: str, run, engine) -> None:
    """Draw the whole testing mode. `header` is the placeholder at the top of the page."""
    header.markdown(ui.header_html("testing", "Same engine as the replay, one transaction at a time",
                                  "Transaction testing"),
                    unsafe_allow_html=True)
    cols = st.columns([1.5, 1.2, 4])
    with cols[0]:
        policy = render_policy_popover()
    with cols[1]:
        costs = render_costs_popover()

    # A server-side view switch, not st.tabs: tabs keep their selection in the browser, so the
    # rerun caused by choosing a file snapped the page back to the first tab.
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key=VIEW_KEY,
                                label_visibility="collapsed", on_change=_keep_view) or VIEWS[0]
    if view == VIEWS[0]:
        _render_single(tag, engine, policy)
    else:
        _render_file(tag, policy, costs)