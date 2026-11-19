"""Shared session context for the five native Streamlit screens."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import pandas as pd

from .datasets import DatasetSpec
from .input_contract import INPUT_CONTRACT

# The names of the five screens, in the order shown in the menu.
PAGES = ["Overview", "Forecast Studio", "Data Hub", "Model Lab", "Help"]


@dataclass
class Workspace:
    """Everything one screen needs: dataset, model, scores, data and errors.

    The app builds one Workspace per run and passes it to every screen.
    If something fails (bad CSV, missing model), the matching *_error
    field holds a message and the screen can show it instead of crashing.
    """

    spec: DatasetSpec
    source_mode: str
    model_name: str
    bundle: dict | None
    metrics: pd.DataFrame
    hourly: pd.DataFrame | None
    source_label: str
    palette: str = "Energy Garden"
    reduce_motion: bool = False
    input_error: str | None = None
    bundle_error: str | None = None
    metrics_error: str | None = None
    release_id: str = "local"
    artifact_digests: dict = field(default_factory=dict)
    managed_release: bool = False

    @property
    def input_key(self) -> tuple:
        """Identity includes clock provenance, even for equivalent UTC records."""
        if self.hourly is None:
            return (self.spec.id, self.source_mode, None)
        digest = hashlib.sha256(pd.util.hash_pandas_object(self.hourly, index=True).values.tobytes()).hexdigest()
        clock = tuple(self.hourly.attrs[name] for name in (
            "timestamp_clock", "timestamp_input_clock", "timestamp_normalization"
        ))
        return (self.spec.id, self.source_mode, digest, clock)

    @property
    def widget_scope(self) -> str:
        """Short ID of the current input, used to keep each input's settings apart."""
        return hashlib.sha256(repr(self.input_key).encode()).hexdigest()[:20]


def forecast_selection(workspace: Workspace, controls: dict) -> tuple[str, pd.Timestamp, list[pd.Timestamp]]:
    """Restore a valid selection without depending on a mounted page widget.

    Returns three things: the view name, the forecast origin (the last hour
    we already know) and the list of origins the user may choose from.
    "Future forecast" starts after the last record, so it has no choices.
    "Historical backtest" starts earlier, so we can compare with real values.
    """
    hourly = workspace.hourly
    if hourly is None:
        raise ValueError("Load valid hourly data before choosing a forecast origin.")
    settings = controls.get(workspace.widget_scope, {})
    # A backtest needs enough rows to leave 24 real hours to compare with.
    can_backtest = len(hourly) >= INPUT_CONTRACT["minimum_backtest_rows"]
    view = settings.get("view", "Historical backtest" if can_backtest else "Future forecast")
    if not can_backtest or view not in {"Historical backtest", "Future forecast"}:
        view = "Future forecast"
    if view == "Future forecast":
        return view, hourly.index[-1], []
    # Possible origins: enough history before them, and 24 real hours after.
    candidates = hourly.index[INPUT_CONTRACT["minimum_history_rows"] - 1:-24]
    # Prefer origins at 23:00, so each choice is one full day.
    daily = candidates[candidates.hour == 23]
    choices = list(daily if not daily.empty else candidates)
    # Use the saved choice if it is still valid. Otherwise use the latest one.
    requested = settings.get("origin")
    origin = next((value for value in choices if str(value) == requested), choices[-1])
    return view, origin, choices


def forecast_key(workspace: Workspace, view: str, origin: pd.Timestamp) -> tuple:
    """Make an ID for one forecast. If any part changes, the old result is not reused."""
    return (*workspace.input_key, workspace.release_id, str(origin), workspace.model_name, view == "Historical backtest")


def matching_result(workspace: Workspace, controls: dict, saved: dict | None) -> pd.DataFrame | None:
    """Return the saved forecast only if it matches the current settings, else None."""
    if saved is None or workspace.hourly is None:
        return None
    view, origin, _ = forecast_selection(workspace, controls)
    return saved["data"] if saved.get("key") == forecast_key(workspace, view, origin) else None
