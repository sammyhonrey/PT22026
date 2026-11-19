"""EcoWatt AI: five native Streamlit screens with real Python inference."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from src import ui, views
from src.artifacts import ArtifactReleaseError, read_release, verified_bytes
from src.data import DataValidationError, validate_hourly_csv
from src.datasets import DATASETS, get_dataset
from src.forecasting import ModelArtifactError, load_model_bundle
from src.input_contract import INPUT_CONTRACT
from src.workspace import PAGES, Workspace

# Page setup. This must be the first Streamlit command.
st.set_page_config(page_title="EcoWatt AI · Energy Intelligence", page_icon="⚡", layout="wide",
                   initial_sidebar_state="collapsed")


# Cache: load the saved models once, not on every click.
@st.cache_resource
def get_bundle(artifact_path: Path, dataset_id: str, release_id: str, expected_sha256: str | None) -> dict:
    """Load and check the saved model file for one dataset (kept in cache)."""
    return load_model_bundle(artifact_path, expected_dataset_id=dataset_id, expected_sha256=expected_sha256)


# Cache: read the test scores table once.
@st.cache_data
def load_metrics(path: Path, release_id: str, expected_sha256: str | None) -> pd.DataFrame:
    """Read the test scores table and check it has the columns the app needs."""
    frame = pd.read_csv(io.BytesIO(verified_bytes(path, expected_sha256)))
    if not {"model", "mae_kwh", "rmse_kwh", "r2"}.issubset(frame.columns):
        raise ValueError("Benchmark metrics are missing required columns. Recreate the selected experiment.")
    return frame


# ---- Sidebar, part 1: page menu, colours, and dataset choice ----
# Apply the approved default once when an older browser session upgrades.
if st.session_state.get("_design_edition") != "energy-garden-1":
    st.session_state["appearance_palette"] = ui.DEFAULT_PALETTE
    st.session_state["_design_edition"] = "energy-garden-1"
with st.sidebar:
    ui.sidebar_brand()
    page = st.radio("Workspace", PAGES, key="workspace_page", label_visibility="collapsed")
    with st.expander("Colour & motion"):
        palette = st.selectbox("Colour palette", list(ui.PALETTES), key="appearance_palette")
        reduce_motion = st.toggle("Reduce motion", key="appearance_reduce_motion",
                                  help="Stop decorative motion and transitions. Your OS preference also takes precedence.")
        st.button("Replay page entrance", on_click=ui.replay, use_container_width=True,
                  help="Replays decoration only; no forecast or training is rerun.")
    ui.sidebar_step("01", "Choose your benchmark")
    dataset_id = st.selectbox("Benchmark dataset", list(DATASETS),
                              format_func=lambda value: DATASETS[value].label, key="benchmark_dataset")
spec = get_dataset(dataset_id)

# ---- Load the saved models and scores for the chosen dataset ----
# If something is missing, we keep the error text and show a clear message later.
release, release_error = None, None
try:
    release = read_release(spec.artifact_dir, dataset_id)
except (ArtifactReleaseError, OSError) as exc:
    release_error = str(exc)
bundle, bundle_error = None, release_error
metrics, metrics_error = pd.DataFrame(), None
if release is not None:
    try:
        bundle = get_bundle(spec.artifact_dir / "model_bundle.joblib", dataset_id, release.release_id,
                            release.digests["model_bundle.joblib"])
    except (ModelArtifactError, OSError) as exc:
        bundle_error = str(exc)
    try:
        metrics = load_metrics(spec.artifact_dir / "test_metrics.csv", release.release_id, release.digests["test_metrics.csv"])
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        metrics_error = str(exc)

# ---- Sidebar, part 2: data source and model choice ----
with st.sidebar:
    ui.sidebar_step("02", "Connect your data")
    source_mode = st.radio("Data source", ["Built-in dataset demo", "Upload hourly CSV"], key="source_mode")
    uploaded = None
    if source_mode == "Upload hourly CSV":
        st.caption(f"At least {INPUT_CONTRACT['minimum_history_rows']} ordered, consecutive hourly readings. UTF-8 CSV, 20 MiB maximum.")
        st.caption("XKY uses UTC; explicit offsets convert to UTC." if spec.input_clock_policy == "utc"
                   else "UCI preserves supplied wall-clock hours and any consistent offset.")
        uploaded = st.file_uploader("Hourly CSV", type=["csv"], key="hourly_upload")
    ui.sidebar_step("03", "Select a model")
    model_options = ["Seasonal Naive", "Ridge", "HistGradientBoosting"]
    recommended = bundle["selected_model"] if bundle else "HistGradientBoosting"
    model_key = f"model_{dataset_id}"
    if st.session_state.get(model_key) not in model_options:
        st.session_state[model_key] = recommended
    model_name = st.selectbox("Forecast model", model_options, key=model_key)
    if bundle:
        st.caption(f"Recommended: {recommended}. Selected by lowest validation MAE on {spec.short_name}.")
    ui.html('<div class="ew-sidebar-footer"><strong>One connected workspace</strong><br>Inspect → Forecast → Compare → Download.<br><br>Recorded data · No live meter feed</div>')

# ---- Read and check the hourly data (built-in demo or uploaded CSV) ----
# Bad files are caught here and shown as a friendly error, not a crash.
hourly, input_error = None, None
source_label = f"Attributed {spec.short_name} excerpt" if source_mode == "Built-in dataset demo" else "User-provided hourly series"
if source_mode == "Built-in dataset demo" or uploaded is not None:
    try:
        hourly = validate_hourly_csv(spec.demo_path if source_mode == "Built-in dataset demo" else uploaded,
                                     clock_policy=spec.input_clock_policy)
    except (DataValidationError, OSError) as exc:
        input_error = str(exc)

# Put everything the screens need into one object.
workspace = Workspace(spec=spec, source_mode=source_mode, model_name=model_name, bundle=bundle,
                      metrics=metrics, hourly=hourly, source_label=source_label, palette=palette,
                      reduce_motion=reduce_motion, input_error=input_error,
                      bundle_error=bundle_error, metrics_error=metrics_error,
                      release_id=release.release_id if release else "unavailable",
                      artifact_digests=release.digests if release else {}, managed_release=release.managed if release else False)
# ---- Show warnings first, then the selected screen ----
ui.shell(palette, reduce_motion)
if bundle_error:
    st.error(bundle_error)
    st.code(f"{spec.prepare_command}\n{spec.train_command}")
    if release_error:
        st.code(f"python scripts/verify_release.py --dataset {dataset_id} --write-manifests")
if metrics_error:
    st.warning(f"Benchmark metrics unavailable: {metrics_error}")
if source_mode == "Upload hourly CSV" and hourly is not None:
    st.warning(f"Uploaded data uses the pretrained {spec.short_name} model. "
               "Its benchmark scores describe the original household only; accuracy on this upload is unverified.")

# Each menu item has its own screen function in src/views.py.
renderers = {"Overview": views.render_overview, "Forecast Studio": views.render_forecast,
             "Data Hub": views.render_data_hub, "Model Lab": views.render_model_lab, "Help": views.render_help}
screen_key = page.lower().replace(" ", "_")
with st.container(key=f"screen_{screen_key}_{st.session_state.get('motion_replay', 0)}"):
    renderers[page](workspace)
ui.footer()
