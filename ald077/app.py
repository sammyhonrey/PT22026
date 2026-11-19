import calendar
import html
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Weather Prediction System",
    page_icon=":material/cloud:",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ML" / "models"
MODEL_PATH = MODEL_DIR / "weather_prediction_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

PAGE_OPTIONS = (
    "Dashboard",
    "Weather Prediction",
    "Model Information",
    "About",
)

DEFAULT_CATEGORICAL_VALUES = {
    "Location": "Sydney",
    "WindGustDir": "N",
    "WindDir9am": "N",
    "WindDir3pm": "N",
    "RainToday": "No",
    "Month": "1",
}

NUMERIC_RANGES = {
    "MinTemp": (-10.0, 35.0),
    "MaxTemp": (0.0, 50.0),
    "Rainfall": (0.0, 100.0),
    "Evaporation": (0.0, 25.0),
    "Sunshine": (0.0, 14.0),
    "WindGustSpeed": (0.0, 150.0),
    "WindSpeed9am": (0.0, 100.0),
    "WindSpeed3pm": (0.0, 100.0),
    "Humidity9am": (0.0, 100.0),
    "Humidity3pm": (0.0, 100.0),
    "Pressure9am": (970.0, 1050.0),
    "Pressure3pm": (970.0, 1050.0),
    "Cloud9am": (0.0, 8.0),
    "Cloud3pm": (0.0, 8.0),
    "Temp9am": (-5.0, 40.0),
    "Temp3pm": (0.0, 50.0),
}
INTEGER_NUMERIC_FEATURES = {"Cloud9am", "Cloud3pm"}

FEATURE_UNITS = {
    "MinTemp": "°C",
    "MaxTemp": "°C",
    "Rainfall": "mm",
    "Evaporation": "mm",
    "Sunshine": "hours",
    "WindGustSpeed": "km/h",
    "WindSpeed9am": "km/h",
    "WindSpeed3pm": "km/h",
    "Humidity9am": "%",
    "Humidity3pm": "%",
    "Pressure9am": "hPa",
    "Pressure3pm": "hPa",
    "Cloud9am": "oktas",
    "Cloud3pm": "oktas",
    "Temp9am": "°C",
    "Temp3pm": "°C",
}

FEATURE_HELP = {
    "MinTemp": "Minimum air temperature recorded for the day.",
    "MaxTemp": "Maximum air temperature recorded for the day.",
    "Temp9am": "Air temperature recorded at 9 am.",
    "Temp3pm": "Air temperature recorded at 3 pm.",
    "Pressure9am": "Atmospheric pressure recorded at 9 am.",
    "Pressure3pm": "Atmospheric pressure recorded at 3 pm.",
    "Rainfall": "Total rainfall recorded for the day.",
    "Evaporation": "Daily evaporation measured in millimetres.",
    "Sunshine": "Total hours of bright sunshine.",
    "Humidity9am": "Relative humidity recorded at 9 am.",
    "Humidity3pm": "Relative humidity recorded at 3 pm.",
    "Cloud9am": "Cloud cover at 9 am, measured in oktas from 0 to 8.",
    "Cloud3pm": "Cloud cover at 3 pm, measured in oktas from 0 to 8.",
    "WindGustSpeed": "Speed of the strongest wind gust recorded for the day.",
    "WindSpeed9am": "Wind speed recorded at 9 am.",
    "WindSpeed3pm": "Wind speed recorded at 3 pm.",
    "Location": "Weather station location.",
    "WindGustDir": "Direction the strongest wind gust came from.",
    "WindDir9am": "Wind direction recorded at 9 am.",
    "WindDir3pm": "Wind direction recorded at 3 pm.",
    "RainToday": "Whether rainfall reached the training dataset's daily threshold.",
    "Month": "Month number for the observation.",
}

FEATURE_GROUPS = (
    (
        "Temperature and atmospheric conditions",
        "Temperature and pressure readings for today's observation.",
        ("MinTemp", "MaxTemp", "Temp9am", "Temp3pm", "Pressure9am", "Pressure3pm"),
    ),
    (
        "Rainfall and sunshine",
        "Precipitation, evaporation, and daily sunshine.",
        ("Rainfall", "Evaporation", "Sunshine"),
    ),
    (
        "Humidity and cloud coverage",
        "Moisture and cloud observations from the morning and afternoon.",
        ("Humidity9am", "Humidity3pm", "Cloud9am", "Cloud3pm"),
    ),
    (
        "Wind information",
        "Wind speed and direction readings.",
        (
            "WindGustSpeed",
            "WindSpeed9am",
            "WindSpeed3pm",
            "WindGustDir",
            "WindDir9am",
            "WindDir3pm",
        ),
    ),
    (
        "Location and weather conditions",
        "Select the station and today's rainfall context.",
        ("Location", "RainToday", "Month"),
    ),
)


@st.cache_resource
def load_model_and_metadata():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")

    required = ("numeric_features", "categorical_features", "feature_columns")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(
            "The metadata file is missing these keys: " + ", ".join(missing)
        )

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]
    feature_columns = metadata["feature_columns"]
    if not all(
        isinstance(features, list)
        and all(isinstance(feature, str) for feature in features)
        for features in (numeric_features, categorical_features, feature_columns)
    ):
        raise ValueError("Feature definitions in model metadata must be lists of strings.")

    expected_features = numeric_features + categorical_features
    if (
        len(feature_columns) != len(set(feature_columns))
        or len(expected_features) != len(set(expected_features))
        or set(feature_columns) != set(expected_features)
    ):
        raise ValueError(
            "Metadata feature_columns must contain each numeric and categorical "
            "feature exactly once."
        )

    # Joblib artifacts use pickle serialization; only load trusted model files.
    model = joblib.load(MODEL_PATH)
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("The saved model must support predict_proba().")

    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("The saved model does not expose its trained classes.")

    positive_class = str(metadata.get("positive_class", "Yes"))
    if positive_class not in [str(value) for value in classes]:
        raise ValueError(
            f"Positive class {positive_class!r} is not present in the model classes."
        )

    threshold = float(metadata.get("threshold", 0.5))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The prediction threshold must be between 0 and 1.")

    return model, metadata, threshold


def extract_categorical_options(model, categorical_features):
    try:
        preprocessor = model.named_steps["preprocessor"]
        categorical_pipeline = preprocessor.named_transformers_["categorical"]
        encoder = categorical_pipeline.named_steps["onehot"]
        categories = encoder.categories_
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            "The saved model does not expose its trained categorical options."
        ) from exc

    if len(categories) != len(categorical_features):
        raise ValueError(
            "The model's categorical options do not match model_metadata.json."
        )

    options = {}
    for feature, values in zip(categorical_features, categories):
        options[feature] = [
            str(value.item() if isinstance(value, np.generic) else value)
            for value in values
        ]
        if not options[feature]:
            raise ValueError(f"The model has no trained values for {feature}.")
    return options


def randomize_weather_values(numeric_features, categorical_options):
    rng = np.random.default_rng()

    for feature in numeric_features:
        if feature not in NUMERIC_RANGES:
            raise ValueError(f"No random value range is configured for {feature}.")
        lower, upper = NUMERIC_RANGES[feature]
        if feature in INTEGER_NUMERIC_FEATURES:
            value = float(rng.integers(int(lower), int(upper) + 1))
        else:
            value = round(float(rng.uniform(lower, upper)), 2)
        st.session_state[f"weather_{feature}"] = value

    for feature, options in categorical_options.items():
        if not options:
            raise ValueError(f"No random values are available for {feature}.")
        st.session_state[f"weather_{feature}"] = options[
            int(rng.integers(0, len(options)))
        ]


def navigate_to(page):
    st.session_state["active_page"] = page


def render_hero(eyebrow, title, description, illustration):
    st.markdown(
        f"""
        <section class="weather-hero">
            <div class="hero-copy">
                <div class="hero-eyebrow">{html.escape(eyebrow)}</div>
                <div class="hero-heading">{html.escape(title)}</div>
                <div class="hero-description">{html.escape(description)}</div>
            </div>
            <div class="hero-illustration" aria-hidden="true">
                {html.escape(illustration)}
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_saved_metrics(metrics):
    if not isinstance(metrics, dict):
        st.caption("No validation metrics are included in the saved metadata.")
        return

    saved_values = []
    for label, raw_value in metrics.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        display_value = f"{value:.1%}" if 0 <= value <= 1 else f"{value:,.2f}"
        saved_values.append((str(label), display_value))

    if not saved_values:
        st.caption("No validation metrics are included in the saved metadata.")
        return

    with st.container(horizontal=True, gap="small"):
        for label, value in saved_values:
            st.metric(label, value, border=True)


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Weather Prediction System
            <span>Powered by Python, Streamlit &amp; Machine Learning</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(metadata, feature_columns, threshold):
    render_hero(
        "INTELLIGENT WEATHER INSIGHTS",
        "Understand tomorrow's weather",
        "Explore a data-driven rain outlook using today's temperature, wind, "
        "humidity, and sky conditions.",
        "🌤️",
    )

    st.space("medium")
    with st.grid("auto", min_column_width=175, gap="small"):
        st.metric(
            "Model",
            str(metadata.get("model_name", "Random Forest")),
            border=True,
        )
        st.metric(
            "Prediction target",
            str(metadata.get("target_column", "RainTomorrow")),
            border=True,
        )
        st.metric("Weather inputs", str(len(feature_columns)), border=True)
        st.metric("Decision threshold", f"{threshold:.0%}", border=True)

    st.subheader("Model validation snapshot", icon=":material/query_stats:")
    render_saved_metrics(metadata.get("metrics", {}))
    st.caption("Validation metrics are shown as saved in the model metadata.")

    st.space("small")
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.subheader("Make a prediction", icon=":material/near_me:")
            st.write(
                "Enter today's observations across five focused sections. "
                "The saved model pipeline prepares the inputs and estimates "
                "tomorrow's rain probability."
            )
            st.button(
                "Open weather prediction",
                key="dashboard_predict",
                type="primary",
                icon=":material/arrow_forward:",
                width="stretch",
                on_click=navigate_to,
                args=("Weather Prediction",),
            )
    with right:
        with st.container(border=True):
            st.subheader("Built for clarity", icon=":material/visibility:")
            st.write(
                "The classification, probability, and decision threshold come "
                "directly from the saved model and its metadata."
            )
            st.button(
                "Explore model information",
                key="dashboard_model_info",
                icon=":material/analytics:",
                width="stretch",
                on_click=navigate_to,
                args=("Model Information",),
            )


def render_feature_input(feature, numeric_features, categorical_options):
    unit = FEATURE_UNITS.get(feature)
    label = f"{feature} ({unit})" if unit else feature
    help_text = FEATURE_HELP.get(feature, "Enter the value recorded for this observation.")
    key = f"weather_{feature}"

    if feature in numeric_features:
        return st.number_input(
            label,
            format="%.2f",
            key=key,
            help=help_text,
        )

    if feature == "Month":
        return st.selectbox(
            label,
            options=categorical_options[feature],
            format_func=lambda value: calendar.month_name[int(value)],
            key=key,
            help=help_text,
        )

    return st.selectbox(
        label,
        options=categorical_options[feature],
        key=key,
        help=help_text,
    )


def render_feature_group(title, description, features, numeric_features, categorical_options):
    with st.container(border=True):
        st.subheader(title)
        st.caption(description)
        columns = st.columns(min(3, len(features)), vertical_alignment="top")
        values = {}
        for index, feature in enumerate(features):
            with columns[index % len(columns)]:
                values[feature] = render_feature_input(
                    feature,
                    numeric_features,
                    categorical_options,
                )
        return values


def render_prediction_result(
    prediction,
    positive_class,
    rain_probability,
    threshold,
    user_values,
):
    is_rain = prediction == positive_class
    headline = "Rain Expected Tomorrow" if is_rain else "No Rain Expected Tomorrow"
    icon = ":material/rainy:" if is_rain else ":material/wb_sunny:"

    with st.container(border=True):
        st.caption("MODEL OUTPUT")
        st.header(headline, icon=icon)
        st.write(
            "The model estimates rain for tomorrow."
            if is_rain
            else "The model estimates no rain for tomorrow."
        )

        probability_column, threshold_column = st.columns(2)
        with probability_column:
            st.metric(
                "Estimated probability of rain",
                f"{rain_probability * 100:.2f}%",
                border=True,
            )
        with threshold_column:
            st.metric(
                "Decision threshold",
                f"{threshold * 100:.2f}%",
                border=True,
            )
        st.progress(float(np.clip(rain_probability, 0.0, 1.0)))
        st.caption(
            "The status above reflects the saved model's probability and decision threshold."
        )

    with st.expander("View submitted weather measurements", icon=":material/table_view:"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": list(user_values),
                    "Value": [str(value) for value in user_values.values()],
                }
            ),
            hide_index=True,
            alt="Submitted weather measurements",
        )


def render_weather_prediction(
    metadata,
    model,
    numeric_features,
    categorical_features,
    feature_columns,
    categorical_options,
    positive_class,
    negative_class,
    model_classes,
    threshold,
):
    render_hero(
        "WEATHER PREDICTION",
        "A clearer view of tomorrow",
        "Add today's weather observations. The trained Random Forest will estimate "
        "the probability of rain for the next day.",
        "🌦️",
    )

    st.space("medium")
    for feature in numeric_features:
        st.session_state.setdefault(f"weather_{feature}", 0.0)
    for feature in categorical_features:
        options = categorical_options[feature]
        default = DEFAULT_CATEGORICAL_VALUES.get(feature, options[0])
        if default not in options:
            default = options[0]
        st.session_state.setdefault(f"weather_{feature}", default)

    random_column, hint_column = st.columns([1, 2], vertical_alignment="center")
    with random_column:
        st.button(
            "Fill with random values",
            key="randomize_weather",
            icon=":material/casino:",
            width="stretch",
            on_click=randomize_weather_values,
            args=(numeric_features, categorical_options),
        )
    with hint_column:
        st.caption(
            "Use a randomized demo profile, then adjust the values to match "
            "actual observations."
        )

    user_values = {}
    submitted = False
    with st.form("weather_form", border=False):
        for title, description, features in FEATURE_GROUPS:
            user_values.update(
                render_feature_group(
                    title,
                    description,
                    features,
                    numeric_features,
                    categorical_options,
                )
            )
        submitted = st.form_submit_button(
            "Generate weather prediction",
            key="predict_weather",
            type="primary",
            icon=":material/bolt:",
            width="stretch",
        )

    if submitted:
        try:
            input_frame = pd.DataFrame(
                [{feature: user_values[feature] for feature in feature_columns}],
                columns=feature_columns,
            )
            with st.spinner("Analyzing the weather observations..."):
                probabilities = np.asarray(model.predict_proba(input_frame))
            if (
                probabilities.ndim != 2
                or probabilities.shape[0] != 1
                or probabilities.shape[1] != len(model_classes)
            ):
                raise ValueError("The model returned probabilities in an unexpected shape.")

            positive_index = model_classes.index(positive_class)
            rain_probability = float(probabilities[0, positive_index])
            if not np.isfinite(rain_probability) or not 0 <= rain_probability <= 1:
                raise ValueError("The model returned an invalid rain probability.")

            prediction = (
                positive_class
                if rain_probability >= threshold
                else negative_class
            )
            st.space("medium")
            render_prediction_result(
                prediction,
                positive_class,
                rain_probability,
                threshold,
                user_values,
            )

        except Exception as exc:
            st.error("Prediction failed.", icon=":material/error:")
            st.exception(exc)


def render_model_information(metadata, feature_columns, threshold):
    st.header("Model Information", icon=":material/analytics:")
    st.write(
        "A transparent summary of the trained model and the validation values "
        "included with its metadata."
    )

    with st.grid("auto", min_column_width=190, gap="small"):
        st.metric("Model", str(metadata.get("model_name", "Random Forest")), border=True)
        st.metric("Prediction target", str(metadata.get("target_column", "RainTomorrow")), border=True)
        st.metric("Input features", str(len(feature_columns)), border=True)
        st.metric("Decision threshold", f"{threshold:.0%}", border=True)

    st.subheader("Saved validation metrics", icon=":material/query_stats:")
    render_saved_metrics(metadata.get("metrics", {}))
    st.caption("These values are read from model_metadata.json; the app does not calculate new metrics.")

    training_start = metadata.get("training_start_date")
    training_end = metadata.get("training_end_date")
    testing_start = metadata.get("testing_start_date")
    testing_end = metadata.get("testing_end_date")
    if any((training_start, training_end, testing_start, testing_end)):
        st.subheader("Evaluation periods", icon=":material/calendar_month:")
        training_column, testing_column = st.columns(2)
        with training_column:
            with st.container(border=True):
                st.markdown("**Training period**")
                st.write(f"{training_start or 'Not specified'} — {training_end or 'Not specified'}")
        with testing_column:
            with st.container(border=True):
                st.markdown("**Testing period**")
                st.write(f"{testing_start or 'Not specified'} — {testing_end or 'Not specified'}")


def render_about():
    st.header("About", icon=":material/info:")
    st.write(
        "Weather Prediction System uses a saved machine-learning pipeline to "
        "estimate whether rain is expected tomorrow from today's weather observations."
    )

    with st.container(border=True):
        st.subheader("How it works", icon=":material/route:")
        step_columns = st.columns(3)
        with step_columns[0]:
            st.markdown("**01 · Add observations**")
            st.caption("Enter the 22 weather, location, and condition inputs.")
        with step_columns[1]:
            st.markdown("**02 · Prepare the inputs**")
            st.caption("The saved preprocessing pipeline handles numeric and categorical fields.")
        with step_columns[2]:
            st.markdown("**03 · Review the estimate**")
            st.caption("The model probability is compared with its saved decision threshold.")

    st.warning(
        "This is an educational prediction tool, not an official meteorological forecast.",
        icon=":material/warning:",
    )
    st.caption(
        "Only load trusted joblib model files. Joblib uses pickle serialization, "
        "which can execute code when loading an artifact."
    )


def main():
    try:
        model, metadata, threshold = load_model_and_metadata()
        numeric_features = metadata["numeric_features"]
        categorical_features = metadata["categorical_features"]
        feature_columns = metadata["feature_columns"]
        categorical_options = extract_categorical_options(model, categorical_features)
    except Exception as exc:
        st.error("The model could not be loaded.", icon=":material/error:")
        st.exception(exc)
        st.info(
            "Check that ML/models contains weather_prediction_model.joblib and "
            "model_metadata.json, and that the joblib file is a trusted Random Forest "
            "pipeline matching this metadata."
        )
        st.stop()
        raise

    grouped_features = [
        feature
        for _, _, group_features in FEATURE_GROUPS
        for feature in group_features
    ]
    if (
        len(grouped_features) != len(set(grouped_features))
        or set(grouped_features) != set(feature_columns)
    ):
        st.error("The weather input groups do not match the model's feature list.")
        st.stop()
        raise ValueError("The weather input groups do not match model metadata.")

    positive_class = str(metadata.get("positive_class", "Yes"))
    model_classes = [str(value) for value in model.classes_]
    negative_class = next(
        (value for value in model_classes if value != positive_class),
        "No",
    )

    st.markdown(
        """
        <style>
        .weather-hero {
            position: relative;
            isolation: isolate;
            display: flex;
            min-height: 250px;
            align-items: center;
            justify-content: space-between;
            overflow: hidden;
            padding: clamp(1.7rem, 4vw, 3.2rem);
            border-radius: 28px;
            background: linear-gradient(125deg, #0b1e33 0%, #0e7490 58%, #38bdf8 100%);
            color: #f7fbff;
            box-shadow: 0 20px 50px rgba(11, 30, 51, 0.16);
        }
        .weather-hero::before,
        .weather-hero::after {
            position: absolute;
            z-index: -1;
            border-radius: 999px;
            content: "";
            pointer-events: none;
        }
        .weather-hero::before {
            top: -135px;
            right: -35px;
            width: 310px;
            height: 310px;
            background: radial-gradient(circle, rgba(125, 211, 252, 0.42), rgba(125, 211, 252, 0));
        }
        .weather-hero::after {
            right: 180px;
            bottom: -180px;
            width: 330px;
            height: 330px;
            background: radial-gradient(circle, rgba(34, 184, 207, 0.28), rgba(34, 184, 207, 0));
        }
        .hero-copy {
            max-width: 700px;
        }
        .hero-eyebrow {
            margin-bottom: 0.85rem;
            color: #a5f3fc;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.16em;
        }
        .hero-heading {
            max-width: 680px;
            font-size: clamp(2.1rem, 5vw, 3.65rem);
            font-weight: 750;
            letter-spacing: -0.045em;
            line-height: 1.06;
        }
        .hero-description {
            max-width: 620px;
            margin-top: 1rem;
            color: #e0f2fe;
            font-size: clamp(1rem, 1.6vw, 1.12rem);
            line-height: 1.65;
        }
        .hero-illustration {
            padding: 0 1.4rem;
            font-size: clamp(4.5rem, 9vw, 7rem);
            filter: drop-shadow(0 12px 20px rgba(4, 25, 45, 0.24));
        }
        .app-footer {
            margin-top: 3rem;
            padding: 1.25rem 0 0.5rem;
            border-top: 1px solid #d9e5ef;
            color: #58718a;
            font-size: 0.85rem;
            text-align: center;
        }
        .app-footer span {
            display: block;
            margin-top: 0.25rem;
            color: #7890a6;
            font-size: 0.78rem;
        }
        .st-key-predict_weather button {
            min-height: 3.2rem;
            border: 0;
            border-radius: 14px;
            background: linear-gradient(120deg, #0e7490, #0284c7);
            color: #ffffff;
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(14, 116, 144, 0.2);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .st-key-predict_weather button:hover {
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(14, 116, 144, 0.28);
            border: 0;
        }
        .st-key-randomize_weather button {
            min-height: 2.8rem;
            border-radius: 12px;
            border-color: #b8d7e6;
            background: #eaf5fb;
            color: #0c526f;
            font-weight: 650;
        }
        @media (max-width: 640px) {
            .weather-hero {
                min-height: 220px;
                border-radius: 22px;
            }
            .hero-illustration {
                display: none;
            }
        }
        @media (prefers-reduced-motion: reduce) {
            .st-key-predict_weather button {
                transition: none;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    header_column, status_column = st.columns([5, 1], vertical_alignment="center")
    with header_column:
        st.title("Weather Prediction System", icon=":material/cloud:")
        st.caption("Intelligent Weather Prediction Powered by Machine Learning")
    with status_column:
        st.badge("Model ready", icon=":material/check_circle:", color="green")

    st.session_state.setdefault("active_page", "Dashboard")
    if st.session_state["active_page"] not in PAGE_OPTIONS:
        st.session_state["active_page"] = "Dashboard"

    with st.sidebar:
        st.markdown("### :material/cloud: Weather intelligence")
        st.caption("MACHINE-LEARNING FORECASTING")
        st.radio(
            "Navigate",
            options=PAGE_OPTIONS,
            key="active_page",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Model at a glance**")
        st.caption(str(metadata.get("model_name", "Random Forest")))
        st.caption(f"{len(feature_columns)} weather inputs · RainTomorrow")
        st.caption("Only use predictions as an estimate.")

    current_page = st.session_state["active_page"]
    if current_page == "Dashboard":
        render_dashboard(metadata, feature_columns, threshold)
    elif current_page == "Weather Prediction":
        render_weather_prediction(
            metadata,
            model,
            numeric_features,
            categorical_features,
            feature_columns,
            categorical_options,
            positive_class,
            negative_class,
            model_classes,
            threshold,
        )
    elif current_page == "Model Information":
        render_model_information(metadata, feature_columns, threshold)
    else:
        render_about()

    render_footer()


main()
