
import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from pathlib import Path
from math import isfinite

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "house_price_model.joblib")
pipeline = joblib.load(BASE_DIR / "models" / "preprocessing_pipeline.joblib")

OCEAN_CHOICES = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

FEATURE_COLUMNS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "ocean_proximity"
]

MAE = 31678.01
RMSE = 47859.54
R2 = 0.8243

CA_Q25 = 119600
CA_MEDIAN = 179700
CA_MEAN = 206856
CA_Q75 = 264725

OCEAN_AVERAGES = {
    "INLAND": 124805,
    "<1H OCEAN": 240084,
    "NEAR OCEAN": 249434,
    "NEAR BAY": 259212,
    "ISLAND": 380440
}

CITIES = {
    "San Francisco": (-122.42, 37.77),
    "San Jose": (-121.89, 37.34),
    "Sacramento": (-121.49, 38.58),
    "Fresno": (-119.77, 36.74),
    "Los Angeles": (-118.24, 34.05),
    "San Diego": (-117.16, 32.72)
}

FOREST = "#142D2B"
FOREST_2 = "#24443F"
GOLD = "#B68C59"
IVORY = "#F5F3EE"
WHITE = "#FFFFFF"
SAGE = "#DCE5E1"
MUTED = "#536765"
GRID = "#E4E7E2"


def money(value):
    return f"${value:,.0f}"


def style_figure(fig, title, height=360):
    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            font=dict(
                family="Playfair Display, Georgia, serif",
                size=18,
                color=FOREST
            )
        ),
        height=height,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(
            family="Inter, Arial, sans-serif",
            color=FOREST,
            size=13
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=FOREST,
            font=dict(color=WHITE)
        )
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED, size=12)
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=MUTED),
        title_font=dict(color=MUTED, size=12)
    )

    return fig


def build_row(**kwargs):
    return pd.DataFrame([{col: kwargs[col] for col in FEATURE_COLUMNS}])


def predict_frame(df):
    return model.predict(pipeline.transform(df))


def error_card(message):
    return f"""
    <div class="result-card result-error">
        <div class="result-label">Check your details</div>
        <div class="error-text">{message}</div>
    </div>
    """


def empty_card():
    return """
    <div class="result-card result-empty">
        <div class="result-label">Estimated value</div>
        <div class="empty-price">$ – – – , – – –</div>
        <div class="empty-text">
            Fill in the home details and press
            <b>Estimate house price</b> to see the estimated value,
            approximate price range and comparison with California.
        </div>
    </div>
    """


def price_tier(price):
    if price < CA_Q25:
        return "Budget-friendly"
    if price < CA_MEDIAN:
        return "Below the California median"
    if price < CA_Q75:
        return "Above the California median"
    return "Premium"


def result_card(price, rooms, bedrooms, population, households):
    low = max(price - MAE, 0)
    high = price + MAE
    diff = (price - CA_MEDIAN) / CA_MEDIAN * 100

    diff_text = f"{diff:+.0f}% vs median"
    diff_class = "chip-up" if diff >= 0 else "chip-down"

    return f"""
    <div class="result-card result-filled">
        <div class="result-label">Estimated value</div>
        <div class="result-price">{money(price)}</div>
        <div class="result-range">
            Approximate range {money(low)} – {money(high)}
        </div>
        <div class="chip-row">
            <span class="chip chip-tier">{price_tier(price)}</span>
            <span class="chip {diff_class}">{diff_text}</span>
        </div>
        <div class="stat-grid">
            <div class="stat">
                <div class="stat-value">{rooms / households:.1f}</div>
                <div class="stat-name">rooms per household</div>
            </div>
            <div class="stat">
                <div class="stat-value">{bedrooms / rooms * 100:.0f}%</div>
                <div class="stat-name">of rooms are bedrooms</div>
            </div>
            <div class="stat">
                <div class="stat-value">{population / households:.1f}</div>
                <div class="stat-name">people per household</div>
            </div>
        </div>
        <div class="result-note">
            The range is an approximate reference based on the model's
            mean absolute error (±{money(MAE)}), not a guaranteed
            confidence interval. DreamHome estimates the median house
            value of a California neighbourhood, not the exact market
            price of an individual property.
        </div>
    </div>
    """


def comparison_chart(price):
    labels = [
        "Your estimate",
        "75% of homes cost less than",
        "California average",
        "California median",
        "25% of homes cost less than"
    ]

    values = [price, CA_Q75, CA_MEAN, CA_MEDIAN, CA_Q25]
    colors = [GOLD, SAGE, SAGE, SAGE, SAGE]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[money(v) for v in values],
            textposition="outside",
            cliponaxis=False,
            error_x=dict(
                type="data",
                array=[MAE, 0, 0, 0, 0],
                color=FOREST_2,
                thickness=2,
                width=6
            ),
            hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"
        )
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="$", tickformat=",.0f")

    fig = style_figure(fig, "How your estimate compares", 340)
    fig.update_layout(margin=dict(l=20, r=90, t=60, b=40))

    return fig


def income_sensitivity_chart(row, price):
    incomes = np.linspace(0.5, 12, 24)

    df = pd.concat(
        [pd.DataFrame([row])] * len(incomes),
        ignore_index=True
    )

    df["median_income"] = incomes
    prices = predict_frame(df)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=incomes,
            y=prices,
            mode="lines",
            line=dict(color=FOREST_2, width=3, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(36, 68, 63, 0.10)",
            hovertemplate="Income %{x:.1f}: %{y:$,.0f}<extra></extra>"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[row["median_income"]],
            y=[price],
            mode="markers+text",
            marker=dict(
                color=GOLD,
                size=14,
                line=dict(color=WHITE, width=2)
            ),
            text=["Your home"],
            textposition="top left",
            textfont=dict(color=FOREST_2, size=12),
            hovertemplate="Your home: %{y:$,.0f}<extra></extra>"
        )
    )

    fig.update_xaxes(title="Median income (×$10,000)")
    fig.update_yaxes(
        title="Estimated price",
        tickprefix="$",
        tickformat=",.0f"
    )

    return style_figure(
        fig,
        "What if the neighbourhood income changed?"
    )


def location_chart(longitude, latitude, price):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[c[0] for c in CITIES.values()],
            y=[c[1] for c in CITIES.values()],
            mode="markers+text",
            text=list(CITIES.keys()),
            textposition="middle right",
            marker=dict(
                color=SAGE,
                size=10,
                line=dict(color=FOREST_2, width=1)
            ),
            textfont=dict(color=MUTED, size=12),
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[longitude],
            y=[latitude],
            mode="markers",
            marker=dict(
                color=GOLD,
                size=20,
                symbol="star",
                line=dict(color=WHITE, width=2)
            ),
            hovertemplate=(
                f"Your home<br>{money(price)}<br>"
                "Lon %{x:.2f}, Lat %{y:.2f}<extra></extra>"
            )
        )
    )

    fig.update_xaxes(
        range=[-125.5, -113.5],
        title="Longitude"
    )

    fig.update_yaxes(
        range=[32, 42.5],
        title="Latitude",
        scaleanchor="x",
        scaleratio=1.25
    )

    return style_figure(
        fig,
        "Where your home sits in California",
        440
    )


def feature_importance_chart():
    try:
        raw_names = pipeline.get_feature_names_out()
        importances = model.feature_importances_

        if len(raw_names) != len(importances):
            return None

    except Exception:
        return None

    grouped = {}

    for name, value in zip(raw_names, importances):
        clean = str(name).split("__")[-1]

        if clean.startswith("ocean_proximity"):
            clean = "ocean_proximity"

        clean = clean.replace("_", " ").capitalize()
        grouped[clean] = grouped.get(clean, 0) + float(value)

    ordered = sorted(grouped.items(), key=lambda item: item[1])

    names = [n for n, _ in ordered]
    vals = [v * 100 for _, v in ordered]

    colors = [FOREST_2] * (len(vals) - 1) + [GOLD]

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
        )
    )

    fig.update_xaxes(
        title="Share of the model's decisions (%)"
    )

    fig = style_figure(
        fig,
        "What drives the price the most?",
        400
    )

    fig.update_layout(
        margin=dict(l=20, r=60, t=60, b=40)
    )

    return fig


def ocean_average_chart():
    names = list(OCEAN_AVERAGES.keys())
    vals = list(OCEAN_AVERAGES.values())

    fig = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker=dict(
                color=[SAGE, FOREST_2, FOREST_2, FOREST_2, GOLD]
            ),
            text=[money(v) for v in vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:$,.0f}<extra></extra>"
        )
    )

    fig.update_yaxes(
        tickprefix="$",
        tickformat=",.0f"
    )

    fig = style_figure(
        fig,
        "Average value by ocean proximity (dataset)",
        400
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig


def metrics_chart():
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=R2 * 100,
            number=dict(
                suffix="%",
                font=dict(color=FOREST)
            ),
            title=dict(
                text="Variance explained (R²)",
                font=dict(size=15, color=MUTED)
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickcolor=MUTED
                ),
                bar=dict(color=FOREST_2),
                bgcolor=SAGE,
                borderwidth=0,
                steps=[dict(range=[0, 100], color=SAGE)]
            )
        )
    )

    return style_figure(
        fig,
        "How much of the price variation the model explains",
        300
    )


def validate(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity
):
    values = [
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income
    ]

    if any(v is None for v in values):
        return "Please complete all the fields."

    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return "Please enter valid numbers."

    if not all(isfinite(v) for v in values):
        return "Please enter valid numbers."

    (
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income
    ) = values

    if not (-124.6 <= longitude <= -114.0 and 32.4 <= latitude <= 42.1):
        return "This location is outside California. Please enter valid California coordinates."

    if not (1 <= housing_median_age <= 52):
        return "Housing age must be between 1 and 52 years."

    if total_rooms <= 0:
        return "Total rooms must be greater than zero."

    if total_bedrooms <= 0:
        return "Total bedrooms must be greater than zero."

    if total_bedrooms > total_rooms:
        return "Bedrooms cannot exceed total rooms."

    if population <= 0:
        return "Population must be greater than zero."

    if households <= 0:
        return "Households must be greater than zero."

    if households > population:
        return "Households cannot exceed population."

    if median_income <= 0:
        return "Median income must be greater than zero."

    if ocean_proximity not in OCEAN_CHOICES:
        return "Please select a valid ocean proximity."

    return None


def make_prediction(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity
):
    problem = validate(
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        ocean_proximity
    )

    if problem:
        gr.Warning(problem)
        return error_card(problem), None, None, None

    row = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }

    try:
        price = float(predict_frame(build_row(**row))[0])

        return (
            result_card(
                price,
                total_rooms,
                total_bedrooms,
                population,
                households
            ),
            comparison_chart(price),
            income_sensitivity_chart(row, price),
            location_chart(longitude, latitude, price)
        )

    except Exception as error:
        message = f"Prediction failed: {str(error)}"
        gr.Warning(message)
        return error_card(message), None, None, None


n_trees = getattr(model, "n_estimators", "n/a")

HERO_HTML = f"""
<div class="dh-header">
    <div class="dh-brand">
        <div class="dh-logo">
            <svg viewBox="0 0 24 24" width="30" height="30"
                 fill="none" stroke="#B68C59" stroke-width="1.7"
                 stroke-linecap="round" stroke-linejoin="round">
                <path d="M3 11.5 12 4l9 7.5"/>
                <path d="M5.5 10v9.5h13V10"/>
                <path d="M10 19.5v-5h4v5"/>
            </svg>
        </div>
        <div class="dh-eyebrow">California real estate valuation</div>
    </div>
    <div class="dh-title">DreamHome</div>
    <div class="dh-sub">
        Discover the value of a California home with the power of machine learning.
    </div>
    <div class="dh-rule"></div>
    <div class="dh-badges">
        <span class="dh-badge"><b>Model</b> Random Forest</span>
        <span class="dh-badge"><b>Dataset</b> 20,640 records</span>
        <span class="dh-badge"><b>R² score</b> {R2:.2f}</span>
    </div>
</div>
"""

METRICS_HTML = f"""
<div class="metric-grid">
    <div class="metric">
        <div class="metric-value">{money(MAE)}</div>
        <div class="metric-name">Mean absolute error</div>
        <div class="metric-note">Typical miss per prediction</div>
    </div>
    <div class="metric">
        <div class="metric-value">{money(RMSE)}</div>
        <div class="metric-name">Root mean squared error</div>
        <div class="metric-note">Penalises large misses more</div>
    </div>
    <div class="metric">
        <div class="metric-value">{R2:.4f}</div>
        <div class="metric-name">R² score</div>
        <div class="metric-note">Variance explained, not accuracy</div>
    </div>
</div>
"""

INSIGHT_NOTES_HTML = """
<div class="info-card">
    <div class="info-title">How to read these metrics</div>
    <ul>
        <li><b>R²</b> is the share of variation in house values explained by the model. It is not prediction accuracy.</li>
        <li><b>MAE</b> is the typical size of the model's error on the test split.</li>
        <li><b>RMSE</b> gives more weight to large prediction errors.</li>
        <li>The model predicts California neighbourhood-level median house values, not exact individual property prices.</li>
    </ul>
</div>
"""

ABOUT_MD = f"""
### About DreamHome

DreamHome is a machine learning app that estimates the median house value of a California neighbourhood from its housing information.

**How it works**

1. Enter the location, property, population and income details.
2. The preprocessing pipeline prepares the data.
3. The Random Forest model predicts the median house value.
4. The app displays the estimate, an approximate range and charts.

**Model**

- Algorithm: Random Forest Regressor ({n_trees} trees)
- Task: Supervised regression
- Target: Median house value
- Dataset: California Housing, 20,640 records, 9 input features
- Libraries: Python, Pandas, NumPy, Scikit-learn, Joblib, Plotly and Gradio
- MAE: {money(MAE)}
- RMSE: {money(RMSE)}
- R²: {R2:.4f}

**Good to know**

- Estimates are not official market prices.
- The model is trained on California data only.
- The original dataset target values are capped near $500,000.
- The approximate range is based on MAE and is not a guaranteed confidence interval.
- R² measures variance explained, not prediction accuracy.
"""

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    color-scheme: light;
}

body, gradio-app {
    background: #F5F3EE !important;
}

.gradio-container {
    --body-background-fill: #F5F3EE;
    --background-fill-primary: #FFFFFF;
    --background-fill-secondary: #F5F3EE;
    --block-background-fill: #FFFFFF;
    --block-border-color: #E4E7E2;
    --border-color-primary: #E4E7E2;
    --input-background-fill: #FFFFFF;
    --input-border-color: #E4E7E2;
    --input-border-color-focus: #B68C59;
    --input-shadow-focus: 0 0 0 3px rgba(182,140,89,0.20);
    --block-label-text-color: #24443F;
    --block-title-text-color: #24443F;
    --block-info-text-color: #536765;
    --body-text-color: #142D2B;
    --body-text-color-subdued: #536765;
    --color-accent: #B68C59;
    --color-accent-soft: #DCE5E1;
    --button-primary-background-fill: #142D2B;
    --button-primary-background-fill-hover: #24443F;
    --button-primary-text-color: #FFFFFF;
    background: #F5F3EE !important;
    font-family: 'Inter', Arial, sans-serif !important;
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding-top: 16px !important;
}

.dh-header {
    background: radial-gradient(circle at 94% 0%, rgba(182,140,89,0.20), transparent 42%), #142D2B;
    border-radius: 24px;
    padding: 34px 42px 30px;
    margin: 4px 0 18px;
    border-bottom: 3px solid #B68C59;
    box-shadow: 0 12px 30px rgba(20,45,43,0.18);
}

.dh-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}

.dh-logo {
    width: 50px;
    height: 50px;
    border-radius: 14px;
    border: 1.5px solid #B68C59;
    background: rgba(182,140,89,0.10);
    display: flex;
    align-items: center;
    justify-content: center;
}

.dh-eyebrow {
    color: #B68C59 !important;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.dh-title {
    font-family: 'Playfair Display', Georgia, serif;
    color: #FFFFFF !important;
    font-size: 54px;
    font-weight: 700;
    line-height: 1.1;
    margin-top: 16px;
}

.dh-sub {
    color: #DCE5E1 !important;
    font-size: 17px;
    line-height: 1.55;
    max-width: 620px;
    margin-top: 10px;
}

.dh-rule {
    width: 64px;
    height: 2px;
    background: #B68C59;
    margin: 20px 0 18px;
}

.dh-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.dh-badge {
    border: 1px solid rgba(182,140,89,0.55);
    background: rgba(255,255,255,0.05);
    color: #F5F3EE !important;
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 13px;
}

.dh-badge b {
    color: #B68C59 !important;
    margin-right: 6px;
}

.tab-nav button, button[role="tab"] {
    font-weight: 600 !important;
    font-size: 15px !important;
    color: #536765 !important;
    padding: 12px 22px !important;
}

button[role="tab"][aria-selected="true"] {
    color: #142D2B !important;
    border-color: #B68C59 !important;
}

.form-section {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 6px 0 2px;
}

.form-step {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1.5px solid #B68C59;
    color: #B68C59;
    font-size: 13px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.form-heading {
    font-family: 'Playfair Display', Georgia, serif;
    color: #142D2B;
    font-size: 20px;
    font-weight: 700;
}

.form-heading-hint {
    color: #536765;
    font-size: 13px;
}

.form-section-next {
    border-top: 1px solid #E4E7E2;
    padding-top: 18px;
    margin-top: 10px;
}

.section-head {
    margin: 22px 2px 6px;
}

.section-title {
    font-family: 'Playfair Display', Georgia, serif;
    color: #142D2B;
    font-size: 24px;
    font-weight: 700;
}

.section-hint {
    color: #536765;
    font-size: 14px;
    margin-top: 2px;
}

.panel {
    background: #FFFFFF !important;
    border: 1px solid #E4E7E2 !important;
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: 0 10px 28px rgba(20,45,43,0.05) !important;
}

.chart-card {
    background: #FFFFFF !important;
    border: 1px solid #E4E7E2 !important;
    border-radius: 18px !important;
    overflow: hidden;
    box-shadow: 0 8px 22px rgba(20,45,43,0.04) !important;
}

.about-card {
    background: #FFFFFF !important;
    border: 1px solid #E4E7E2 !important;
    border-radius: 20px !important;
    padding: 28px 34px !important;
    box-shadow: 0 10px 28px rgba(20,45,43,0.05) !important;
}

.about-card h3 {
    font-family: 'Playfair Display', Georgia, serif;
    color: #142D2B;
    font-size: 28px;
}

.about-card strong {
    color: #24443F;
}

.about-card li {
    line-height: 1.7;
}

.gradio-container input[type="number"],
.gradio-container input[type="text"],
.gradio-container select,
.gradio-container textarea {
    border-radius: 10px !important;
    font-size: 15px !important;
}

.predict-button {
    background: #142D2B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-bottom: 4px solid #B68C59 !important;
    border-radius: 14px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    padding: 16px !important;
    margin-top: 10px;
    box-shadow: 0 10px 22px rgba(20,45,43,0.22) !important;
    transition: background 0.2s ease, transform 0.1s ease;
}

.predict-button:hover {
    background: #24443F !important;
    transform: translateY(-1px);
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #E4E7E2;
    border-radius: 22px;
    padding: 30px 32px;
    position: relative;
    overflow: hidden;
}

.result-label {
    color: #B68C59;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.result-filled {
    background: #142D2B;
    border: 1px solid #24443F;
    box-shadow: 0 16px 36px rgba(20,45,43,0.20);
}

.result-filled::before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    background: #B68C59;
}

.result-price {
    font-family: 'Playfair Display', Georgia, serif;
    color: #FFFFFF !important;
    font-size: 60px;
    font-weight: 700;
    line-height: 1.08;
    margin: 8px 0 6px;
}

.result-range {
    color: #E3C79C !important;
    font-size: 15px;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 18px 0 6px;
}

.chip {
    border-radius: 999px;
    padding: 5px 14px;
    font-size: 13px;
    font-weight: 600;
}

.chip-tier {
    background: rgba(182,140,89,0.18);
    color: #E3C79C !important;
    border: 1px solid rgba(182,140,89,0.6);
}

.chip-up {
    background: rgba(220,229,225,0.16);
    color: #DCE5E1 !important;
    border: 1px solid rgba(220,229,225,0.35);
}

.chip-down {
    background: rgba(220,229,225,0.10);
    color: #E3C79C !important;
    border: 1px solid rgba(182,140,89,0.45);
}

.stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 20px;
}

.stat {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(220,229,225,0.14);
    border-radius: 14px;
    padding: 14px;
}

.stat-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 26px;
    color: #FFFFFF !important;
    font-weight: 700;
}

.stat-name {
    color: #A9BDB8 !important;
    font-size: 12px;
    margin-top: 2px;
}

.result-note {
    margin-top: 20px;
    padding-top: 14px;
    border-top: 1px solid rgba(220,229,225,0.18);
    color: #A9BDB8 !important;
    font-size: 12px;
    line-height: 1.6;
}

.result-empty {
    border: 1.5px dashed #C5D0CB;
    background: #FFFFFF;
}

.empty-price {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 44px;
    color: #DCE5E1;
    margin: 12px 0;
}

.empty-text {
    color: #536765;
    font-size: 14px;
    line-height: 1.65;
}

.result-error {
    background: #FBF4F2;
    border: 1px solid #E8CFC9;
}

.result-error .result-label {
    color: #9B4A3C;
}

.error-text {
    color: #6F3329;
    font-size: 15px;
    margin-top: 8px;
    line-height: 1.55;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 10px;
}

.metric {
    background: #FFFFFF;
    border: 1px solid #E4E7E2;
    border-radius: 20px;
    padding: 24px;
    position: relative;
    box-shadow: 0 8px 22px rgba(20,45,43,0.04);
}

.metric::before {
    content: "";
    position: absolute;
    left: 24px;
    top: 0;
    width: 34px;
    height: 3px;
    background: #B68C59;
}

.metric-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 34px;
    color: #142D2B;
    font-weight: 700;
}

.metric-name {
    color: #24443F;
    font-weight: 600;
    font-size: 14px;
    margin-top: 4px;
}

.metric-note {
    color: #536765;
    font-size: 12.5px;
    margin-top: 2px;
}

.info-card {
    background: #FFFFFF;
    border: 1px solid #E4E7E2;
    border-left: 4px solid #B68C59;
    border-radius: 18px;
    padding: 22px 26px;
    height: 100%;
    box-shadow: 0 8px 22px rgba(20,45,43,0.04);
}

.info-title {
    font-family: 'Playfair Display', Georgia, serif;
    color: #142D2B;
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.info-card ul {
    padding-left: 18px;
    margin: 0;
}

.info-card li {
    color: #536765;
    font-size: 14px;
    line-height: 1.65;
    margin-bottom: 8px;
}

.footer-text {
    text-align: center;
    color: #536765;
    font-size: 13px;
    margin: 28px 0 8px;
}

@media (max-width: 720px) {
    .dh-header {
        padding: 26px 22px;
    }

    .dh-title {
        font-size: 38px;
    }

    .result-price {
        font-size: 42px;
    }

    .stat-grid,
    .metric-grid {
        grid-template-columns: 1fr;
    }
}
"""

with gr.Blocks(title="DreamHome | House Price Estimator") as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

        with gr.Tab("Estimate"):

            with gr.Row(equal_height=False):

                with gr.Column(scale=6, elem_classes=["panel"]):

                    gr.HTML(
                        '<div class="form-section">'
                        '<span class="form-step">1</span>'
                        '<div><div class="form-heading">Location</div>'
                        '<div class="form-heading-hint">Where is the neighbourhood?</div>'
                        '</div></div>'
                    )

                    with gr.Row():
                        longitude = gr.Number(
                            label="Longitude",
                            value=-122.23,
                            info="West is negative, e.g. -122.23"
                        )

                        latitude = gr.Number(
                            label="Latitude",
                            value=37.88,
                            info="North is positive, e.g. 37.88"
                        )

                    ocean_proximity = gr.Dropdown(
                        choices=OCEAN_CHOICES,
                        value="NEAR BAY",
                        label="Ocean Proximity"
                    )

                    gr.HTML(
                        '<div class="form-section form-section-next">'
                        '<span class="form-step">2</span>'
                        '<div><div class="form-heading">Property</div>'
                        '<div class="form-heading-hint">Totals for the whole block of homes.</div>'
                        '</div></div>'
                    )

                    with gr.Row():
                        housing_median_age = gr.Number(
                            label="Housing Median Age (years)",
                            value=30
                        )

                        total_rooms = gr.Number(
                            label="Total Rooms",
                            value=2000
                        )

                        total_bedrooms = gr.Number(
                            label="Total Bedrooms",
                            value=400
                        )

                    gr.HTML(
                        '<div class="form-section form-section-next">'
                        '<span class="form-step">3</span>'
                        '<div><div class="form-heading">People and Income</div>'
                        '<div class="form-heading-hint">Who lives in the neighbourhood?</div>'
                        '</div></div>'
                    )

                    with gr.Row():
                        population = gr.Number(
                            label="Population",
                            value=1000
                        )

                        households = gr.Number(
                            label="Households",
                            value=350
                        )

                        median_income = gr.Number(
                            label="Median Income (×$10,000)",
                            value=5.0,
                            info="5.0 means $50,000"
                        )

                    predict_button = gr.Button(
                        "Estimate house price",
                        elem_classes=["predict-button"]
                    )

                with gr.Column(scale=5):
                    result_html = gr.HTML(empty_card())

            gr.HTML(
                '<div class="section-head">'
                '<div class="section-title">Price analysis</div>'
                '<div class="section-hint">The comparison, income and location charts update with every estimate.</div>'
                '</div>'
            )

            with gr.Row():
                comparison_plot = gr.Plot(
                    label="Comparison",
                    show_label=False,
                    elem_classes=["chart-card"]
                )

                income_plot = gr.Plot(
                    show_label=False,
                    elem_classes=["chart-card"]
                )

            with gr.Row():
                location_plot = gr.Plot(
                    show_label=False,
                    elem_classes=["chart-card"]
                )

                importance_plot_main = gr.Plot(
                    show_label=False,
                    elem_classes=["chart-card"]
                )

            input_components = [
                longitude,
                latitude,
                housing_median_age,
                total_rooms,
                total_bedrooms,
                population,
                households,
                median_income,
                ocean_proximity
            ]

            output_components = [
                result_html,
                comparison_plot,
                income_plot,
                location_plot
            ]

            gr.HTML(
                '<div class="section-head">'
                '<div class="section-title">Sample neighbourhoods</div>'
                '<div class="section-hint">Click a row to fill the form, then press Estimate house price.</div>'
                '</div>'
            )

            gr.Examples(
                label="Try a sample neighbourhood",
                examples=[
                    [-122.23, 37.88, 30, 2000, 400, 1000, 350, 5.0, "NEAR BAY"],
                    [-118.49, 34.03, 25, 3200, 650, 1400, 600, 8.5, "NEAR OCEAN"],
                    [-119.77, 36.74, 20, 2600, 520, 1700, 480, 3.2, "INLAND"],
                    [-117.16, 32.72, 35, 1800, 420, 1300, 400, 4.1, "<1H OCEAN"]
                ],
                inputs=input_components
            )

            predict_button.click(
                fn=make_prediction,
                inputs=input_components,
                outputs=output_components
            )

        with gr.Tab("Market Insights"):

            gr.HTML(METRICS_HTML)

            with gr.Row():
                importance_plot = gr.Plot(
                    show_label=False,
                    elem_classes=["chart-card"]
                )

                ocean_plot = gr.Plot(
                    show_label=False,
                    elem_classes=["chart-card"]
                )

            with gr.Row(equal_height=True):

                with gr.Column(scale=5):
                    gauge_plot = gr.Plot(
                        show_label=False,
                        elem_classes=["chart-card"]
                    )

                with gr.Column(scale=6):
                    gr.HTML(INSIGHT_NOTES_HTML)

            demo.load(
                fn=lambda: (
                    feature_importance_chart(),
                    feature_importance_chart(),
                    ocean_average_chart(),
                    metrics_chart()
                ),
                outputs=[
                    importance_plot_main,
                    importance_plot,
                    ocean_plot,
                    gauge_plot
                ]
            )

        with gr.Tab("About"):

            with gr.Column(elem_classes=["about-card"]):
                gr.Markdown(ABOUT_MD)

    gr.HTML(
        '<div class="footer-text">'
        'DreamHome · Machine Learning Project · 2026 · Estimates only, not financial advice'
        '</div>'
    )


if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="amber",
            neutral_hue="stone"
        ),
        css=CUSTOM_CSS,
        server_name="127.0.0.1",
    
    )
