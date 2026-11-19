"""Offline, presentation-only components for the EcoWatt Streamlit workspace."""

from __future__ import annotations

from html import escape
import base64
import hashlib
import json
import re
from pathlib import Path

import streamlit as st

from .chart_tables import chart_table_markup
from .viewport import layout_width
from .workspace import PAGES

ROOT = Path(__file__).resolve().parents[1]
PALETTES = json.loads((ROOT / "config" / "palettes.json").read_text(encoding="utf-8"))
DEFAULT_PALETTE = next(iter(PALETTES))
# Navigation labels. They use the same full screen names as the sidebar radio.
PAGE_LABELS = ("Overview", "Forecast Studio", "Data Hub", "Model Lab", "Help")

# The mark is a meter needle crossing an hour tick: the product measures one
# household's hours, so the logo is built from the same parts as the interface.
BOLT = ('<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">'
        '<path d="M3 20h18" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>'
        '<path d="M6 20V9m6 11V4m6 16v-7" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>')
CHECK = '<svg viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="m3 8 3 3 7-7" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'


def html(markup: str) -> None:
    """Render trusted markup without Markdown treating indentation as code.

    Markdown shows a line that starts with 4 spaces as code, not as HTML.
    So we remove the spaces at the start of every line first. Only our own
    fixed HTML is passed here. Text from users or data must use escape() first.
    """
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def brand_assets() -> str:
    """Load licensed local fonts for the interface and numeric readouts."""
    # The font files are put inside the CSS as base64 text. This way the app
    # does not need to download fonts from the internet.
    faces = []
    for family, filename, weight, style in (
            ("Inter", "Inter.woff2", "100 900", "normal"),
            ("Martian Mono", "MartianMono.woff2", "100 900", "normal"),
    ):
        data = base64.b64encode((ROOT / "assets/brand" / filename).read_bytes()).decode("ascii")
        faces.append(f'@font-face{{font-family:"{family}";src:url(data:font/woff2;base64,{data}) '
                     f'format("woff2");font-weight:{weight};font-style:{style};font-display:swap;}}')
    return "".join(faces)


@st.dialog("Explore EcoWatt", width="large")
def navigation_menu(page: str) -> None:
    """A native modal keeps focus inside the menu and closes after navigation."""
    with st.container(key="navigation_menu"):
        html('<p class="ew-eyebrow">A clearer energy day</p>')
        for label, destination in zip(PAGE_LABELS, PAGES):
            if st.button(label, key=f"nav_{destination}", on_click=navigate, args=(destination,),
                         use_container_width=True,
                         type="primary" if page == destination else "secondary"):
                # Run the app again, so the pop-up closes and the new page opens.
                st.rerun()
        st.caption("Forecast Studio · Data Hub · Model Lab")
        st.caption("Reference data. Real predictions. One connected workspace.")


def shell(palette: str = DEFAULT_PALETTE, reduce_motion: bool = False) -> None:
    """Add the page style (colours, fonts, artwork) and start the top bar.

    The colours come from the chosen palette. If reduce_motion is True,
    animations are switched off.
    """
    colours = PALETTES[palette]
    # Every palette key becomes --ew-<key>. The older alias names are kept so
    # that any stylesheet rule written against them still resolves.
    legacy = {"panel": "surface", "raised": "surface-raised", "hover": "surface-hover",
              "primary": "emerald", "secondary": "cyan"}
    pairs = [f"--ew-{name}:{value}" for name, value in colours.items()]
    pairs += [f"--ew-{alias}:{colours[name]}" for name, alias in legacy.items() if name in colours]
    variables = ";".join(pairs)
    css = (ROOT / "assets" / "dashboard.css").read_text(encoding="utf-8")
    css += brand_assets() + (ROOT / "assets" / "atmosphere.css").read_text(encoding="utf-8")
    css += (ROOT / "assets" / "refine.css").read_text(encoding="utf-8")
    if reduce_motion:
        # Native dialogs are rendered in a portal outside .stApp.
        css += "\nhtml,html *,html *::before,html *::after {animation:none!important;transition:none!important;scroll-behavior:auto!important;}"
    # Local artwork keeps the interface available without a media CDN.
    photo = base64.b64encode((ROOT / "assets/brand/home-dusk.webp").read_bytes()).decode("ascii")
    garden = base64.b64encode((ROOT / "assets/brand/energy-garden.webp").read_bytes()).decode("ascii")
    # Streamlit sends style-only HTML to its event container, reserving no row.
    st.html(f'<style>:root{{{variables};--ew-home:url(data:image/webp;base64,{photo});'
            f'--ew-garden:url(data:image/webp;base64,{garden})}}{css}</style>')
    with st.container(key="layout_observer"):
        st.session_state["_layout_width"] = layout_width()
    page = st.session_state.get("workspace_page", "Overview")
    with st.container(key=f"masthead_{page}"):
        columns = st.columns([4, 3, 2, 1], vertical_alignment="center")
        with columns[0]:
            html(f'<div class="ew-mast"><span class="ew-mark">{BOLT}</span>'
                 '<span class="ew-wordmark">EcoWatt <span>AI</span></span></div>')
        with columns[1]:
            html('<span class="ew-mast-caption">A clearer energy day</span>')
        with columns[2]:
            st.button("Open studio ↗", key="header_studio", on_click=navigate,
                      args=("Forecast Studio",), use_container_width=True)
        with columns[3]:
            if st.button("Menu", key="open_navigation", use_container_width=True):
                navigation_menu(page)


def _accent(text: str) -> str:
    """Escape text, then turn *word* into an italic serif accent."""
    return re.sub(r"\*(.+?)\*", r"<em>\1</em>", escape(text))


def _page_title(title: str) -> str:
    """The last word of a page title is set in the italic serif."""
    head, _, tail = title.rpartition(" ")
    return f"{escape(head)} <em>{escape(tail)}</em>" if head else escape(title)


def hero_title(*lines: str) -> None:
    """Large headline, one line per row. Each row slides up in turn when the page opens."""
    rows = "".join(f'<span class="ew-line"><span style="--i:{i}">{_accent(line)}</span></span>'
                   for i, line in enumerate(lines))
    html(f'<h1 class="ew-title">{rows}</h1>')


def hero_lede(text: str) -> None:
    """Show one short intro paragraph under the main headline."""
    html(f'<p class="ew-hero-lede">{escape(text)}</p>')


def hero(headline: str, lede: str) -> None:
    """The opening: a short headline over the home photo, with the job in plain words."""
    html(f'<header class="ew-lede"><h1>{escape(headline)}</h1><p>{escape(lede)}</p></header>')


def opening_details() -> None:
    """Decorative particles and scene labels; none encode measured data."""
    dust = ''.join(f'<i style="--n:{n}"></i>' for n in range(12))
    html(f'<div class="ew-scene-dust" aria-hidden="true">{dust}</div>'
         '<div class="ew-opening-foot"><span>24 hourly predictions · Real reference data</span>'
         '<a href="#energy-story">Scroll to explore ↓</a><span>Original illustrative scene</span></div>')


def home_story() -> None:
    """Show the short story section on the Overview page (fixed text)."""
    html('<section class="ew-story" id="energy-story">'
         '<p class="ew-eyebrow">A little more understanding</p>'
         '<div class="ew-story-grid"><h2>Energy is part<br>of <em>every day.</em></h2>'
         '<div><p>A kettle in the morning. Lights in the evening. A home’s electricity use follows a rhythm.</p>'
         '<p>EcoWatt uses recorded hours to forecast the next day, so you can explore the likely total and peak hour.</p>'
         '<p class="ew-story-small">One household at a time. Forecasts support planning; they do not guarantee savings.</p></div></div>'
         '<div class="ew-home-scene" role="img" aria-label="Illustration of a warmly lit home at dusk">'
         '<span class="ew-eyebrow">Illustrative home</span>'
         '<p>Small patterns.<br><em>A clearer picture.</em></p></div>'
         '<div class="ew-facts"><div><strong>24</strong><span>Hours in every forecast</span></div>'
         '<div><strong>3</strong><span>Benchmark methods</span></div>'
         '<div><strong>2</strong><span>Reference households</span></div></div></section>')


def overview_evidence(metrics) -> None:
    """Render the original scores with one common zero-baseline scale.

    Input: the test scores table. Each model gets one bar. The best model
    (lowest MAE) is first.
    """
    ordered = metrics.sort_values("mae_kwh")
    # Every bar width = its MAE divided by the same maximum, so bars can be
    # compared fairly. The 1.1 leaves a little space after the longest bar.
    maximum = max(float(ordered.mae_kwh.max()) * 1.1, 1e-12)
    rows = []
    for row in ordered.itertuples():
        rows.append(f'''<li><div><h3>{escape(row.model)}</h3><span>RMSE {row.rmse_kwh:.4f} kWh</span></div>
        <div class="ew-evidence-track" aria-hidden="true"><i style="width:{row.mae_kwh/maximum*100:.6f}%"></i></div>
        <div class="ew-evidence-value"><strong>{row.mae_kwh:.4f}</strong><span>MAE kWh</span></div></li>''')
    html('<ol class="ew-evidence">' + ''.join(rows) + '</ol>')
    st.caption("Lower is better. Bars share a zero baseline and one scale. These source-household test scores do not measure accuracy on your upload.")


def hour_ribbon(values, start_hour: int, *, threshold: float | None = None,
                measured: bool = True, caption: str = "") -> None:
    """Draw one cell per hour of the day, from the values the caller supplies.

    Heights are proportional to the real values; the exact numbers stay in the
    table beside this. Night hours carry a cooler ground, so the shape of the
    day is readable without reading the axis.

    Inputs: the values (kWh), the clock hour of the first value, an optional
    high-demand threshold, True if the values are measured (False if they are
    a forecast), and an optional caption text.
    """
    numbers = [float(value) for value in values]
    # The tallest bar is 100% high. The tiny number 1e-12 avoids dividing by zero.
    maximum = max(max(numbers, default=0.0), 1e-12)
    role = "measured" if measured else "forecast"
    cells, axis = [], []
    for index, value in enumerate(numbers):
        # "% 24" makes the hour go back to 0 after 23 (the day wraps around).
        hour = (start_hour + index) % 24
        # Night is before 06:00 or from 21:00. "peak" marks the tallest bar.
        night ="1" if hour < 6 or hour >= 21 else "0"
        peak = "1" if value == max(numbers, default=0.0) else "0"
        kind = "high" if (threshold is not None and not measured and value >= threshold) else role
        cells.append(f'<span data-night="{night}" data-peak="{peak}">'
                     f'<i data-role="{kind}" style="height:{value/maximum*100:.6f}%;--ew-hour:{index}"></i></span>')
        axis.append(f'<span data-show="{"1" if hour % 6 == 0 else "0"}">{hour:02d}</span>')
    keys = [('ew-key-' + ('measured' if measured else 'forecast'),
             'Recorded hour' if measured else 'Forecast hour')]
    if threshold is not None and not measured:
        keys.append(("ew-key-high", f"At or above {threshold:.2f} kWh"))
    keys.append(("ew-key-night", "Night hours"))
    legend = "".join(f'<b class="{name}">{escape(text)}</b>' for name, text in keys)
    html(f'<div class="ew-ribbon"><div class="ew-ribbon-grid" aria-hidden="true">{"".join(cells)}</div>'
         f'<div class="ew-ribbon-axis" aria-hidden="true">{"".join(axis)}</div>'
         f'<div class="ew-ribbon-legend">{legend}</div></div>')
    if caption:
        st.caption(caption)


def recorded_bars(recent) -> None:
    """Twenty-four recorded hours as the hour ribbon, measured values only."""
    hour_ribbon(recent.energy_kwh, int(recent.index[0].hour), measured=True,
                caption=f"Recorded hourly energy in kWh, {recent.index[0]:%d %b %H:%M} to "
                        f"{recent.index[-1]:%d %b %H:%M}. These are measurements, not predictions.")


def sidebar_brand() -> None:
    """Show the EcoWatt AI logo and name at the top of the sidebar."""
    html(f'<div class="ew-sidebar-brand"><span class="ew-mark">{BOLT}</span>'
         '<span class="ew-wordmark">EcoWatt <span>AI</span></span></div>')


def sidebar_step(number: str, title: str) -> None:
    """Show a small heading in the sidebar. (Only the title is shown; number is not used.)"""
    html(f'<div class="ew-sidebar-step">{escape(title)}</div>')


def section(title: str, description: str) -> None:
    """A heading whose words carry the meaning; these screens are not a sequence."""
    html(f'<div class="ew-section"><h2 class="ew-section-title">{_accent(title)}</h2>'
         f'<p class="ew-section-copy">{escape(description)}</p></div>')


def spec(items: list[tuple[str, str]]) -> None:
    """Labelled pairs, so no value is left to be guessed from a run-on line."""
    content = ''.join(f'<div><p class="ew-spec-key">{escape(label)}</p>'
                      f'<p class="ew-spec-value">{escape(value)}</p></div>' for label, value in items)
    html(f'<div class="ew-spec">{content}</div>')


# Kept as the former name used across the views.
source_strip = spec


def summary(items: list[tuple[str, str]]) -> None:
    """Show a row of small notes. Each item is a (label, value) pair."""
    content = ''.join(f'<div class="ew-note"><strong>{escape(label)}</strong><span>{escape(value)}</span></div>'
                      for label, value in items)
    html(f'<div class="ew-notes">{content}</div>')


def empty_state(stale: bool = False) -> None:
    """An empty result area says what is missing and what the next action is."""
    title = "The settings changed" if stale else "No forecast yet"
    text = ("The earlier result no longer matches the data and settings you have now. "
            "Run the forecast again to see the hours for this selection."
            if stale else
            "Choose how far back to start, then run the forecast. The 24 hourly values, "
            "the chart and the downloadable table appear here.")
    html(f'<div class="ew-empty"><h3>{escape(title)}</h3><p>{escape(text)}</p>'
         '<p class="ew-empty-next">Next: Run the 24-hour forecast</p></div>')


def footer() -> None:
    """Show the footer line at the bottom of every screen."""
    html('<footer class="ew-footer"><span><strong>EcoWatt AI</strong> — Professional Training project, '
         'Yunnan University</span><span>Recorded smart-meter data. Energy in kWh. No live meter feed.</span></footer>')


def page_header(title: str, description: str) -> None:
    """Show the heading area at the top of a screen, with a short description."""
    display = {
        "Forecast Studio": "Meet the<br>next <em>24 hours.</em>",
        "Data Hub": "Good forecasts<br>begin with <em>history.</em>",
        "Model Lab": "Evidence behind<br>the <em>outlook.</em>",
        "Help": "A little<br><em>clarity.</em>",
    }.get(title, _page_title(title))
    html(f'<header class="ew-page-header"><div><span class="ew-eyebrow">{escape(title)}</span>'
         f'<h1>{display}</h1></div><p>{escape(description)}</p></header>')


def navigate(page: str) -> None:
    """Native widget callback runs before the navigation radio is recreated.

    It remembers the page name in st.session_state (the app's memory between
    clicks). The app reads this name on the next run and shows that page.
    """
    st.session_state["workspace_page"] = page


def start_guided_demo() -> None:
    """A native callback selects real reference input before the app reruns.

    It sets the choices for the one-click demo: XKY dataset, built-in demo
    data, HistGradientBoosting and the Forecast Studio page. The last line is
    a flag. Forecast Studio sees it and runs the forecast by itself.
    """
    st.session_state["benchmark_dataset"] = "xky_2025"
    st.session_state["source_mode"] = "Built-in dataset demo"
    st.session_state["model_xky_2025"] = "HistGradientBoosting"
    st.session_state["workspace_page"] = "Forecast Studio"
    st.session_state["guided_demo_requested"] = True


def replay() -> None:
    """Add 1 to a counter, so the page can play its intro motion again."""
    st.session_state["motion_replay"] = st.session_state.get("motion_replay", 0) + 1


def card(title: str, text: str, tag: str = "") -> None:
    """Show one feature card with a small tag, a title and a short text."""
    html(f'<article class="ew-feature-card"><span class="ew-eyebrow">{escape(tag)}</span><h3>{escape(title)}</h3><p>{escape(text)}</p></article>')


SHORT_AXIS_UNIT = re.compile(r"^(?P<name>.+?)\s*\((?P<unit>[^()]{1,12})\)$")


def _shorten_axis(figure, narrow: bool) -> None:
    """Keep the rotated y-axis label to its unit and move the words to the subtitle.

    A long label rotated by 90 degrees is hard to read and eats chart width, so
    the unit stays on the axis and the full wording goes under the title.
    """
    label = figure.layout.yaxis.title.text
    match = SHORT_AXIS_UNIT.match(label or "")
    if match is None:
        return
    figure.update_yaxes(title_text=match.group("unit"))
    if not figure.layout.title.subtitle or not figure.layout.title.subtitle.text:
        figure.update_layout(title_subtitle_text=f"{match.group('name')}, in {match.group('unit')}")
    if narrow:
        # On a phone even the unit competes with the numbers, so drop it; the
        # subtitle above the chart still names the unit.
        figure.update_yaxes(title_text=None)


def chart(figure, palette: str = DEFAULT_PALETTE, reduce_motion: bool = False) -> None:
    """Style a new figure per session; never change shared model or chart globals.

    Inputs: a Plotly figure, the colour palette name, and the reduce-motion
    setting. It changes only the look of the figure (colours, fonts, margins),
    never the data values. Then it draws the chart and, below it, an HTML table
    with the same values for people who cannot use the chart.
    """
    colours = PALETTES[palette]
    # Build the accessible table from the figure as the view supplied it, before
    # any display-only shortening, so the table keeps the full axis wording.
    table_markup = chart_table_markup(figure)
    narrow = 0 < int(st.session_state.get("_layout_width", 0) or 0) < 640
    _shorten_axis(figure, narrow)
    figure.update_layout(
        autosize=True,
        width=None,
        paper_bgcolor=colours["panel"], plot_bgcolor=colours["panel"],
        font=dict(family="Inter, sans-serif", size=14, color=colours["text"]), title_font_color=colours["text"],
        legend_font_color=colours["muted"], legend_font_size=12, title_font_size=15, transition_duration=0,
        hoverlabel_bgcolor=colours["raised"], hoverlabel_font_color=colours["text"],
        title_subtitle_font_color=colours["muted"], title_subtitle_font_size=12,
    )
    # Make the grid line colour from the text colour: convert its hex code to
    # red, green and blue numbers, then use only 10% opacity (a faint line).
    grid = f'rgba({",".join(str(int(colours["text"].lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))},0.10)'
    figure.update_xaxes(tickfont_color=colours["muted"], tickfont_size=12, title_font_color=colours["muted"],
                        linecolor=grid, gridcolor=grid, zerolinecolor=grid)
    figure.update_yaxes(tickfont_color=colours["muted"], tickfont_size=12, title_font_color=colours["muted"],
                        linecolor=grid, gridcolor=grid, zerolinecolor=grid)
    if narrow:
        # Fewer ticks and tighter margins keep labels from overlapping on a phone.
        figure.update_xaxes(nticks=5, tickfont_size=11)
        figure.update_yaxes(nticks=5, tickfont_size=11)
        # A smaller subtitle (still 11px) helps the whole line fit on a phone.
        figure.update_layout(title_subtitle_font_size=11)
        margin = figure.layout.margin
        figure.update_layout(margin={"l": min(margin.l or 50, 54), "r": 12,
                                     "t": margin.t or 72, "b": margin.b or 70})
    for annotation in figure.layout.annotations:
        if annotation.bgcolor:
            annotation.bgcolor = colours["panel"]
            annotation.font.color = colours["text"]
        if annotation.arrowcolor:
            annotation.arrowcolor = colours["muted"]
    # Give each line or bar the colour of the active palette. Each model has
    # its own colour and line style, so the models are easy to tell apart.
    for trace in figure.data:
        if trace.type == "scatter":
            # Colour carries meaning: a prediction is never the colour of a measurement.
            model_colours = {"HistGradientBoosting": colours["primary"], "Ridge": colours["secondary"], "Seasonal Naive": colours["violet"]}
            named = {"Observed": colours["subtle"], "Recorded input": colours["subtle"],
                     "Forecast peak": colours["text"], "Simple rule (7-day median)": colours["muted"]}
            colour = model_colours.get(trace.name, named.get(trace.name, colours["primary"]))
            trace.update(line_color=colour, marker_color=colour)
            if trace.name in model_colours:
                trace.line.dash = {"HistGradientBoosting": "solid", "Ridge": "dash", "Seasonal Naive": "dot"}[trace.name]
            if trace.fill == "tozeroy":
                trace.update(fillcolor=f"rgba({colours['rgb']},0.08)")
        elif trace.type == "bar":
            current = trace.marker.color
            trace.marker.color = [colours["primary"]] + [colours["secondary"]] * (len(current) - 1) if isinstance(current, (tuple, list)) else colours["primary"]
            if trace.error_x.visible:
                trace.error_x.color = colours["secondary"]
    # Keep native frontend chart state separate when different pages reuse a
    # Streamlit element slot. In particular, old canvas widths must not follow
    # a chart into a different page or hidden tab.
    page = st.session_state.get("workspace_page", "Overview")
    identity = f"{page}:{figure.layout.title.text}:{[trace.type for trace in figure.data]}:{st.session_state.get('_layout_width', 0)}"
    chart_key = "energy_chart_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
    st.plotly_chart(figure, key=chart_key, use_container_width=True, theme=None,
                    config={"displaylogo": False, "displayModeBar": False, "scrollZoom": False, "responsive": True})
    html(table_markup)
