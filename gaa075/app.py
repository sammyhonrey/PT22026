"""Music Recommendation System — single-page Streamlit interface.

One page, top to bottom:

    A. Header
    B. Song search
    C. Selected song
    D. Optional preferences
    E. Number of recommendations
    F. "Find Similar Songs"
    G. Results (cards) + a premium Now Playing experience

Presentation only: scores, ranking, explanations and duplicate rules all come
from :class:`recommender.Recommender`. Audio previews come from :mod:`previews`
(public iTunes Search API, official ~30 s previews streamed by Apple).

Opening a preview expands a dark-violet Now Playing view inside the results
component: circular verified artwork, progress with elapsed/remaining time,
play/pause, a circular replay icon, next, and clear status. Closing it returns to
the recommendations without stopping playback or losing the selected song.

Run with::

    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import time
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from arabic_catalogue import (
    ArabicCatalogue,
    ArabicCatalogueError,
    load_arabic_catalogue,
    load_artist_aliases,
    provenance as arabic_provenance,
)
from preprocessing import DEFAULT_DATA_PATH, DOWNLOAD_HINT, ID_COLUMN, build_feature_matrix
from previews import PREVIEW_SECONDS, find_preview, resolve_many, search_url
from recommender import (
    InvalidRequestError,
    Preferences,
    Recommender,
    SCORE_INTERPRETATION_WARNING,
    UnknownTrackError,
)

APP_TITLE = "Music Recommendation System"
SUBTITLE = "Find songs that sound like the one you love — no account needed."
MAX_SEARCH_RESULTS = 25
METADATA_SCORE_NOTE = (
    "Arabic mode uses metadata matching only: same artist, same genre, same region. "
    "No audio features exist for these records, so this score is not an audio similarity "
    "and must not be compared with the international audio scores."
)


# --------------------------------------------------------------------------- #
# Pure helpers (importable and testable without running Streamlit)
# --------------------------------------------------------------------------- #


def format_score(value: Any) -> str:
    """Raw cosine similarity with three decimals. Never a percentage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:.3f}"


def track_label(title: Any, artists: Any) -> str:
    """Plain-text label for the search results list."""
    return f"{str(title or 'Unknown track')} — {str(artists or 'Unknown artist')}"


def genres_display(genres: Any) -> str:
    """Comma-separated genre text for a result card."""
    if isinstance(genres, str):
        return ", ".join(part.strip() for part in genres.split(",") if part.strip())
    return ", ".join(str(g).strip() for g in (genres or ()) if str(g).strip())


def search_tracks(catalogue: pd.DataFrame, query: str, limit: int = MAX_SEARCH_RESULTS) -> pd.DataFrame:
    """Case-insensitive title/artist search.

    An empty query returns the most popular tracks, so the page always offers
    something to select.
    """
    query = (query or "").strip().casefold()
    if not query:
        ordered = catalogue.reset_index(drop=True).sort_values(
            ["popularity", ID_COLUMN], ascending=[False, True]
        )
        return ordered.head(limit)

    titles = catalogue["track_name"].fillna("").astype(str).str.casefold()
    artists = catalogue["artists"].fillna("").astype(str).str.casefold()
    mask = titles.str.contains(query, regex=False) | artists.str.contains(query, regex=False)
    hits = catalogue.loc[mask].reset_index(drop=True).sort_values(
        ["popularity", ID_COLUMN], ascending=[False, True]
    )
    return hits.head(limit)


def build_result_payload(
    results: pd.DataFrame,
    previews_by_id: dict[str, Any],
) -> list[dict]:
    """Shape engine results + previews into the data the player needs."""
    payload: list[dict] = []
    for _, row in results.iterrows():
        track_id = str(row["track_id"])
        preview = previews_by_id.get(track_id)
        payload.append(
            {
                "rank": int(row["rank"]),
                "track_id": track_id,
                "title": str(row["track_name"]),
                "artist": str(row["artists"]),
                "genres": genres_display(row["genres"]),
                "score": format_score(row["score"]),
                "explanation": str(row["explanation"]),
                "preview_url": preview.url if preview else "",
                "external_url": preview.external_url
                if preview
                else search_url(str(row["track_name"]), str(row["artists"])),
                "artwork_url": preview.artwork_url if preview else "",
            }
        )
    return payload


def results_height(count: int) -> int:
    """Pixel height for the embedded component (mini player + cards + player room)."""
    return 250 + max(1, count) * 122


def build_arabic_payload(results: pd.DataFrame, previews_by_id: dict[str, Any]) -> list[dict]:
    """Shape Arabic metadata results into the player payload (no audio scores)."""
    payload: list[dict] = []
    for _, row in results.iterrows():
        track_id = str(row["track_id"])
        preview = previews_by_id.get(track_id)
        payload.append(
            {
                "rank": int(row["rank"]),
                "track_id": track_id,
                "title": str(row["title"]),
                "artist": str(row["artist"]),
                "genres": f"{row['genre']} · {row['region']}",
                "score": f"{float(row['metadata_score']):.2f}",
                "explanation": str(row["explanation"]),
                "preview_url": preview.url if preview else "",
                "external_url": preview.external_url
                if preview
                else search_url(str(row["title"]), str(row["artist"])),
                "artwork_url": preview.artwork_url if preview else "",
            }
        )
    return payload


def previews_disabled() -> bool:
    """Test/deployment hook: ``MRS_DISABLE_PREVIEWS=1`` skips network lookups."""
    return os.environ.get("MRS_DISABLE_PREVIEWS") == "1"


# --------------------------------------------------------------------------- #
# Results + Now Playing component
# --------------------------------------------------------------------------- #

ICON_PLAY = (
    '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">'
    '<path d="M8 5.5v13a1 1 0 0 0 1.53.85l10-6.5a1 1 0 0 0 0-1.7l-10-6.5A1 1 0 0 0 8 5.5z"/></svg>'
)
ICON_PAUSE = (
    '<svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor" aria-hidden="true">'
    '<rect x="6.5" y="5" width="4" height="14" rx="1.4"/><rect x="13.5" y="5" width="4" height="14" rx="1.4"/></svg>'
)
ICON_REPLAY = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1"/><polyline points="3.5 3.8 3.5 9.2 8.9 9.2"/></svg>'
)
ICON_NEXT = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="currentColor" aria-hidden="true">'
    '<path d="M5.5 5.6v12.8a1 1 0 0 0 1.55.83l8.2-6.4a1 1 0 0 0 0-1.66l-8.2-6.4A1 1 0 0 0 5.5 5.6z"/>'
    '<rect x="16.6" y="5.2" width="2.6" height="13.6" rx="1.3"/></svg>'
)
ICON_BACK = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<polyline points="14.5 5 8 12 14.5 19"/></svg>'
)

PLAYER_STYLE = """
<style>
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0 2px 14px 2px;
    font-family: -apple-system, "Segoe UI", Inter, Roboto, Arial, sans-serif;
    background: #090611; color: #F5F0FF;
  }
  .mini {
    position: sticky; top: 0; z-index: 5;
    background: linear-gradient(150deg, #170D2B, #0E0819);
    border: 1px solid rgba(150,59,255,0.28); border-radius: 16px;
    padding: 12px 16px; margin-bottom: 14px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
  }
  .mini-row { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
  .mini-now { flex: 1 1 240px; min-width: 0; cursor: pointer; }
  .mini-label { font-size: 10px; letter-spacing: .18em; text-transform: uppercase; color: #B66BFF; }
  .mini-title { font-size: 14px; font-weight: 650; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .mini-artist { font-size: 12px; color: #B9A8D9; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .mini-controls { display: flex; gap: 8px; align-items: center; }
  button.ctrl {
    display: inline-flex; align-items: center; justify-content: center;
    border: 1px solid rgba(150,59,255,0.35); background: rgba(150,59,255,0.10);
    color: #EDE4FF; border-radius: 50%; width: 40px; height: 40px; cursor: pointer;
    transition: transform .12s ease, background .12s ease, box-shadow .12s ease;
  }
  button.ctrl:hover:not(:disabled) { background: rgba(150,59,255,0.24); box-shadow: 0 0 18px rgba(182,107,255,0.35); }
  button.ctrl:disabled { opacity: .4; cursor: not-allowed; }
  button.ctrl.primary {
    width: 44px; height: 44px; border: 0; color: #150A26;
    background: linear-gradient(145deg, #B66BFF, #963BFF);
    box-shadow: 0 6px 22px rgba(150,59,255,0.45);
  }
  button.ctrl.primary:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 8px 26px rgba(182,107,255,0.55); }
  .card {
    display: flex; gap: 14px; align-items: flex-start;
    background: linear-gradient(160deg, #150C27, #0D0818);
    border: 1px solid rgba(150,59,255,0.16); border-radius: 16px;
    padding: 14px 16px; margin-bottom: 10px;
    transition: border-color .16s ease, box-shadow .16s ease, transform .16s ease;
  }
  .card:hover { border-color: rgba(182,107,255,0.45); box-shadow: 0 10px 28px rgba(0,0,0,0.55); transform: translateY(-1px); }
  .card.playing { border-color: #963BFF; box-shadow: 0 0 0 1px rgba(150,59,255,0.6) inset, 0 0 26px rgba(150,59,255,0.24); }
  .art { width: 58px; height: 58px; border-radius: 12px; object-fit: cover; flex: 0 0 58px; }
  .art-fallback {
    display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle at 30% 25%, rgba(182,107,255,0.35), rgba(23,13,43,0.9));
    color: #D9C8FF; font-size: 22px;
  }
  .body { flex: 1 1 auto; min-width: 0; }
  .title { font-size: 15px; font-weight: 650; line-height: 1.3; color: #F7F2FF; }
  .artist { font-size: 12.5px; color: #B9A8D9; margin-top: 1px; }
  .genres { font-size: 11.5px; color: #C79BFF; margin-top: 5px; }
  .explain { font-size: 12px; color: #9C8CBE; line-height: 1.5; margin-top: 6px; }
  .right { flex: 0 0 auto; text-align: right; display: flex; flex-direction: column; gap: 6px; align-items: flex-end; }
  .score { font-size: 18px; font-weight: 700; color: #C79BFF; line-height: 1; }
  .score-label { font-size: 10px; letter-spacing: .12em; text-transform: uppercase; color: #7E6FA3; }
  button.play {
    display: inline-flex; align-items: center; gap: 7px;
    border: 1px solid rgba(150,59,255,0.5); background: rgba(150,59,255,0.12);
    color: #EADCFF; border-radius: 10px; padding: 7px 13px; font-size: 12px;
    font-weight: 650; cursor: pointer; transition: background .12s ease, box-shadow .12s ease;
  }
  button.play:hover { background: rgba(150,59,255,0.3); box-shadow: 0 0 16px rgba(150,59,255,0.3); }
  .unavailable { font-size: 11.5px; color: #8E7FAE; }
  .listen { font-size: 11.5px; color: #B66BFF; text-decoration: none; }
  .listen:hover { text-decoration: underline; }
  .empty { padding: 18px; border: 1px solid rgba(150,59,255,0.2); border-radius: 14px; color: #B9A8D9; background: #120A22; }

  /* ---------------- Now Playing overlay ---------------- */
  .np {
    position: fixed; inset: 0; z-index: 40;
    display: flex; flex-direction: column;
    background:
      radial-gradient(900px 420px at 50% -8%, rgba(150,59,255,0.34), transparent 62%),
      radial-gradient(700px 400px at 50% 108%, rgba(182,107,255,0.16), transparent 65%),
      linear-gradient(180deg, #0B0716 0%, #090611 55%, #0C0718 100%);
    opacity: 0; transform: translateY(14px) scale(0.995);
    transition: opacity .2s ease, transform .2s ease;
    padding: 18px 20px 26px 20px; overflow: auto;
  }
  .np[hidden] { display: none; }
  .np.open { opacity: 1; transform: none; }
  .np-top { display: flex; align-items: center; gap: 12px; }
  .np-back {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; border-radius: 50%;
    border: 1px solid rgba(182,107,255,0.35); background: rgba(23,13,43,0.7);
    color: #EADCFF; cursor: pointer; transition: background .15s ease, box-shadow .15s ease;
  }
  .np-back:hover { background: rgba(150,59,255,0.22); box-shadow: 0 0 18px rgba(182,107,255,0.35); }
  .np-label { font-size: 11px; letter-spacing: .28em; text-transform: uppercase; color: #C79BFF; }
  .np-center { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding-top: 10px; }
  .np-art-wrap {
    position: relative; width: 246px; height: 246px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; margin: 8px 0 22px 0;
    animation: floatIn .45s ease both;
  }
  .np-ring {
    position: absolute; inset: -14px; border-radius: 50%;
    background: conic-gradient(from 140deg, rgba(150,59,255,0.0), rgba(182,107,255,0.85), rgba(150,59,255,0.0) 70%);
    filter: blur(1px); opacity: .85;
    animation: spin 9s linear infinite;
  }
  .np-ring::after {
    content: ""; position: absolute; inset: 5px; border-radius: 50%; background: #090611;
  }
  .np-art {
    position: relative; width: 246px; height: 246px; border-radius: 50%;
    object-fit: cover; z-index: 2;
    box-shadow: 0 0 0 1px rgba(255,255,255,0.08), 0 22px 60px rgba(0,0,0,0.65),
                0 0 70px rgba(150,59,255,0.45);
  }
  .np-art.fallback {
    display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle at 32% 28%, #3B2270, #170D2B 70%);
    color: #C79BFF; font-size: 54px; z-index: 2;
  }
  .np-title { font-size: 24px; font-weight: 700; line-height: 1.25; max-width: 92%; color: #FFFFFF; }
  .np-artist { font-size: 14px; color: #B9A8D9; margin-top: 6px; max-width: 92%; }
  .np-genres { font-size: 12px; color: #8E7FAE; margin-top: 8px; }
  .np-progress-block { width: min(560px, 92%); margin: 26px auto 0 auto; }
  .np-times { display: flex; justify-content: space-between; font-size: 11.5px; color: #9C8CBE; margin-bottom: 7px; font-variant-numeric: tabular-nums; }
  .np-bar { height: 5px; border-radius: 99px; background: rgba(150,59,255,0.18); overflow: hidden; }
  .np-bar > span {
    display: block; height: 100%; width: 0%;
    background: linear-gradient(90deg, #963BFF, #B66BFF);
    box-shadow: 0 0 14px rgba(182,107,255,0.7); transition: width .18s linear;
  }
  .np-controls { display: flex; align-items: center; justify-content: center; gap: 26px; margin-top: 24px; }
  .np-controls .ctrl { width: 48px; height: 48px; }
  .np-controls .primary { width: 68px; height: 68px; }
  .np-controls .primary svg { width: 28px; height: 28px; }
  .np-status { margin-top: 16px; font-size: 12.5px; color: #A896C9; min-height: 18px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes floatIn { from { opacity: 0; transform: translateY(10px) scale(.98); } to { opacity: 1; transform: none; } }
  @media (max-width: 640px) {
    .np { padding: 14px 14px 22px 14px; }
    .np-art-wrap, .np-art { width: 196px; height: 196px; }
    .np-title { font-size: 20px; }
    .np-controls { gap: 18px; }
    .np-controls .primary { width: 62px; height: 62px; }
    .card { padding: 12px; gap: 10px; }
    .explain { display: none; }
  }
  @media (prefers-reduced-motion: reduce) {
    .np, .np-ring, .card, button { animation: none !important; transition: none !important; }
  }
</style>
"""

PLAYER_SCRIPT = """
<script>
(function () {
  const TRACKS = __TRACKS__;
  const LIMIT = __LIMIT__;
  const ICON_PLAY = '__ICON_PLAY__';
  const ICON_PAUSE = '__ICON_PAUSE__';

  const audio = new Audio();
  audio.preload = "metadata";

  let current = -1;
  let finished = false;
  let overlayOpen = false;

  const $ = (id) => document.getElementById(id);
  const np = $("np");

  function fmt(seconds) {
    const s = Math.max(0, Math.round(seconds));
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }

  function playableIndexes() {
    return TRACKS.filter((t) => t.preview_url).map((t) => TRACKS.indexOf(t));
  }

  function effectiveDuration() {
    const real = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.duration : LIMIT;
    return Math.min(real, LIMIT);
  }

  function paintProgress() {
    const dur = effectiveDuration();
    const elapsed = Math.min(audio.currentTime || 0, dur);
    const ratio = dur > 0 ? (elapsed / dur) * 100 : 0;
    $("np-progress").style.width = Math.min(100, ratio) + "%";
    $("np-elapsed").textContent = fmt(elapsed);
    $("np-remaining").textContent = "-" + fmt(Math.max(0, dur - elapsed));
  }

  function setToggle(playing) {
    $("np-toggle").innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
    $("np-toggle").setAttribute("aria-label", playing ? "Pause preview" : "Play preview");
    $("mini-toggle").innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
    $("mini-toggle").setAttribute("aria-label", playing ? "Pause preview" : "Play preview");
  }

  function setStatus(text) {
    $("np-status").textContent = text;
  }

  function highlight() {
    document.querySelectorAll(".card").forEach((c) => c.classList.remove("playing"));
    if (current >= 0) {
      const card = document.getElementById("card-" + TRACKS[current].rank);
      if (card) card.classList.add("playing");
    }
  }

  function paintTrack() {
    const track = TRACKS[current];
    if (!track) return false;
    $("np-title").textContent = track.title;
    $("np-artist").textContent = track.artist;
    $("np-genres").textContent = track.genres;
    $("mini-title").textContent = track.title;
    $("mini-artist").textContent = track.artist;
    $("np-art").style.display = "none";
    $("np-fallback").style.display = "flex";
    if (track.artwork_url) {
      $("np-art").src = track.artwork_url;
      $("np-art").style.display = "block";
      $("np-fallback").style.display = "none";
    }
    paintProgress();
    highlight();
    return true;
  }

  function openOverlay() {
    np.hidden = false;
    requestAnimationFrame(() => np.classList.add("open"));
    overlayOpen = true;
  }

  function closeOverlay() {
    np.classList.remove("open");
    overlayOpen = false;
    setTimeout(() => {
      if (!overlayOpen) np.hidden = true;
    }, 200);
  }

  function load(index, autoplay) {
    const track = TRACKS[index];
    if (!track || !track.preview_url) {
      setStatus("Preview unavailable for this track.");
      return false;
    }
    current = index;
    finished = false;
    audio.src = track.preview_url;
    audio.currentTime = 0;
    paintTrack();
    paintProgress();
    if (autoplay) {
      audio.play().then(() => setToggle(true)).catch(() => {
        setToggle(false);
        setStatus("Your browser blocked playback — press play.");
      });
    } else {
      setToggle(false);
    }
    return true;
  }

  function playAfterStop() {
    if (current < 0) {
      const first = playableIndexes()[0];
      if (first === undefined) {
        setStatus("No legally available previews for these results.");
        return;
      }
      if (load(first, true)) setStatus("Playing 30-second preview.");
      return;
    }
    if (finished) {
      audio.currentTime = 0;
      finished = false;
    }
    audio.play().then(() => setToggle(true)).catch(() => setStatus("Your browser blocked playback — press play."));
    setStatus("Playing 30-second preview.");
  }

  function stopPreview(message) {
    audio.pause();
    audio.currentTime = 0;
    finished = true;
    setToggle(false);
    paintProgress();
    setStatus(message);
  }

  function nextPreview() {
    const indexes = playableIndexes();
    if (!indexes.length) {
      setStatus("No legally available previews for these results.");
      return;
    }
    const position = indexes.indexOf(current);
    const nextIndex = indexes[(position + 1) % indexes.length];
    if (load(nextIndex, true)) setStatus("Playing next preview.");
  }

  /* ---- wire up ---- */
  document.querySelectorAll("button.play").forEach((button) => {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.idx);
      if (!load(index, true)) {
        openOverlay();
        return;
      }
      openOverlay();
      setStatus("Playing 30-second preview.");
    });
  });

  $("np-back").addEventListener("click", closeOverlay);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && overlayOpen) closeOverlay();
  });

  $("np-toggle").addEventListener("click", () => {
    if (!audio.paused) {
      audio.pause();
      setToggle(false);
      setStatus("Paused.");
      return;
    }
    playAfterStop();
  });

  $("mini-toggle").addEventListener("click", () => {
    if (!audio.paused) {
      audio.pause();
      setToggle(false);
      setStatus("Paused.");
      return;
    }
    playAfterStop();
  });

  $("np-replay").addEventListener("click", () => {
    if (current < 0) {
      playAfterStop();
      return;
    }
    finished = false;
    audio.currentTime = 0;
    audio.play().then(() => setToggle(true)).catch(() => setStatus("Your browser blocked playback — press play."));
    paintProgress();
    setStatus("Replaying from the beginning.");
  });

  $("mini-replay").addEventListener("click", () => {
    $("np-replay").click();
  });

  $("np-next").addEventListener("click", nextPreview);
  $("mini-next").addEventListener("click", nextPreview);
  $("mini-open").addEventListener("click", () => {
    if (current >= 0) openOverlay();
  });

  audio.addEventListener("timeupdate", () => {
    paintProgress();
    if (audio.currentTime >= LIMIT) stopPreview("Preview ended at 30 seconds. Press replay to hear it again.");
  });

  audio.addEventListener("loadedmetadata", paintProgress);
  audio.addEventListener("pause", () => setToggle(false));
  audio.addEventListener("play", () => setToggle(true));
  audio.addEventListener("ended", () => stopPreview("Preview finished. Press replay to hear it again."));

  if (!playableIndexes().length) {
    $("mini-toggle").disabled = true;
    $("mini-replay").disabled = true;
    $("mini-next").disabled = true;
    $("mini-open").disabled = true;
    setToggle(false);
    setStatus("No legally available previews were found for these results.");
  }
  setToggle(false);
})();
</script>
"""


def render_results_html(
    payload: list[dict],
    preview_seconds: int = PREVIEW_SECONDS,
    score_label: str = "similarity",
) -> str:
    """Build the self-contained results + Now Playing component (one audio element)."""
    if not payload:
        return '<div class="empty">No recommendations are available for this track.</div>'

    ready = sum(1 for item in payload if item["preview_url"])

    cards = []
    for item in payload:
        title = html.escape(item["title"])
        artist = html.escape(item["artist"])
        genres = html.escape(item["genres"])
        explanation = html.escape(item["explanation"])
        score = html.escape(item["score"])
        artwork = (
            f'<img class="art" src="{html.escape(item["artwork_url"])}" alt="" loading="lazy"/>'
            if item["artwork_url"]
            else '<div class="art art-fallback">♪</div>'
        )
        if item["preview_url"]:
            action = (
                f'<button class="play" data-idx="{item["rank"] - 1}" '
                f'aria-label="Play preview of {title}">'
                f'<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" aria-hidden="true">'
                f'<path d="M8 5.5v13a1 1 0 0 0 1.53.85l10-6.5a1 1 0 0 0 0-1.7l-10-6.5A1 1 0 0 0 8 5.5z"/></svg>'
                f"Play preview</button>"
            )
        else:
            action = (
                '<span class="unavailable">Preview unavailable</span>'
                f'<a class="listen" href="{html.escape(item["external_url"])}" '
                f'target="_blank" rel="noopener">Listen on Apple Music</a>'
            )

        cards.append(
            f"""
            <div class="card" id="card-{item['rank']}">
              {artwork}
              <div class="body">
                <div class="title">{title}</div>
                <div class="artist">{artist}</div>
                <div class="genres">{genres}</div>
                <div class="explain">{explanation}</div>
              </div>
              <div class="right">
                <div class="score">{score}</div>
                <div class="score-label">{html.escape(score_label)}</div>
                {action}
              </div>
            </div>
            """
        )

    tracks_json = json.dumps(payload).replace("</", "<\\/")
    disabled = "" if ready else "disabled"

    body = f"""
<div class="mini" id="mini">
  <div class="mini-row">
    <div class="mini-now" id="mini-open" role="button" tabindex="0" aria-label="Open the Now Playing view">
      <div class="mini-label">Now playing</div>
      <div class="mini-title" id="mini-title">Nothing playing</div>
      <div class="mini-artist" id="mini-artist">{ready} of {len(payload)} previews available · {preview_seconds}s each</div>
    </div>
    <div class="mini-controls">
      <button class="ctrl" id="mini-toggle" {disabled} aria-label="Play or pause"></button>
      <button class="ctrl" id="mini-replay" {disabled} aria-label="Replay preview from the beginning">{ICON_REPLAY}</button>
      <button class="ctrl" id="mini-next" {disabled} aria-label="Play the next preview">{ICON_NEXT}</button>
    </div>
  </div>
</div>

<div class="np" id="np" hidden role="dialog" aria-modal="true" aria-label="Now Playing">
  <div class="np-top">
    <button class="np-back" id="np-back" aria-label="Back to recommendations">{ICON_BACK}</button>
    <div class="np-label">Now Playing</div>
  </div>
  <div class="np-center">
    <div class="np-art-wrap">
      <div class="np-ring"></div>
      <img class="np-art" id="np-art" alt="" src=""/>
      <div class="np-art fallback" id="np-fallback">♪</div>
    </div>
    <div class="np-title" id="np-title">Nothing playing</div>
    <div class="np-artist" id="np-artist">Choose a song to preview</div>
    <div class="np-genres" id="np-genres"></div>
    <div class="np-progress-block">
      <div class="np-times"><span id="np-elapsed">0:00</span><span id="np-remaining">-0:30</span></div>
      <div class="np-bar"><span id="np-progress"></span></div>
    </div>
    <div class="np-controls">
      <button class="ctrl" id="np-replay" aria-label="Replay preview from the beginning">{ICON_REPLAY}</button>
      <button class="ctrl primary" id="np-toggle" aria-label="Play preview"></button>
      <button class="ctrl" id="np-next" aria-label="Play the next preview">{ICON_NEXT}</button>
    </div>
    <div class="np-status" id="np-status">Click “Play preview” on any song.</div>
  </div>
</div>

{''.join(cards)}
"""

    return (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        + PLAYER_STYLE
        + "</head><body>"
        + body
        + PLAYER_SCRIPT.replace("__TRACKS__", tracks_json)
        .replace("__LIMIT__", str(preview_seconds))
        .replace("__ICON_PLAY__", ICON_PLAY.replace("'", "\\'"))
        .replace("__ICON_PAUSE__", ICON_PAUSE.replace("'", "\\'"))
        + "</body></html>"
    )


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading the music catalogue…")
def load_engine() -> Recommender:
    return Recommender(build_feature_matrix())


@st.cache_data(show_spinner=False)
def cached_preview(title: str, artists: str):
    return find_preview(title, artists)


# --- Arabic query candidates ------------------------------------------------ #
# MusicBrainz writes bilingual labels as "Arabic = Latin" or "Arabic (Latin)" or
# "Arabic - Translation". Sending that raw string to Apple returns nothing, so we
# build a small ordered set of query candidates. Verification is NOT touched: every
# candidate still goes through previews.find_preview and its strict checks.

_ANNOTATION_SEP = re.compile(r"\s*=\s*")
_BRACKETED = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
_HAS_LATIN = re.compile(r"[A-Za-z]")
_HAS_ARABIC = re.compile(r"[\u0600-\u06FF]")
_MAX_TITLE_CANDIDATES = 3
_MAX_QUERY_PAIRS = 12


def clean_label(text: object) -> str:
    """Strip MusicBrainz annotation tails and bracketed glosses.

    ``"ميال = Mayyal"`` → ``"ميال"``; ``"قمرين (Amarain)"`` → ``"قمرين"``;
    a trailing ``" - Translation"`` is dropped only when the head is Arabic and the
    tail has no Arabic (a translation, not part of the title).
    """
    value = str(text or "").strip()
    head = _ANNOTATION_SEP.split(value)[0]
    head = _BRACKETED.sub(" ", head)

    if " - " in head:
        base, tail = head.split(" - ", 1)
        if _HAS_ARABIC.search(base) and not _HAS_ARABIC.search(tail):
            head = base

    return " ".join(head.split()).strip(" -–—")


def label_candidates(text: object, limit: int = _MAX_TITLE_CANDIDATES) -> list[str]:
    """Ordered, de-duplicated search variants for a MusicBrainz label.

    Order: the original value, its cleaned form, then useful Latin/Arabic variants
    taken from a ``=`` annotation, a bracketed transliteration, or a trailing
    translation. Deterministic and duplicate-free.
    """
    raw = str(text or "").strip()
    candidates: list[str] = []

    def add(value: str) -> None:
        value = " ".join(str(value or "").split()).strip(" -–—")
        if value and value not in candidates:
            candidates.append(value)

    add(raw)
    add(clean_label(raw))

    parts = [part for part in _ANNOTATION_SEP.split(raw) if part.strip()]
    for part in parts[1:]:
        add(clean_label(part))

    for match in _BRACKETED.finditer(raw):
        inner = match.group(1).strip()
        if _HAS_LATIN.search(inner) or _HAS_ARABIC.search(inner):
            add(clean_label(inner))

    return candidates[:limit]


def preview_query_candidates(
    title: object,
    artist: object,
    artist_latin: object = "",
    aliases: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """Ordered, de-duplicated ``(title, artist)`` search pairs for an Arabic track.

    Order: the original credited pair first (so clean catalogue rows keep their
    current behaviour), then cleaned variants, then the Latin artist spelling,
    then MusicBrainz aliases. Aliases are additional *spellings* only — each pair
    is still verified by ``previews.find_preview``.
    """
    titles = label_candidates(title)
    artists = label_candidates(artist)
    for value in (str(artist_latin or "").strip(),):
        if value and value not in artists:
            artists.append(value)
    for alias in aliases:
        cleaned = clean_label(alias)
        if cleaned and cleaned not in artists:
            artists.append(cleaned)

    pairs: list[tuple[str, str]] = []
    for artist_value in artists:
        for title_value in titles:
            pair = (title_value, artist_value)
            if pair not in pairs:
                pairs.append(pair)
    return pairs[:_MAX_QUERY_PAIRS]


@st.cache_data(show_spinner=False)
def cached_aliases() -> dict:
    """Artist aliases per MBID (loaded once; empty when not built)."""
    try:
        return load_artist_aliases()
    except Exception:  # noqa: BLE001
        return {}


# Preview cache policy (Phase 12.3): a *verified* preview is cached for the
# session, but a failure is remembered only for a short cooldown. A transient
# Apple outage therefore cannot turn into a permanent "preview unavailable".
_PREVIEW_SUCCESS_CACHE: dict[tuple, Any] = {}
_PREVIEW_FAILED_AT: dict[tuple, float] = {}
NEGATIVE_PREVIEW_TTL_SECONDS = 300


def clear_arabic_preview_cache() -> None:
    """Drop cached successes and failure timestamps (used by tests)."""
    _PREVIEW_SUCCESS_CACHE.clear()
    _PREVIEW_FAILED_AT.clear()


def _preview_cache_key(title: str, artist: str, artist_latin: str, aliases: tuple) -> tuple:
    return (title, artist, artist_latin, tuple(aliases))


def cached_arabic_preview(
    title: str, artist: str, artist_latin: str, aliases: tuple[str, ...] = ()
):
    """First verified preview across the ordered Arabic query candidates.

    Caching policy:

    * **Success** — cached for the process lifetime (no repeat Apple request).
    * **Failure** — cached only for ``NEGATIVE_PREVIEW_TTL_SECONDS`` so a temporary
      network/API error cannot stick forever. After the cooldown the resolver's
      own negative entries are dropped before the retry, and the request is
      repeated at most once per cooldown window per track.

    Each candidate is still resolved by the unchanged ``previews.find_preview``,
    so strict artist/title verification and the preview-URL requirement apply.
    Returns ``None`` when no candidate verifies — the UI then shows the honest
    "Listen in Apple Music" fallback.
    """
    if previews_disabled():
        return None

    key = _preview_cache_key(title, artist, artist_latin, aliases)
    cached = _PREVIEW_SUCCESS_CACHE.get(key)
    if cached is not None:
        return cached

    now = time.monotonic()
    failed_at = _PREVIEW_FAILED_AT.get(key)
    if failed_at is not None and now - failed_at < NEGATIVE_PREVIEW_TTL_SECONDS:
        return None
    if failed_at is not None:
        # Cooldown elapsed: clear the resolver's cached negative result so the
        # retry is real rather than an instant re-read of the same failure.
        clear = getattr(find_preview, "cache_clear", None)
        if callable(clear):
            clear()
        _PREVIEW_FAILED_AT.pop(key, None)

    for candidate_title, candidate_artist in preview_query_candidates(
        title, artist, artist_latin, aliases
    ):
        preview = find_preview(candidate_title, candidate_artist)
        if preview is not None:
            _PREVIEW_SUCCESS_CACHE[key] = preview
            return preview

    _PREVIEW_FAILED_AT[key] = now
    return None


cached_arabic_preview.clear = clear_arabic_preview_cache  # type: ignore[attr-defined]


def arabic_previews_for(
    results: pd.DataFrame, catalogue: "ArabicCatalogue"
) -> dict[str, Any]:
    """Resolve previews for recommendation rows using each track's full catalogue row.

    ``recommend()`` intentionally returns display/scoring columns only, so the
    resolver must look the complete row up by ``track_id`` to obtain
    ``artist_latin`` and ``mb_artist_id``. Unknown ids are skipped, which leaves
    that card on the honest fallback instead of raising.
    """
    previews: dict[str, Any] = {}
    for track_id in results["track_id"]:
        key = str(track_id)
        try:
            full_row = catalogue.get(key)
        except KeyError:
            continue
        previews[key] = arabic_preview(full_row)
    return previews


def arabic_preview(row: pd.Series):
    """Preview for an Arabic row, using cleaned/Latin/alias query candidates."""
    mbid = str(row.get("mb_artist_id", "") or "")
    aliases = tuple(cached_aliases().get(mbid, ())) if mbid else ()
    return cached_arabic_preview(
        str(row["title"]),
        str(row["artist"]),
        str(row.get("artist_latin", "") or ""),
        aliases,
    )


@st.cache_data(show_spinner=False)
def cached_previews(items: tuple[tuple[str, str, str], ...]) -> dict:
    return resolve_many(list(items))


@st.cache_resource(show_spinner="Loading the Arabic catalogue…")
def load_arabic() -> ArabicCatalogue:
    return load_arabic_catalogue()


# --------------------------------------------------------------------------- #
# Page chrome
# --------------------------------------------------------------------------- #


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #090611; --panel: #120A22; --panel-2: #170D2B;
          --violet: #963BFF; --glow: #B66BFF;
          --text: #F5F0FF; --muted: #B9A8D9; --faint: #8E7FAE;
          --line: rgba(150,59,255,0.20);
        }
        .stApp, [data-testid="stAppViewContainer"] { background: var(--bg); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
        .block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 920px; }
        html, body, [class*="css"] {
          font-family: -apple-system, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif;
          color: var(--text);
        }
        h1, h2, h3, h4 { color: #FFFFFF; letter-spacing: -0.02em; }
        .app-header { margin-bottom: 26px; }
        .app-title { font-size: 34px; font-weight: 700; margin: 0 0 6px 0; line-height: 1.15; }
        .app-sub { color: var(--muted); font-size: 15px; margin: 0; }
        .step-label {
          font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
          color: var(--glow); font-weight: 650; margin-bottom: 4px;
        }
        .step-title { font-size: 18px; font-weight: 650; margin: 0 0 2px 0; color: #FFFFFF; }
        .step-help { color: var(--muted); font-size: 13px; margin: 0 0 10px 0; }
        .section { margin: 30px 0 6px 0; }
        .selected {
          display: flex; gap: 16px; align-items: center;
          background: linear-gradient(160deg, var(--panel-2), #0D0818);
          border: 1px solid var(--line); border-radius: 16px; padding: 16px;
          box-shadow: 0 12px 34px rgba(0,0,0,0.45), 0 0 0 1px rgba(182,107,255,0.05) inset;
        }
        .selected img { width: 86px; height: 86px; border-radius: 12px; object-fit: cover; }
        .selected .ph {
          width: 86px; height: 86px; border-radius: 12px;
          background: radial-gradient(circle at 30% 25%, rgba(182,107,255,0.35), rgba(23,13,43,0.9));
          display: flex; align-items: center; justify-content: center;
          color: #C79BFF; font-size: 26px;
        }
        .selected .title { font-size: 19px; font-weight: 700; line-height: 1.25; color: #FFFFFF; }
        .selected .artist { color: var(--muted); font-size: 13.5px; margin-top: 2px; }
        .selected .genres { color: #C79BFF; font-size: 12.5px; margin-top: 7px; }
        .hint { color: var(--faint); font-size: 12.5px; margin-top: 8px; }
        .note {
          background: rgba(150,59,255,0.08); border: 1px solid var(--line); border-radius: 12px;
          color: #CBB6EC; font-size: 12.5px; line-height: 1.55; padding: 11px 14px; margin: 6px 0 0 0;
        }
        div.stButton > button {
          background: linear-gradient(145deg, #B66BFF, #963BFF); color: #140A24; border: 0;
          border-radius: 12px; padding: 0.62rem 1.4rem; font-weight: 700; font-size: 15px;
          box-shadow: 0 8px 26px rgba(150,59,255,0.4);
          transition: transform .14s ease, box-shadow .14s ease;
        }
        div.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 10px 32px rgba(182,107,255,0.55); color: #140A24; }
        div.stButton > button:focus { box-shadow: 0 0 0 3px rgba(150,59,255,0.35); }
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
          background: #120A22 !important; border: 1px solid rgba(150,59,255,0.28) !important;
          border-radius: 11px !important; color: var(--text) !important;
        }
        [data-testid="stTextInput"] input::placeholder { color: #7E6FA3 !important; }
        label, .stMarkdown p { color: var(--text); }
        [data-testid="stCaptionContainer"] { color: var(--muted); }
        hr { border-color: var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header() -> None:
    st.markdown(
        f"""
        <div class="app-header">
          <h1 class="app-title">{APP_TITLE}</h1>
          <p class="app-sub">{SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(number: str, title: str, help_text: str = "") -> None:
    st.markdown(
        f'<div class="section"><div class="step-label">{number}</div>'
        f'<div class="step-title">{html.escape(title)}</div>'
        f'<p class="step-help">{html.escape(help_text)}</p></div>',
        unsafe_allow_html=True,
    )


def selected_song_panel(row: pd.Series, preview) -> None:
    """Section C: title, artists, genres, and verified artwork when available."""
    artwork = (
        f'<img src="{html.escape(preview.artwork_url)}" alt=""/>'
        if preview is not None and preview.artwork_url
        else '<div class="ph">♪</div>'
    )
    genres = genres_display(row["genre_set"])
    availability = (
        f'<div class="hint">30-second preview available · {html.escape(preview.provider)}</div>'
        if preview is not None
        else '<div class="hint">Preview unavailable for this track.</div>'
    )
    st.markdown(
        f"""
        <div class="selected">
          {artwork}
          <div>
            <div class="title">{html.escape(str(row['track_name']))}</div>
            <div class="artist">{html.escape(str(row['artists']))}</div>
            <div class="genres">{html.escape(genres)}</div>
            {availability}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_results(results: pd.DataFrame, preferences: Preferences) -> None:
    """Section G: cards + Now Playing experience."""
    if results.empty:
        st.info("No recommendations are available for this track. Try another song.")
        return

    if not preferences.is_empty() and float(results["preference_score"].max()) == 0.0:
        st.info(
            "None of these results matched your preferences, so they were ranked on audio "
            "and genre similarity only. Try a different genre or artist."
        )

    items = tuple(
        (str(row["track_id"]), str(row["track_name"]), str(row["artists"]))
        for _, row in results.iterrows()
    )
    if previews_disabled():
        previews_by_id: dict[str, Any] = {}
    else:
        with st.spinner("Finding 30-second previews…"):
            previews_by_id = cached_previews(items)

    payload = build_result_payload(results, previews_by_id)
    components.html(
        render_results_html(payload),
        height=results_height(len(payload)),
        scrolling=False,
    )
    st.caption(SCORE_INTERPRETATION_WARNING)


def show_more_by_artist(engine: Recommender, track_id: str, artist: str, k: int = 5) -> None:
    """Extra list: the preferred artist's own songs, closest in sound to the reference.

    Shown below the main results so the main ranking (and its evaluation) stays
    unchanged; it answers "I asked for this artist, where are their songs?".
    """
    try:
        extra = engine.more_by_artist(track_id, artist, k=k)
    except Exception:  # noqa: BLE001
        return
    if extra.empty:
        st.info(f'No other songs by "{artist}" are in the catalogue.')
        return

    st.markdown(f"#### More by {html.escape(artist.title())} — closest in sound to your song")
    items = tuple(
        (str(row["track_id"]), str(row["track_name"]), str(row["artists"]))
        for _, row in extra.iterrows()
    )
    if previews_disabled():
        previews_by_id: dict[str, Any] = {}
    else:
        with st.spinner("Finding 30-second previews…"):
            previews_by_id = cached_previews(items)
    payload = build_result_payload(extra, previews_by_id)
    components.html(
        render_results_html(payload),
        height=results_height(len(payload)),
        scrolling=False,
    )


# --------------------------------------------------------------------------- #
# Arabic metadata mode
# --------------------------------------------------------------------------- #


def page_arabic() -> None:
    """Metadata-based Arabic discovery (no audio features, no audio scores)."""
    try:
        catalogue = load_arabic()
    except ArabicCatalogueError as exc:
        st.info(str(exc))
        return

    stats = catalogue.stats()
    prov = arabic_provenance()
    st.markdown(
        f'<div class="note">{html.escape(METADATA_SCORE_NOTE)}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{stats.tracks:,} Arabic tracks · {stats.artists:,} artists · "
        f"{stats.genres} curated genres · {stats.regions} regions · "
        f"metadata from MusicBrainz ({prov.get('licence', 'CC0')})"
    )

    # Search ------------------------------------------------------------- #
    section_label(
        "1 · Search",
        "ابحث عن أغنية أو فنان / Find an Arabic song",
        "Arabic or English spelling both work — try فيروز, أم كلثوم, Amr Diab.",
    )
    query = st.text_input(
        "Arabic song or artist",
        placeholder="فيروز، أم كلثوم، كاظم الساهر، Fairuz…",
        label_visibility="collapsed",
        key="arabic_query",
    )
    matches = catalogue.search(query)
    if matches.empty:
        st.warning(f'لا توجد نتائج لـ "{query}" — no match. Try a shorter phrase or another spelling.')
        return

    st.caption(f"{len(matches)} matching track(s)")
    labels = [
        [f"{row['title']} — {row['artist']}" for _, row in matches.iterrows()][i]
        for i in range(len(matches))
    ]
    choice = st.selectbox(
        "Matching Arabic songs",
        options=list(range(len(matches))),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
        key="arabic_select",
    )
    reference = matches.iloc[choice]
    track_id = str(reference["track_id"])

    # Selected song ------------------------------------------------------ #
    section_label("2 · Selected song", "المرجع / Your reference track")
    try:
        preview = None if previews_disabled() else arabic_preview(reference)
    except Exception:  # noqa: BLE001
        preview = None

    artwork = (
        f'<img src="{html.escape(preview.artwork_url)}" alt="" dir="ltr"/>'
        if preview is not None and preview.artwork_url
        else '<div class="ph">♪</div>'
    )
    availability = (
        f'<div class="hint">30-second preview available · {html.escape(preview.provider)}</div>'
        if preview is not None
        else '<div class="hint">Preview unavailable for this track.</div>'
    )
    st.markdown(
        f"""
        <div class="selected">
          {artwork}
          <div dir="rtl" lang="ar" style="text-align:right">
            <div class="title">{html.escape(str(reference['title']))}</div>
            <div class="artist" dir="auto">{html.escape(str(reference['artist']))}</div>
            <div class="genres">{html.escape(str(reference['genre']))} · {html.escape(str(reference['region']))}</div>
            <div class="artist" dir="auto" style="font-size:12.5px;margin-top:6px">ألبوم: {html.escape(str(reference['album']))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if preview is not None:
        st.caption(f"Verified preview match: {preview.matched_title} — {preview.matched_artist}")
    if preview is not None and getattr(preview, "url", ""):
        # Same as the main page: let the user hear the song they picked.
        st.caption("استمع إلى الأغنية / Listen to your reference song (30-second Apple preview):")
        st.audio(preview.url)

    # Filters ------------------------------------------------------------ #
    section_label(
        "3 · Filters",
        "Optional genre filter",
        "Leave empty to search the whole Arabic catalogue.",
    )
    selected_genres = st.multiselect(
        "Arabic genres",
        options=list(catalogue.genres()),
        help="Restricts candidates to this curated genre before matching.",
    )
    k = st.radio("How many recommendations?", options=[5, 10], index=0, horizontal=True, key="arabic_k")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    clicked = st.button("Find Similar Songs", type="primary", key="arabic_button")

    if clicked:
        try:
            results = catalogue.recommend(
                track_id, limit=int(k), genres=tuple(selected_genres)
            )
        except KeyError:
            st.error("That track is no longer in the Arabic catalogue. Please search again.")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while matching Arabic metadata: {exc}")
            return
        st.session_state["arabic_results"] = results
        st.session_state["arabic_reference"] = track_id

    results = st.session_state.get("arabic_results")
    if results is None or results.empty:
        if results is not None:
            st.info(
                "No other track shares this track's artist, genre or region with the current "
                "filter. Try clearing the genre filter."
            )
        return

    section_label("4 · Results", "أغانٍ ذات صلة / Related Arabic songs")
    st.caption(
        "Ranked by metadata match — same artist (0.60), same genre (0.30), same region (0.10). "
        "This is not an audio similarity score."
    )

    items = tuple(
        (str(row["track_id"]), str(row["title"]), str(row["artist"]))
        for _, row in results.iterrows()
    )
    if previews_disabled():
        previews_by_id: dict[str, Any] = {}
    else:
        with st.spinner("Finding 30-second previews…"):
            previews_by_id = arabic_previews_for(results, catalogue)

    payload = build_arabic_payload(results, previews_by_id)
    components.html(
        render_results_html(payload, score_label="metadata match"),
        height=results_height(len(payload)),
        scrolling=False,
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def run_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎵", layout="centered")
    inject_styles()
    header()

    mode = st.radio(
        "Discovery mode",
        options=["International Music", "Arabic Music"],
        index=0,
        horizontal=True,
        help="International mode ranks by audio similarity. Arabic mode ranks by metadata only.",
    )

    if mode == "Arabic Music":
        page_arabic()
        return

    if not DEFAULT_DATA_PATH.is_file():
        st.error("The music catalogue file is missing, so no recommendations can be made.")
        st.markdown(f"Download it first:\n\n```\n{DOWNLOAD_HINT}\n```")
        return

    try:
        engine = load_engine()
    except FileNotFoundError:
        st.error("The dataset could not be found. See the download instructions above.")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"The catalogue could not be prepared: {exc}")
        return

    catalogue = engine._catalogue

    # B. Song search ------------------------------------------------------- #
    section_label("1 · Search", "Find a song", "Search by song title or artist. Nothing here requires an account.")
    query = st.text_input(
        "Song title or artist",
        placeholder="e.g. Cold Heart, Elton John, Nirvana…",
        label_visibility="collapsed",
    )
    matches = search_tracks(catalogue, query)

    if matches.empty:
        st.warning(f'No song matches "{query}". Try a shorter phrase or check the spelling.')
        return

    st.caption(f"{len(matches)} matching track(s) — pick one as your reference.")
    labels = [track_label(row["track_name"], row["artists"]) for _, row in matches.iterrows()]
    choice = st.selectbox(
        "Matching songs",
        options=list(range(len(matches))),
        format_func=lambda i: labels[i],
        label_visibility="collapsed",
    )
    reference = matches.iloc[choice]
    track_id = str(reference[ID_COLUMN])

    # C. Selected song ----------------------------------------------------- #
    section_label("2 · Selected song", "Your reference track")
    try:
        selected_preview = (
            None
            if previews_disabled()
            else cached_preview(str(reference["track_name"]), str(reference["artists"]))
        )
    except Exception:  # noqa: BLE001
        selected_preview = None
    selected_song_panel(reference, selected_preview)
    if selected_preview is not None and getattr(selected_preview, "url", ""):
        # Let the user listen to the song they searched for, not only to the results.
        st.caption("Listen to your reference song (30-second Apple preview):")
        st.audio(selected_preview.url)

    # D. Optional preferences ---------------------------------------------- #
    section_label(
        "3 · Preferences",
        "Refine the results (optional)",
        "Preferences are optional — you can press “Find Similar Songs” without choosing anything.",
    )
    pref_col_left, pref_col_right = st.columns([1.4, 1], gap="large")
    with pref_col_left:
        preferred_genres = st.multiselect(
            "Preferred genres",
            options=list(engine.genres),
            help="Tracks in these genres get a bounded boost (20% of the final score).",
        )
        preferred_artist = st.text_input(
            "Preferred artist",
            placeholder="e.g. Elton John",
            help="Matched against the credited artists; leave empty to ignore.",
        )
    with pref_col_right:
        st.markdown('<div class="step-label">4 · Results</div>', unsafe_allow_html=True)
        k = st.radio("How many recommendations?", options=[5, 10], index=0, horizontal=True)

    st.markdown(
        '<div class="note">No preferences? That is fine — recommendations are then ranked on '
        "audio and genre similarity alone.</div>",
        unsafe_allow_html=True,
    )

    preferences = Preferences.build(
        genres=preferred_genres,
        artists=[preferred_artist] if preferred_artist.strip() else [],
    )

    # F. Recommendation button --------------------------------------------- #
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    clicked = st.button("Find Similar Songs", type="primary")

    if clicked:
        try:
            with st.spinner("Comparing feature vectors…"):
                results = engine.recommend(track_id, k=int(k), preferences=preferences)
        except UnknownTrackError:
            st.error("That track is not in the recommendation catalogue. Please search again.")
            return
        except InvalidRequestError as exc:
            st.error(f"That request could not be processed: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while building recommendations: {exc}")
            return

        st.session_state["results"] = results
        st.session_state["results_preferences"] = preferences
        st.session_state["results_track"] = track_id

    # G. Results ------------------------------------------------------------ #
    results = st.session_state.get("results")
    # Results belong to the song (and preferences) they were computed for. If the
    # user has since picked another song or changed preferences, do not show the
    # old list under the new song -- ask them to press the button again.
    if results is not None and (
        st.session_state.get("results_track") != track_id
        or st.session_state.get("results_preferences") != preferences
    ):
        st.info("You changed the song or your preferences. Press **Find Similar Songs** to update the results.")
        results = None
    if results is not None:
        section_label("5 · Results", "Songs similar to your reference")
        show_results(results, st.session_state.get("results_preferences", Preferences()))
        result_preferences = st.session_state.get("results_preferences", Preferences())
        if result_preferences.artists:
            show_more_by_artist(engine, track_id, result_preferences.artists[0])


def main() -> None:  # pragma: no cover - Streamlit entry point
    run_app()


if __name__ == "__main__":
    main()
