"""Evaluation procedures for the content-based recommender (Phase 6).

What this module measures — and what it deliberately does not.

The dataset contains **no user ratings, play counts or relevance labels**, so
accuracy, precision, recall and NDCG **cannot be computed honestly** and are not
computed anywhere in this project. Instead this module measures structural and
descriptive properties that are fully computable from the data:

* **intra-list similarity (ILS)** — how alike the returned tracks are to each
  other, per block and combined;
* **diversity** — distinct genres and artists inside a list;
* **genre consistency** — share of results sharing at least one genre with the
  reference;
* **catalogue coverage** — share of the catalogue the system can reach across many
  reference tracks;
* **duplicate suppression** — full-signature repeats (must be 0) and look-alike
  title+artist repeats (allowed by the approved rule);
* **self-match exclusion** — the reference must never appear;
* **preference influence** — how the list changes when preferences are supplied.

These are descriptive measurements of behaviour, not claims about user
satisfaction. A high ILS is not "better" or "worse" on its own: it trades breadth
for tightness, and the report discusses that trade-off rather than scoring it.

Reproducible: reference tracks are chosen deterministically, `k` is fixed, and
nothing is random. Run:

    .venv\\Scripts\\python.exe evaluation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import FeatureMatrix, build_feature_matrix
from recommender import Preferences, Recommender

PROJECT_ROOT = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "report"
RESULTS_CSV = REPORT_DIR / "evaluation_results.csv"
RESULTS_MD = REPORT_DIR / "evaluation_results.md"

DEFAULT_K = 10

#: Genres used to build the reference set (all verified to exist in the catalogue).
REFERENCE_GENRES = (
    "rock",
    "hip-hop",
    "classical",
    "techno",
    "sleep",
    "jazz",
    "acoustic",
    "heavy-metal",
)
TRACKS_PER_GENRE = 2
MULTI_GENRE_REFERENCES = 3


@dataclass(frozen=True)
class ReferenceTrack:
    """A reference track chosen for evaluation."""

    label: str
    track_id: str
    genres: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Reference selection
# --------------------------------------------------------------------------- #


def select_reference_tracks(
    feature_matrix: FeatureMatrix,
    k_per_genre: int = TRACKS_PER_GENRE,
    multi_genre_count: int = MULTI_GENRE_REFERENCES,
) -> list[ReferenceTrack]:
    """Deterministically choose references spanning genres and multi-genre tracks.

    Within each genre the most popular tracks are taken (ties broken by
    ``track_id``); multi-genre references are the tracks with the largest genre
    sets, again ordered by popularity then ``track_id``. No randomness is used.
    """
    catalogue = feature_matrix.catalogue
    chosen: list[ReferenceTrack] = []
    used: set[str] = set()

    for genre in REFERENCE_GENRES:
        mask = catalogue["genre_set"].map(lambda g: genre in g)
        subset = catalogue.loc[mask, ["popularity", "track_id"]].reset_index(drop=True)
        subset = subset.sort_values(["popularity", "track_id"], ascending=[False, True])
        for track_id in subset["track_id"].tolist():
            if track_id in used:
                continue
            used.add(track_id)
            chosen.append(
                ReferenceTrack(
                    label=f"genre:{genre}",
                    track_id=str(track_id),
                    genres=tuple(feature_matrix.catalogue.loc[track_id, "genre_set"]),
                )
            )
            break
        # a second track from the same genre, for a within-genre comparison
        for track_id in subset["track_id"].tolist()[1:]:
            if track_id in used:
                continue
            used.add(track_id)
            label = f"genre:{genre} (2nd)"
            chosen.append(
                ReferenceTrack(
                    label=label,
                    track_id=str(track_id),
                    genres=tuple(feature_matrix.catalogue.loc[track_id, "genre_set"]),
                )
            )
            break

    sizes = catalogue["genre_set"].map(len)
    multi = catalogue.loc[sizes == sizes.max(), ["popularity", "track_id"]].reset_index(drop=True)
    multi = multi.sort_values(["popularity", "track_id"], ascending=[False, True])
    for track_id in multi["track_id"].tolist()[:multi_genre_count]:
        if track_id in used:
            continue
        used.add(track_id)
        chosen.append(
            ReferenceTrack(
                label="multi-genre",
                track_id=str(track_id),
                genres=tuple(feature_matrix.catalogue.loc[track_id, "genre_set"]),
            )
        )

    return chosen


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def intra_list_similarity(block: np.ndarray) -> float:
    """Mean pairwise cosine similarity inside a list of unit-norm rows.

    ``block`` must contain one row per result. Returns 0.0 for fewer than two
    rows. Because rows are unit-norm, the pairwise cosines are the Gram matrix,
    and the diagonal (self-similarity 1.0) is excluded.
    """
    rows = np.asarray(block, dtype=float)
    n = rows.shape[0]
    if n < 2:
        return 0.0
    gram = rows @ rows.T
    total = float(gram.sum() - np.trace(gram))
    return total / (n * (n - 1))


def genre_diversity(genre_sets: list[tuple[str, ...]]) -> int:
    """Number of distinct genres appearing across a result list."""
    return len({genre for genres in genre_sets for genre in genres})


def genre_consistency(reference_genres: tuple[str, ...], genre_sets: list[tuple[str, ...]]) -> float:
    """Share of results sharing at least one genre with the reference."""
    if not genre_sets:
        return 0.0
    reference = set(reference_genres)
    hits = sum(1 for genres in genre_sets if reference & set(genres))
    return hits / len(genre_sets)


def artist_diversity(artists: list[str]) -> float:
    """Distinct credited artists divided by the number of results."""
    if not artists:
        return 0.0
    return len(set(artists)) / len(artists)


def lookalike_repeats(results: pd.DataFrame) -> int:
    """How many results repeat an earlier title+artist pair (case-insensitive).

    These are *allowed* under the approved suppression rule when audio features
    differ; the count is reported so the behaviour is visible rather than hidden.
    """
    keys = [
        (str(title).strip().casefold(), str(artist).strip().casefold())
        for title, artist in zip(results["track_name"], results["artists"], strict=True)
    ]
    return len(keys) - len(set(keys))


def full_signature_repeats(results: pd.DataFrame, engine: Recommender) -> int:
    """How many results repeat an earlier full recording signature (must be 0)."""
    keys = [engine._duplicate_keys[engine.track_row(t)] for t in results["track_id"]]
    return len(keys) - len(set(keys))


def evaluate_reference(
    engine: Recommender,
    reference: ReferenceTrack,
    k: int = DEFAULT_K,
    preferences: Preferences | None = None,
) -> dict:
    """Compute every metric for one reference track."""
    results = engine.recommend(reference.track_id, k=k, preferences=preferences)
    if results.empty:
        return {
            "label": reference.label,
            "track_id": reference.track_id,
            "genres": ", ".join(reference.genres),
            "n_results": 0,
            "distinct_genres": 0,
            "genre_consistency": 0.0,
            "artist_diversity": 0.0,
            "ils_numeric": 0.0,
            "ils_genre": 0.0,
            "ils_combined": 0.0,
            "mean_score": float("nan"),
            "lookalike_repeats": 0,
            "full_signature_repeats": 0,
            "self_present": False,
        }

    rows = [engine.track_row(t) for t in results["track_id"]]
    numeric_in_list = np.asarray(engine._numeric[rows].todense(), dtype=float)
    genre_in_list = np.asarray(engine._genre[rows].todense(), dtype=float)

    return {
        "label": reference.label,
        "track_id": reference.track_id,
        "genres": ", ".join(reference.genres),
        "n_results": len(results),
        "distinct_genres": genre_diversity(list(results["genres"].str.split(", "))),
        "genre_consistency": genre_consistency(reference.genres, list(results["genres"].str.split(", "))),
        "artist_diversity": artist_diversity(list(results["artists"])),
        "ils_numeric": intra_list_similarity(numeric_in_list),
        "ils_genre": intra_list_similarity(genre_in_list),
        "ils_combined": float(
            engine.numeric_weight * intra_list_similarity(numeric_in_list)
            + engine.genre_weight * intra_list_similarity(genre_in_list)
        ),
        "mean_score": float(results["score"].mean()),
        "lookalike_repeats": lookalike_repeats(results),
        "full_signature_repeats": full_signature_repeats(results, engine),
        "self_present": reference.track_id in set(results["track_id"]),
    }


def evaluate_all(
    engine: Recommender,
    references: list[ReferenceTrack],
    k: int = DEFAULT_K,
    preferences: Preferences | None = None,
) -> pd.DataFrame:
    """Metrics for every reference, plus the list of returned track ids."""
    rows = []
    for reference in references:
        row = evaluate_reference(engine, reference, k=k, preferences=preferences)
        row["_returned"] = set(
            engine.recommend(reference.track_id, k=k, preferences=preferences)["track_id"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_reference: pd.DataFrame, catalogue_size: int) -> dict:
    """Aggregate metrics, including catalogue coverage over all lists."""
    returned: set[str] = set()
    for ids in per_reference["_returned"]:
        returned |= ids

    return {
        "references": len(per_reference),
        "mean_distinct_genres": float(per_reference["distinct_genres"].mean()),
        "mean_genre_consistency": float(per_reference["genre_consistency"].mean()),
        "mean_artist_diversity": float(per_reference["artist_diversity"].mean()),
        "mean_ils_numeric": float(per_reference["ils_numeric"].mean()),
        "mean_ils_genre": float(per_reference["ils_genre"].mean()),
        "mean_ils_combined": float(per_reference["ils_combined"].mean()),
        "mean_score": float(per_reference["mean_score"].mean()),
        "lookalike_repeats": int(per_reference["lookalike_repeats"].sum()),
        "full_signature_repeats": int(per_reference["full_signature_repeats"].sum()),
        "self_match_violations": int(per_reference["self_present"].sum()),
        "catalogue_coverage": len(returned) / catalogue_size,
        "catalogue_size": catalogue_size,
    }


# --------------------------------------------------------------------------- #
# Preference influence
# --------------------------------------------------------------------------- #


def preference_scenarios(engine: Recommender, references: list[ReferenceTrack]) -> list[dict]:
    """Deterministic preference settings, including an unmatched one."""
    aligned = references[0].genres[0]
    cross = "classical"
    return [
        {"name": "aligned genre", "preferences": Preferences.build(genres=[aligned])},
        {"name": "cross genre", "preferences": Preferences.build(genres=[cross])},
        {"name": "artist", "preferences": Preferences.build(artists=["Elton John"])},
    ]


def compare_preferences(
    engine: Recommender,
    references: list[ReferenceTrack],
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """Ranking change caused by each preference scenario, per reference."""
    rows = []
    for scenario in preference_scenarios(engine, references):
        preferences = scenario["preferences"]
        for reference in references:
            baseline = engine.recommend(reference.track_id, k=k)
            adjusted = engine.recommend(reference.track_id, k=k, preferences=preferences)

            baseline_ids = list(baseline["track_id"])
            adjusted_ids = list(adjusted["track_id"])
            shared = set(baseline_ids) & set(adjusted_ids)

            baseline_rank = {tid: i for i, tid in enumerate(baseline_ids)}
            adjusted_rank = {tid: i for i, tid in enumerate(adjusted_ids)}
            rank_shift = (
                float(np.mean([abs(baseline_rank[t] - adjusted_rank[t]) for t in shared]))
                if shared
                else float("nan")
            )

            rows.append(
                {
                    "scenario": scenario["name"],
                    "label": reference.label,
                    "reference_genres": ", ".join(reference.genres),
                    "changed_items": len(set(adjusted_ids) - set(baseline_ids)),
                    "shared_items": len(shared),
                    "mean_rank_shift": rank_shift,
                    "baseline_mean_score": float(baseline["score"].mean()),
                    "adjusted_mean_score": float(adjusted["score"].mean()),
                    "baseline_pref_match": float(
                        baseline["preference_score"].mean() if "preference_score" in baseline else 0.0
                    ),
                    "adjusted_pref_match": float(adjusted["preference_score"].mean()),
                    "top1_changed": bool(adjusted_ids[0] != baseline_ids[0]),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def build_report() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Run the full evaluation and return (per_reference, preference, summary,
    summary_pref, samples)."""
    feature_matrix = build_feature_matrix()
    engine = Recommender(feature_matrix)
    references = select_reference_tracks(feature_matrix)

    per_reference = evaluate_all(engine, references, k=DEFAULT_K)
    summary = summarise(per_reference, engine.n_tracks)

    preference_table = compare_preferences(engine, references, k=DEFAULT_K)
    preference_summary = {
        "scenarios": preference_table["scenario"].nunique(),
        "mean_changed_items": float(preference_table["changed_items"].mean()),
        "mean_rank_shift": float(preference_table["mean_rank_shift"].mean()),
        "top1_changed_share": float(preference_table["top1_changed"].mean()),
        "mean_adjusted_pref_match": float(preference_table["adjusted_pref_match"].mean()),
    }

    # raw material for the qualitative review (task 6.2)
    samples = {
        reference.label: engine.recommend(reference.track_id, k=5)
        for reference in references
    }
    return per_reference, preference_table, summary, preference_summary, samples


def write_report(
    per_reference: pd.DataFrame,
    preference_table: pd.DataFrame,
    summary: dict,
    preference_summary: dict,
    samples: dict,
) -> None:
    """Write the CSV and the markdown summary into ``report/``."""
    REPORT_DIR.mkdir(exist_ok=True)

    per_reference.drop(columns=["_returned"]).to_csv(RESULTS_CSV, index=False)

    lines: list[str] = []
    add = lines.append
    add("# Evaluation results")
    add("")
    add("Generated by `evaluation.py`. **Descriptive measurements only** — the dataset has no")
    add("relevance labels, so accuracy, precision, recall and NDCG are not computed and must not")
    add("be inferred from these numbers. ILS = intra-list similarity; a high value means the")
    add("returned tracks are alike, which is a trade-off, not a score to maximise.")
    add("")
    add(f"- references: **{summary['references']}** tracks across "
        f"{len(REFERENCE_GENRES)} genres plus multi-genre tracks")
    add(f"- k per list: **{DEFAULT_K}**")
    add(f"- catalogue: **{summary['catalogue_size']:,}** tracks")
    add("")
    add("## Aggregate")
    add("")
    add("| measure | value |")
    add("|---|---|")
    add(f"| mean distinct genres per list | {summary['mean_distinct_genres']:.2f} |")
    add(f"| mean genre consistency (share sharing a genre with the reference) | "
        f"{summary['mean_genre_consistency']:.3f} |")
    add(f"| mean artist diversity | {summary['mean_artist_diversity']:.3f} |")
    add(f"| mean ILS (numeric block) | {summary['mean_ils_numeric']:.4f} |")
    add(f"| mean ILS (genre block) | {summary['mean_ils_genre']:.4f} |")
    add(f"| mean ILS (combined) | {summary['mean_ils_combined']:.4f} |")
    add(f"| mean score | {summary['mean_score']:.4f} |")
    add(f"| full-signature duplicate repeats | {summary['full_signature_repeats']} |")
    add(f"| look-alike title+artist repeats (allowed) | {summary['lookalike_repeats']} |")
    add(f"| self-match violations | {summary['self_match_violations']} |")
    add(f"| catalogue coverage (union of all lists) | {summary['catalogue_coverage']:.4%} |")
    add("")
    add("## Per reference")
    add("")
    add("| reference | genres | distinct genres | genre consistency | artist diver. | "
        "ILS num | ILS genre | ILS comb | mean score | look-alike | self |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, row in per_reference.iterrows():
        add(f"| {row['label']} | {row['genres'][:34]} | {int(row['distinct_genres'])} | "
            f"{row['genre_consistency']:.2f} | {row['artist_diversity']:.2f} | "
            f"{row['ils_numeric']:.3f} | {row['ils_genre']:.3f} | {row['ils_combined']:.3f} | "
            f"{row['mean_score']:.3f} | {int(row['lookalike_repeats'])} | "
            f"{'YES' if row['self_present'] else 'no'} |")
    add("")
    add("## Preference influence")
    add("")
    add(f"- scenarios: {preference_summary['scenarios']} (aligned genre, cross genre, artist)")
    add(f"- mean items changed in a top-{DEFAULT_K} list: "
        f"**{preference_summary['mean_changed_items']:.2f}**")
    add(f"- mean rank shift of items that stayed: "
        f"**{preference_summary['mean_rank_shift']:.2f}** positions")
    add(f"- share of references whose #1 result changed: "
        f"**{preference_summary['top1_changed_share']:.1%}**")
    add(f"- mean preference match of the adjusted top-{DEFAULT_K}: "
        f"**{preference_summary['mean_adjusted_pref_match']:.3f}**")
    add("")
    add("Per-scenario detail:")
    add("")
    add("| scenario | mean changed items | mean rank shift | top-1 changed | mean pref match |")
    add("|---|---|---|---|---|")
    for scenario, group in preference_table.groupby("scenario", sort=False):
        add(f"| {scenario} | {group['changed_items'].mean():.2f} | "
            f"{group['mean_rank_shift'].mean():.2f} | "
            f"{group['top1_changed'].mean():.0%} | "
            f"{group['adjusted_pref_match'].mean():.3f} |")
    add("")
    add("## Sampled top-5 lists (qualitative review)")
    add("")
    add("Raw material for the qualitative inspection required by task 6.2. The judgement")
    add("column is deliberately left to the student: only a listener can say whether a")
    add("neighbour is musically sensible, and this project does not fabricate that.")
    add("")
    for label, results in samples.items():
        add(f"### {label}")
        add("")
        add("| # | track | artist | genres | score |")
        add("|---|---|---|---|---|")
        for _, row in results.iterrows():
            add(f"| {int(row['rank'])} | {row['track_name']} | {row['artists']} | "
                f"{row['genres']} | {row['score']:.3f} |")
        add("")

    add("## Limitations")
    add("")
    add("- No relevance labels exist: none of these numbers measures recommendation *quality*")
    add("  in the sense of user preference. They describe behaviour, not satisfaction.")
    add("- ILS and diversity pull in opposite directions; neither is inherently better.")
    add("- Catalogue coverage is computed over 19 reference tracks, not the whole catalogue.")
    add("- Look-alike repeats are permitted by the approved suppression rule (audio differs);")
    add("  they are reported, not hidden, and are not counted as violations.")
    add("- Preference influence is measured by rank movement, not by usefulness.")

    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    per_reference, preference_table, summary, preference_summary, samples = build_report()
    write_report(per_reference, preference_table, summary, preference_summary, samples)

    print(f"references evaluated: {summary['references']}")
    print(f"mean ILS numeric={summary['mean_ils_numeric']:.4f} "
          f"genre={summary['mean_ils_genre']:.4f} combined={summary['mean_ils_combined']:.4f}")
    print(f"mean genre consistency={summary['mean_genre_consistency']:.3f} "
          f"mean distinct genres={summary['mean_distinct_genres']:.2f}")
    print(f"coverage={summary['catalogue_coverage']:.4%} "
          f"full-signature repeats={summary['full_signature_repeats']} "
          f"look-alike repeats={summary['lookalike_repeats']} "
          f"self-match violations={summary['self_match_violations']}")
    print(f"preference: mean changed items={preference_summary['mean_changed_items']:.2f} "
          f"top1 changed={preference_summary['top1_changed_share']:.1%} "
          f"pref match={preference_summary['mean_adjusted_pref_match']:.3f}")
    print(f"written: {RESULTS_MD.relative_to(PROJECT_ROOT)}, {RESULTS_CSV.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
