"""Checkpoint C3 — sensitivity of recommendations to the numeric/genre block weights.

Reproducible, read-only analysis. It builds the approved preprocessing matrix
once and then compares:

    S_num    cosine similarity of the 9 numeric audio features
    S_genre  cosine similarity of the 114 multi-label genre indicators
    combined = alpha * S_num + beta * S_genre        (alpha + beta = 1)

for alpha in {0.5, 0.6, 0.7, 0.8, 0.9}.

Nothing here implements a recommender: there is no ranking API, no preference
handling, no score presentation. It exists to gather evidence for the C3 gate.

Run from the project root:

    .venv\\Scripts\\python.exe analysis\\weight_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import FeatureMatrix, NUMERIC_FEATURES, build_feature_matrix  # noqa: E402

ALPHAS = (0.50, 0.60, 0.70, 0.80, 0.90)
TOP_K = 10
GLOBAL_SAMPLE = 300
GLOBAL_SEED = 42
OUT_DOC = PROJECT_ROOT / "docs" / "weight-sensitivity.md"


# --------------------------------------------------------------------------- #
# Reference tracks
# --------------------------------------------------------------------------- #

def pick_reference_tracks(fm: FeatureMatrix) -> list[dict]:
    """Deterministic, documented selection: one popular track per genre profile.

    For each profile the most popular track wins; ties break on track_id, so the
    same six tracks are chosen on every run.
    """
    cat = fm.catalogue
    genres = cat["genre_set"]

    def most_popular(mask: pd.Series) -> str | None:
        # reset_index: the catalogue keeps track_id as both index and column.
        subset = cat.loc[mask, ["popularity", "track_id"]].reset_index(drop=True)
        if subset.empty:
            return None
        subset = subset.sort_values(["popularity", "track_id"], ascending=[False, True])
        return str(subset.iloc[0]["track_id"])

    picks: list[tuple[str, str | None]] = []

    picks.append(("single-genre rock (popular)", most_popular(genres == ("rock",))))
    picks.append(("single-genre hip-hop (popular)", most_popular(genres == ("hip-hop",))))
    picks.append(("single-genre classical", most_popular(genres == ("classical",))))
    picks.append(("single-genre techno", most_popular(genres == ("techno",))))

    multi = genres.map(len) >= 4
    picks.append(("multi-genre (>=4 genres)", most_popular(multi)))

    seven = genres.map(len) == genres.map(len).max()
    picks.append((f"most multi-genre ({int(genres.map(len).max())} genres)", most_popular(seven)))

    refs = []
    for label, track_id in picks:
        if track_id is None:
            continue
        row = fm.catalogue.loc[track_id]
        refs.append(
            {
                "label": label,
                "track_id": track_id,
                "track_name": row["track_name"],
                "artists": row["artists"],
                "genres": row["genre_set"],
                "popularity": int(row["popularity"]),
                "row": fm.row_index(track_id),
            }
        )
    return refs


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #

def block_similarities(fm: FeatureMatrix, row: int) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity of the reference row against every track, per block.

    Both blocks are row-normalised in the preprocessing pipeline, so a
    matrix-vector product is exactly the block cosine similarity.

    Note: ``FeatureMatrix.numeric_matrix`` is declared as ``csr_matrix`` but is
    actually a dense ndarray (sklearn's ``normalize`` returns dense for dense
    input). That declaration bug is reported to the student; here both blocks are
    converted explicitly so the analysis works either way.
    """
    numeric = sparse.csr_matrix(fm.numeric_matrix)
    genre = sparse.csr_matrix(fm.genre_matrix)

    ref_num = numeric.getrow(row)
    ref_genre = genre.getrow(row)
    s_num = (numeric @ ref_num.T).toarray().ravel()
    s_genre = (genre @ ref_genre.T).toarray().ravel()
    return s_num, s_genre


def rank(
    fm: FeatureMatrix,
    s_num: np.ndarray,
    s_genre: np.ndarray,
    alpha: float,
    exclude_row: int,
    k: int = TOP_K,
) -> pd.DataFrame:
    """Top-k by alpha*S_num + beta*S_genre, excluding the reference itself.

    Ties are broken by track_id so the ordering is deterministic.
    """
    combined = alpha * s_num + (1.0 - alpha) * s_genre

    order = np.lexsort((fm.catalogue["track_id"].to_numpy(), -combined))
    order = order[order != exclude_row][:k]

    out = fm.catalogue.iloc[order][["track_id", "track_name", "artists", "genre_set"]].copy()
    out["score"] = combined[order]
    out["s_num"] = s_num[order]
    out["s_genre"] = s_genre[order]
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def genre_match_count(fm: FeatureMatrix, ref_genres: tuple[str, ...], order_rows: np.ndarray) -> int:
    ref = set(ref_genres)
    return int(sum(bool(ref & set(fm.catalogue.iloc[r]["genre_set"])) for r in order_rows))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def fmt_genres(genres: tuple[str, ...], limit: int = 3) -> str:
    if len(genres) <= limit:
        return ", ".join(genres)
    return ", ".join(genres[:limit]) + f" +{len(genres) - limit}"


def main() -> int:
    fm = build_feature_matrix()
    refs = pick_reference_tracks(fm)

    print(f"catalogue: {fm.n_tracks:,} tracks x {fm.n_features} features")
    print(f"references: {len(refs)}")
    for r in refs:
        print(f"  - {r['label']:<34} {r['track_name'][:40]!r} | {fmt_genres(r['genres'])}")

    # Numerical check that the weighted cosine equals the weighted sum of blocks.
    r0 = refs[0]["row"]
    s_num0, s_genre0 = block_similarities(fm, r0)
    ref_vec = fm.matrix.getrow(r0)
    cosine_full = (fm.matrix @ ref_vec.T).toarray().ravel()
    max_gap = float(np.max(np.abs(cosine_full - (0.7 * s_num0 + 0.3 * s_genre0))))

    lines: list[str] = []
    add = lines.append

    add("# Checkpoint C3 — feature-weight sensitivity analysis")
    add("")
    add("Read-only analysis over the approved Phase 4 matrix. No recommender is")
    add("implemented; this document exists to gather evidence before Phase 5.")
    add("")
    add("## Method")
    add("")
    add(f"- Catalogue: **{fm.n_tracks:,} tracks**, matrix **{fm.n_features} columns** "
        f"(9 numeric + 114 genre), built by `build_feature_matrix()`.")
    add("- `S_num` = cosine similarity of the 9 standardised, row-normalised audio features.")
    add("- `S_genre` = cosine similarity of the 114 multi-label genre indicators.")
    add("- Combined score = `alpha * S_num + beta * S_genre`, with `beta = 1 - alpha`.")
    add("- Top 10 per reference, reference track itself excluded, ties broken by `track_id`.")
    add(f"- Selected reference tracks are deterministic (most popular in each genre profile, "
        f"ties by `track_id`).")
    add(f"- Sanity check: `cosine(weighted matrix)` vs `0.7*S_num + 0.3*S_genre` differ by at most "
        f"**{max_gap:.2e}** — the block decomposition holds.")
    add("")
    add("**Interpretation guard:** every number below is a cosine similarity between feature")
    add("vectors. It is not a percentage, not a probability, and not a prediction that a")
    add("listener will like a track.")
    add("")

    add("## Reference tracks")
    add("")
    add("| id | reference | artist | genres | popularity |")
    add("|---|---|---|---|---|")
    for r in refs:
        add(f"| {r['track_id']} | {r['track_name']} | {r['artists']} | "
            f"{fmt_genres(r['genres'], 4)} | {r['popularity']} |")
    add("")

    # ---- duplicate-song diagnostics -------------------------------------- #
    pairs = fm.catalogue.groupby(["track_name", "artists"]).size()
    multi_pairs = pairs[pairs > 1]
    dup_rows = int(multi_pairs.sum())

    by_pair = fm.catalogue.groupby(["track_name", "artists"])[list(NUMERIC_FEATURES)]
    varying_within_group = by_pair.nunique().max(axis=1).loc[multi_pairs.index]
    same_groups = int((varying_within_group <= 1).sum())
    diff_groups = int(len(multi_pairs) - same_groups)

    add("## Duplicate songs that de-duplication by `track_id` does not catch")
    add("")
    add("The Phase 4 catalogue is one row per `track_id`, and the 24,259 repeated-id rows are")
    add("gone. But **the same recording can carry two different `track_id`s**, and those rows")
    add("survive de-duplication:")
    add("")
    add(f"- distinct `(track_name, artists)` pairs in the catalogue: **{len(pairs):,}**")
    add(f"- pairs with more than one `track_id`: **{len(multi_pairs):,}**")
    add(f"- rows involved: **{dup_rows:,}** ({dup_rows / len(fm.catalogue):.1%} of the catalogue)")
    add(f"- largest group: **{int(pairs.max())}** rows for the same title and artist")
    add("")
    add("Example found directly in a rock top-10: *Follow You* by Imagine Dragons appears twice")
    add("with different ids (`1skWrgJrEi6ZB4Wt5tYYAb`, `7FdUvDkaE24o3FPIWTvzv2`), identical tempo")
    add("and energy, popularity 2 vs 73. This is why a top-10 can contain the same song twice —")
    add("a real defect against the brief's \"avoid returning duplicate recommendations\".")
    add("")
    add("| group type | groups |")
    add("|---|---|")
    add(f"| same title+artist, identical audio features (same recording) | **{same_groups:,}** |")
    add(f"| same title+artist, differing audio features (possibly a different version) | **{diff_groups:,}** |")
    add("")
    add("So a duplicate-suppression rule at ranking time must be deliberate: collapsing every")
    add("same-title/artist group would also hide legitimate live/remaster variants whose audio")
    add("features differ. This needs an explicit decision in Phase 5, not a silent fix.")
    add("")

    # ---- per-reference detail -------------------------------------------- #
    add("## Top-10 at the extremes and at the current default")
    add("")
    for r in refs:
        s_num, s_genre = block_similarities(fm, r["row"])
        add(f"### {r['label']} — *{r['track_name']}* ({fmt_genres(r['genres'], 4)})")
        add("")
        for alpha in (0.50, 0.70, 0.90):
            top = rank(fm, s_num, s_genre, alpha, r["row"])
            ref_genres = set(r["genres"])
            matched = int(sum(bool(ref_genres & set(g)) for g in top["genre_set"]))
            distinct_songs = top.groupby(["track_name", "artists"]).ngroups
            add(f"**alpha={alpha:.2f} / beta={1-alpha:.2f}** "
                f"— {matched}/10 share at least one genre; "
                f"{distinct_songs}/10 distinct songs; "
                f"mean S_num {top['s_num'].mean():.3f}, mean S_genre {top['s_genre'].mean():.3f}")
            add("")
            add("| # | track | artist | genres | combined | S_num | S_genre |")
            add("|---|---|---|---|---|---|---|")
            for _, row in top.iterrows():
                add(f"| {int(row['rank'])} | {row['track_name']} | {row['artists']} | "
                    f"{fmt_genres(row['genre_set'])} | {row['score']:.3f} | "
                    f"{row['s_num']:.3f} | {row['s_genre']:.3f} |")
            add("")

    # ---- overlap across alphas ------------------------------------------- #
    add("## How much the list changes with the weight")
    add("")
    add("Jaccard overlap of the top-10 sets (|intersection| / |union|) against alpha=0.50,")
    add("and against the previous alpha.")
    add("")
    add("| reference | 0.50 vs 0.60 | 0.50 vs 0.70 | 0.50 vs 0.80 | 0.50 vs 0.90 | 0.50 set vs 0.90 set (shared) |")
    add("|---|---|---|---|---|---|")

    overlap_rows = []
    for r in refs:
        s_num, s_genre = block_similarities(fm, r["row"])
        sets = {a: set(rank(fm, s_num, s_genre, a, r["row"]).track_id) for a in ALPHAS}
        base = sets[0.50]
        jac = {a: len(base & sets[a]) / len(base | sets[a]) for a in ALPHAS}
        shared = len(base & sets[0.90])
        overlap_rows.append((r["label"], jac, shared))
        add(f"| {r['label']} | {jac[0.60]:.2f} | {jac[0.70]:.2f} | {jac[0.80]:.2f} | "
            f"{jac[0.90]:.2f} | {shared}/10 |")
    add("")

    # ---- block diagnostics over a sample -------------------------------- #
    add("## Does one block dominate? (sample of "
        f"{GLOBAL_SAMPLE} tracks)")
    add("")
    add("For each sampled reference the top-10 was computed at beta-heavy (0.5/0.5) and")
    add("numeric-heavy (0.9/0.1) settings, then the *composition of the returned list* was")
    add("measured. `genre-match` = share of the top-10 sharing at least one genre with the")
    add("reference.")
    add("")

    rng = np.random.default_rng(GLOBAL_SEED)
    sample_rows = rng.choice(fm.n_tracks, size=min(GLOBAL_SAMPLE, fm.n_tracks), replace=False)

    stats = {a: {"genre_match": [], "s_num": [], "s_genre": []} for a in (0.50, 0.70, 0.90)}
    for row in sample_rows:
        s_num, s_genre = block_similarities(fm, int(row))
        ref_genres = set(fm.catalogue.iloc[int(row)]["genre_set"])
        for a in (0.50, 0.70, 0.90):
            top = rank(fm, s_num, s_genre, a, int(row))
            match = sum(bool(ref_genres & set(g)) for g in top["genre_set"])
            stats[a]["genre_match"].append(match / len(top))
            stats[a]["s_num"].append(float(top["s_num"].mean()))
            stats[a]["s_genre"].append(float(top["s_genre"].mean()))

    add("| setting | mean genre-match in top-10 | mean S_num of top-10 | mean S_genre of top-10 |")
    add("|---|---|---|---|")
    for a in (0.50, 0.70, 0.90):
        add(f"| alpha={a:.2f} | {np.mean(stats[a]['genre_match']):.3f} | "
            f"{np.mean(stats[a]['s_num']):.3f} | {np.mean(stats[a]['s_genre']):.3f} |")
    add("")

    # ---- conclusions ----------------------------------------------------- #
    weakest = min(overlap_rows, key=lambda t: t[2])
    strongest = max(overlap_rows, key=lambda t: t[2])

    add("## Observations")
    add("")
    add(f"- The top-10 changes most for **{weakest[0]}** ({weakest[2]}/10 tracks survive from")
    add(f"  alpha=0.50 to 0.90) and least for **{strongest[0]}** ({strongest[2]}/10).")
    add(f"- Across the {len(sample_rows)} sampled references, genre-match in the top-10 rises from")
    add(f"  **{np.mean(stats[0.50]['genre_match']):.3f}** at 0.5/0.5 to "
        f"**{np.mean(stats[0.90]['genre_match']):.3f}** at 0.9/0.1.")
    add(f"- Mean S_num of the returned list moves from **{np.mean(stats[0.50]['s_num']):.3f}** "
        f"(0.5/0.5) to **{np.mean(stats[0.90]['s_num']):.3f}** (0.9/0.1), while mean S_genre")
    add(f"  moves from **{np.mean(stats[0.50]['s_genre']):.3f}** to "
        f"**{np.mean(stats[0.90]['s_genre']):.3f}**.")
    add("")
    add("### Does one block dominate?")
    add("")
    add(f"- **At 0.5/0.5 and 0.7/0.3 the genre block effectively decides the list**: "
        f"{np.mean(stats[0.50]['genre_match']):.0%} and "
        f"{np.mean(stats[0.70]['genre_match']):.0%} of sampled top-10 results share a genre with "
        f"the reference. A genre weight of only 0.3 already makes nearly every neighbour "
        f"same-genre.")
    add(f"- **Only at 0.9/0.1 does audio similarity visibly lead**: genre-match falls to "
        f"{np.mean(stats[0.90]['genre_match']):.0%} and cross-genre results appear (e.g. a house "
        f"track beside a multi-genre reference with S_genre = 0.000).")
    add("- This is structural rather than a bug: two tracks in the same genre have S_genre = 1.0 "
        "exactly, while two very close audio profiles rarely exceed S_num ≈ 0.95, so genre wins "
        "whenever its weight is comparable.")
    add("- One reference (*classical*) returns an identical top-10 at every weight, because its "
        "nearest tracks are close in **both** blocks; re-weighting cannot change a list whose "
        "candidates already lead in both.")
    add("")
    add("### Defect found: duplicate songs inside one list")
    add("")
    add(f"- A rock top-10 contains *Follow You* twice: {len(multi_pairs):,} "
        f"`(track_name, artists)` groups hold more than one `track_id`. De-duplication by "
        f"`track_id` therefore does **not** yet satisfy the brief's no-duplicate-results rule.")
    add("- Phase 5 needs an explicit duplicate-suppression rule, and it must not be silent: "
        f"{diff_groups:,} same-title groups differ in audio features and may be genuinely "
        "different versions (live/remaster).")
    add("")
    add("- No weight is 'correct': without relevance labels there is no ground truth to optimise")
    add("  against, and the highest-scoring list is not necessarily the one a listener prefers.")
    add("  The choice is a documented design decision, to be defended qualitatively and with the")
    add("  structural checks planned in Phase 6.")
    add("")
    add("## Exclusions applied")
    add("")
    add("- **Self-match:** each reference row is removed before ranking (it would score 1.0).")
    add("- **Duplicates:** the 24,259 repeated-`track_id` rows are gone (Phase 4), and the 1 junk "
        "row plus 157 zero-coded tracks are absent from the catalogue. **However**, the same "
        "recording under two different `track_id`s is *not* removed — see the duplicate section "
        "above; that is a known defect to resolve in Phase 5, reported rather than silently "
        "patched here.")
    add("")

    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(f"max |cosine_full - (0.7*S_num + 0.3*S_genre)| = {max_gap:.2e}")
    print(f"written: {OUT_DOC.relative_to(PROJECT_ROOT)}")
    print()
    print("genre-match in top-10 (sample):")
    for a in (0.50, 0.70, 0.90):
        print(f"  alpha={a:.2f}  genre-match={np.mean(stats[a]['genre_match']):.3f}  "
              f"meanS_num={np.mean(stats[a]['s_num']):.3f}  "
              f"meanS_genre={np.mean(stats[a]['s_genre']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
