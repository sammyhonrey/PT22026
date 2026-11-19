"""Print sample recommendations from the real catalogue (read-only, reproducible).

Run from the project root:

    .venv\\Scripts\\python.exe analysis\\sample_recommendations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import FeatureMatrix, build_feature_matrix  # noqa: E402
from recommender import Preferences, Recommender  # noqa: E402


def pick(fm: FeatureMatrix, genre: str) -> str:
    cat = fm.catalogue
    subset = cat.loc[cat["genre_set"].map(lambda g: genre in g), ["popularity", "track_id"]]
    subset = subset.reset_index(drop=True).sort_values(
        ["popularity", "track_id"], ascending=[False, True]
    )
    return str(subset.iloc[0]["track_id"])


def most_multi_genre(fm: FeatureMatrix) -> str:
    cat = fm.catalogue
    sizes = cat["genre_set"].map(len)
    top = cat.loc[sizes == sizes.max()].sort_values("popularity", ascending=False)
    return str(top.iloc[0]["track_id"])


def show(engine: Recommender, label: str, track_id: str, k: int = 5) -> None:
    ref = engine._catalogue.loc[track_id]
    print("=" * 108)
    print(f"{label}")
    print(f"reference : {ref['track_name']} — {ref['artists']}  |  genres: {', '.join(ref['genre_set'])}")
    results = engine.recommend(track_id, k=k)
    for _, row in results.iterrows():
        print(
            f"  {int(row['rank'])}. {str(row['track_name'])[:36]:<36} | {str(row['artists'])[:24]:<24} "
            f"| {str(row['genres'])[:28]:<28} | S {row['score']:.3f} "
            f"(num {row['numeric_score']:.3f} / gen {row['genre_score']:.3f})"
        )
    titles = list(zip(results["track_name"].str.casefold(), results["artists"].str.casefold()))
    print(f"  distinct (title, artist) pairs in this list: {len(set(titles))}/{len(results)}")


def main() -> int:
    fm = build_feature_matrix()
    engine = Recommender(fm)
    print(f"catalogue {engine.n_tracks:,} tracks | weights numeric={engine.numeric_weight} "
          f"genre={engine.genre_weight}")
    print(f"duplicate-recording suppression: {engine._suppress}")

    for label, tid in [
        ("rock", pick(fm, "rock")),
        ("hip-hop", pick(fm, "hip-hop")),
        ("classical", pick(fm, "classical")),
        ("sleep (quiet/ambient profile)", pick(fm, "sleep")),
        ("most multi-genre", most_multi_genre(fm)),
    ]:
        show(engine, label, tid)

    # duplicate checks over a sample of references, split by severity
    rng = np.random.default_rng(11)
    sample = rng.choice(engine.n_tracks, size=300, replace=False)
    full_signature_dups = 0
    same_title_artist = 0
    same_title_artist_duration = 0
    for row in sample:
        tid = engine._track_ids[row]
        results = engine.recommend(tid, k=10)

        keys = [engine._duplicate_keys[engine.track_row(t)] for t in results["track_id"]]
        if len(keys) != len(set(keys)):
            full_signature_dups += 1

        ta = list(zip(results["track_name"].str.casefold(), results["artists"].str.casefold()))
        if len(ta) != len(set(ta)):
            same_title_artist += 1

        tad = list(zip(
            results["track_name"].str.casefold(),
            results["artists"].str.casefold(),
            results["duration_ms"],
        ))
        if len(tad) != len(set(tad)):
            same_title_artist_duration += 1

    print("=" * 108)
    print(f"sampled {len(sample)} references x top-10:")
    print(f"  lists with the same full recording signature twice : {full_signature_dups}"
          f"   (must be 0 — the engine's guarantee)")
    print(f"  lists with same title+artist twice (any duration)   : {same_title_artist}"
          f"   (allowed: audio or duration differ)")
    print(f"  lists with same title+artist+duration twice         : {same_title_artist_duration}"
          f"   (allowed only when audio features differ)")

    show_preferences(fm, engine)
    return 0


def show_preferences(fm: FeatureMatrix, engine: Recommender) -> None:
    """Same reference, with and without preferences (task 5.4 demonstration)."""
    catalogue = engine._catalogue
    match = catalogue[
        (catalogue["track_name"] == "I Ain't Worried")
        & (catalogue["artists"] == "OneRepublic")
    ]
    track_id = str(match.index[0])

    scenarios = [
        ("no preferences (reference ranking)", None),
        ("genre preference: piano", Preferences.build(genres=["piano"])),
        ("artist preference: Elton John", Preferences.build(artists=["Elton John"])),
        ("combined: genre piano + artist Elton John",
         Preferences.build(genres=["piano"], artists=["Elton John"])),
    ]

    for label, prefs in scenarios:
        results = engine.recommend(track_id, k=5, preferences=prefs)
        print("=" * 108)
        print(f"{label}   (preferences_applied={results.attrs['preferences_applied']})")
        for _, row in results.iterrows():
            print(
                f"  {int(row['rank'])}. {str(row['track_name'])[:32]:<32} "
                f"| {str(row['artists'])[:22]:<22} | score {row['score']:.3f} "
                f"(base {row['base_score']:.3f} + pref {row['preference_score']:.2f})"
            )
            print(f"      why: {row['explanation']}")

    print("=" * 108)
    print("score warning shown with every result list:")
    print("  " + engine.recommend(track_id, k=1).attrs["score_warning"])


if __name__ == "__main__":
    raise SystemExit(main())
