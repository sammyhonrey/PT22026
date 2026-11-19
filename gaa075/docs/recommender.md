# Recommendation engine

Implemented in `recommender.py` (Phase 5, tasks 5.1–5.8), tested in
`tests/test_recommender.py`. It consumes the approved Phase 4 matrix and adds no
new data preparation.

## 1. Scoring

```text
base  = 0.90 * S_num + 0.10 * S_genre
final = base                                          (no preferences given)
final = 0.80 * base + 0.20 * preference_score         (preferences given)
```

* `S_num` — cosine similarity of the 9 standardised, row-normalised audio
  features (danceability, energy, speechiness, acousticness, instrumentalness,
  liveness, valence, loudness, tempo).
* `S_genre` — cosine similarity of the 114-dimension multi-label genre vectors.
* Both blocks are unit-normalised, so each component is a matrix–vector product;
  the combined value equals the cosine of the weighted concatenation (verified to
  4.4e-16 in `docs/weight-sensitivity.md`).
* Scores are clipped to `[-1, 1]`: float arithmetic produced 1.0000000000000002
  in testing.

**Why 0.90 / 0.10.** The C3 analysis showed the genre block decides the list
whenever its weight reached 0.3 (100% of sampled top-10 results shared a genre at
0.7/0.3). 0.90/0.10 lets audio lead while genre still breaks near-ties. This is a
**documented design decision, not a proven optimum** — the dataset has no
relevance labels. Weights stay configurable.

## 2. User preferences (task 5.4)

```python
from recommender import Preferences

prefs = Preferences.build(genres=["piano", "rock"], artists=["Elton John"])
results = engine.recommend(track_id, k=10, preferences=prefs)
```

`preference_score` is in `[0, 1]` and is built from the requested values only:

| Component | Definition |
|---|---|
| genres | share of the *requested* genres the track belongs to (multi-label aware) |
| artists | share of the *requested* artists the track is credited to (any of the `;`-separated credits) |

* **Genre-only, artist-only and combined** are all supported.
* When both are given, `preference_score` is the **mean** of the two components.
* **No preferences → the ranking and the scores are exactly the content-based
  result** (`score == base_score`, `preference_score == 0`). This is asserted by
  `test_no_preferences_preserves_ranking_and_scores`.
* Input is normalised: stripped, case-folded, de-duplicated, sorted, so the same
  intent always yields identical output.
* An unknown genre raises `InvalidRequestError` instead of silently scoring 0;
  `Recommender.genres` exposes the valid vocabulary for the UI.
* Preferences influence **ranking only within their 0.20 share**: a preference
  match can lift a track by at most 0.20, and it cannot rescue a track whose
  audio profile is far away.
* If preferences match *nothing* in the candidate set, every score is multiplied
  by 0.80 and the order is unchanged (0.8·base is monotonic). This is expected
  behaviour, not a bug — and it is why the UI should say when nothing matched.

## 3. Explanations (task 5.5)

Each result carries an `explanation` string built from real values only, in a
fixed order:

1. `shared genre(s): …` — intersection of the reference's and the candidate's
   genre sets (up to 3, alphabetically).
2. `closest audio traits: …` — the 1–2 features with the largest **positive**
   element-wise contribution `z_ref[i] · z_cand[i]`. Because both rows are unit
   normalised, these products sum exactly to `S_num`, so the claim is arithmetic,
   not prose.
3. `matches preferred genre: …` / `matches preferred artist: …` — only when the
   user asked for that value and the track satisfies it.
4. Fallback `no shared genre; the audio profile is the only link` when nothing
   above applies.

Deterministic by construction: genres are sorted, traits are sorted by
contribution then name. `test_explanations_are_deterministic` asserts it.

## 4. Score warning (task 5.6)

One reusable constant carries the caveat:

> **`SCORE_INTERPRETATION_WARNING`** — "Similarity scores compare track features
> (audio profile and genre). They are not probabilities, not percentages, and not
> a prediction that you will like a song; only the ranking order is meaningful."

Every returned DataFrame carries it in `results.attrs["score_warning"]`, including
empty results, and `attrs["preferences_applied"]` records whether preferences were
applied. The UI must display it wherever scores appear.

## 5. Duplicate-recording suppression (approved rule, unchanged)

A row is a repeat of another only when **all four** parts match:

| Part | Normalisation |
|---|---|
| `track_name` | strip, then `casefold()` |
| `artists` | strip, then `casefold()` |
| 9 audio features | equal after rounding to 9 decimals |
| `duration_ms` | exact integer equality |

Results are scanned in score order, so the **highest-scoring** duplicate survives;
rows whose audio or duration differ stay as separate variants; catalogue rows are
never removed. The reference's own signature is pre-seeded, so the same recording
cannot reappear under a different `track_id` (a defect found at C3).

## 6. Ordering and errors

Ordering is final-score descending, `track_id` ascending as tie-break
(`np.lexsort`), so calls are reproducible.

| Situation | Behaviour |
|---|---|
| unknown `track_id` | `UnknownTrackError` |
| `track_id` None/empty/whitespace/non-string | `InvalidRequestError` |
| `k` not an int, `<1` or `>50` | `InvalidRequestError` |
| unknown genre in preferences | `InvalidRequestError` |
| preferences not a `Preferences` instance | `InvalidRequestError` |
| weights not summing to 1, or negative | `InvalidRequestError` at construction |
| fewer distinct candidates than `k` | returns as many as exist, ranks contiguous |
| no candidates (single-track catalogue) | empty DataFrame with documented columns |

## 7. API

```python
engine = Recommender(build_feature_matrix())          # 0.90 numeric / 0.10 genre
engine.recommend(track_id, k=10)                      # content-based
engine.recommend(track_id, k=10, preferences=prefs)   # preference-adjusted
engine.scores(track_id)                               # per-track components, no prefs
```

Result columns: `rank, track_id, track_name, artists, album_name, genres,
duration_ms, score, base_score, preference_score, numeric_score, genre_score,
explanation`.

## 8. Measured behaviour (real catalogue, 89,583 tracks)

Reference *I Ain't Worried* — OneRepublic (genres: piano, pop, rock):

| setting | top result | score (base + pref) |
|---|---|---|
| no preferences | *Bad Blood* — Bastille | 0.930 (0.930 + 0.00) |
| genre = piano | *Crocodile Rock* — Elton John | 0.930 (0.913 + 1.00) |
| artist = Elton John | *Crocodile Rock* — Elton John | 0.930 (0.913 + 1.00) |
| genre = piano + artist = Elton John | *Crocodile Rock* — Elton John | 0.930 (0.913 + 1.00) |

Duplicate diagnostics over 300 sampled references × top-10 (from
`analysis/sample_recommendations.py`):

| Check | Result |
|---|---|
| lists containing the same full signature twice | **0** |
| lists containing the same title+artist twice | **41** |
| lists containing the same title+artist+duration twice | **21** |

## 9. Limitations (honest)

1. **Look-alike duplicates remain** by design: suppression requires matching audio
   *and* duration, so 41/300 sampled lists still show the same title+artist twice
   when stored audio values differ slightly (e.g. *Hold Me Closer – Purple Disco
   Machine Remix* at 0.881 and 0.878). Those may be one recording measured twice or
   two versions; the data cannot distinguish them.
2. **Preferences are a bounded nudge**, not personalization learning: at most
   0.20 of the score, no memory, no feedback, no user profile.
3. **Unmatched preferences silently scale scores by 0.8** (order preserved). The UI
   should surface "no track matched your preferences".
4. **No user model / no collaborative signal.** Nothing here learns from behaviour.
5. **Popularity bias is not corrected**; popularity is not in the vector.
6. **157 zero-coded tracks** are unreachable as references (Phase 4) — the UI must
   handle "not found".
7. **Low diversity** is inherent to content-based neighbours; single-genre
   references stay in-genre.
8. **Scores are not calibrated**: 0.93 is not "93% chance of liking"; only the
   ordering is meaningful.

## 10. Not implemented (deliberately)

* Streamlit UI (`app.py` untouched).
* On-disk matrix caching (optional task 4.8).
* Multi-user persistence, feedback loops, or any "learning" claim.
