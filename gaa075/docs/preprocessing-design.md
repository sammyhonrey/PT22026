# Proposed preprocessing design (Phase 4)

Status: **implemented 2026-11-18 (Phase 4, tasks 4.1–4.6).** The implementation
record with real numbers and test results is in §10 at the end. §1–§9 are the
design as approved, updated where the approval changed a decision.

Verified inputs (see `docs/dataset.md`): 114,000 rows × 21 columns, 89,741 unique
`track_id`s, 114 genres × 1,000 rows, 1 junk row, 157 zero-coded rows.

---

## 0. Pipeline overview

```
raw CSV
  │ load_raw_data()                          [implemented, task 2.3]
  ▼
drop junk row + zero-coded rows              [step 1]
  ▼
build per-track catalogue, de-duplicate      [step 2]  ← one row per track_id
  ├── genre membership sets (multi-label)    [step 3]  ← preserved, not collapsed
  ▼
transform numeric features                   [step 4]
  ▼
scale + clip                                 [step 5]
  ▼
assemble weighted matrix                     [step 6]
  └── X  (rows = tracks, cols = 9 numeric + 114 genre indicators)
```

---

## 1. Dropping unusable rows

### 1a. The junk row — drop, with a recorded reason

Exactly **one** row (`index 65900`, `track_id 1kR4gIb7nGxHPI3D2ifs59`, genre
`k-pop`) has `NaN` in `artists`, `album_name` and `track_name` **and**
`duration_ms == 0` **and** `popularity == 0`. It cannot be displayed to a user,
cannot be a reference song, and carries no usable content. **Drop it.**

### 1b. Zero-coded missing values — the 157 rows

Investigated in full. The evidence:

| Observation | Value |
|---|---|
| Rows with `tempo == 0` | **157** |
| Of those, `danceability == 0` **exactly** | **157 / 157** |
| Of those, `speechiness == 0` **exactly** | **157 / 157** |
| Of those, `valence == 0` **exactly** | **157 / 157** |
| Genre of those rows | **sleep 138**, iranian 4, guitar 4, ambient 3, world-music 3, opera 2, jazz 1, show-tunes 1, romance 1 |
| Their mean `energy` | 0.123 (dataset 0.641) |
| Their mean `loudness` | −22.1 dB (dataset −8.3 dB) |
| Their mean `instrumentalness` | 0.65 (dataset 0.16) |
| Do these `track_id`s appear elsewhere with a real tempo? | **0** |

Three features are exactly zero simultaneously in all 157 rows, and `tempo` is
zero too. A real track cannot have danceability, speechiness and valence all at
exactly 0.0; the values are absent measurements encoded as zero. The genre mix
(88% `sleep`, plus ambient/opera/jazz — quiet or spoken-word material) is exactly
where such analysis fails.

**Decision: treat `{tempo, danceability, speechiness, valence} == 0` as missing,
then drop the affected rows from the recommendation catalogue (157 rows = 0.14%).**

Why drop instead of impute:

- Imputing would fabricate audio measurements that the project cannot verify.
- Any imputation (mean/median/genre-median) would place 157 tracks at the centre
  of the feature space, making them look artificially "average" and therefore
  similar to everything — a systematic distortion.
- At 0.14% of 114,000 rows the loss is negligible.

**Known consequence (documented, not hidden):** the `sleep` genre loses 138 of
its 1,000 tracks (13.8%) and remains with **862** usable tracks. Genre counts are
therefore no longer exactly balanced after cleaning, and the report must say so.

Measurement subtlety (found during implementation): counting *rows* labelled
`sleep` after de-duplication gives **997 → 859**, not 1000 → 862. The difference
is three tracks whose first row was seen under another genre. The correct
multi-label count is by genre **membership** (a track is a sleep track if `sleep`
is in its genre set), which gives exactly **1000 → 862**. The pipeline reports
the membership-based numbers.

---

## 2. De-duplication by `track_id` — deterministic

Facts: 24,259 rows repeat a `track_id`; 89,741 unique tracks; 16,299 repeated ids
appear under more than one genre (max 7 genres for one track).

**Proposed rule (deterministic, order-based):** group by `track_id` and keep the
**first occurrence in original file order**. All other rows of that group are
discarded from the numeric catalogue, but their genre labels are not lost (step 3).

Why this rule is safe rather than arbitrary:

- All 720 ids whose repeated rows *differ* were investigated: **the only column
  that ever differs is `popularity`** (median range 1, max 44). Every audio
  feature, `duration_ms`, `explicit` and the text fields are byte-identical
  across repeats.
- `popularity` is excluded from the similarity vector by your approved feature
  set, so **which row is kept cannot change a single similarity score.**
- A "first occurrence" rule therefore needs no averaging, no majority vote and no
  invented tie-breaking, and is exactly reproducible.

Explicitly rejected alternatives: averaging the duplicate rows (invents values
and is pointless when values are identical), keeping max popularity (same result
for the model, more code), random choice (non-deterministic).

---

## 3. Preserving multiple genre associations

Genre must not be collapsed to one arbitrary bucket. Per `track_id`, collect the
**set** of every genre the row appeared under:

```python
genres = df.groupby("track_id")["track_genre"].agg(lambda s: sorted(set(s)))
```

This set is used twice:

1. **In the similarity vector** — multi-label one-hot: a track that is both
   `rock` and `country` gets 1.0 in both columns. The block is then L2-normalised
   per row, so a track with `k` genres ends with `1/√k` in each of its columns
   (1.0 for a single-genre track). That keeps the genre block at unit norm so it
   contributes exactly `β` to the cosine similarity, and it dilutes each
   individual genre for multi-genre tracks instead of letting them look similar
   to everything.

   *Implementation note:* an earlier draft also divided by `k` before
   normalising. That step is directionally identical to the final normalisation,
   so it was dropped rather than kept as dead arithmetic; the dilution comes from
   the normalisation itself.
2. **For display and preference filtering** — the same list is shown in the UI
   and used by the later genre preference filter, which can test membership
   rather than equality.

---

## 4. Numeric transformations

Approved features: `danceability`, `energy`, `speechiness`, `acousticness`,
`instrumentalness`, `liveness`, `valence`, `loudness`, `tempo`, `duration_ms`.

| Feature(s) | Transformation | Justification |
|---|---|---|
| 7 unit features (0–1) | none, then standardised (step 5) | already bounded and comparable |
| `loudness` (dB, −49.5…+4.5) | none, then standardised | roughly symmetric; 90 rows above 0 dB are real, not errors |
| `tempo` (BPM, 0…243) | none after zero→missing removal | zeros are the only pathology; the rest is a plausible BPM range |
| `duration_ms` | **excluded from the vector** (approved Decision 4) | kept in the catalogue for display and future filtering only; it is arguably a format property rather than a musical one, and including it risks "similar because both are long" |

Nothing else is transformed. No binning, no PCA in the first version — both would
reduce explainability for an intermediate-level project without evidence of need.

---

## 5. Scaling

**Choice: `StandardScaler` fitted once on the cleaned catalogue, applied to the
9 numeric features.** Rationale:

- Cosine similarity compares *directions*; without scaling, `duration_ms`
  (range ~5 × 10⁶) would swamp `valence` (range ~1), and the "similarity" would
  effectively be a duration match.
- `StandardScaler` puts each feature on a comparable footing (variance 1), which
  is a documented, defensible default and easy to explain in the report.
- Outlier protection is handled by the `log1p` transform for duration rather than
  by removing rows; if the loudness/tempo tails still distort results at
  Checkpoint C3, the fallback is `RobustScaler`, recorded as a tested alternative.

The scaler is fitted on the **recommendation catalogue only** (after cleaning and
dedup), never on a subset, so the app and the evaluation path share one fit.

---

## 6. Block weighting — not assumed equal

Two blocks, each row-wise unit-normalised, then weighted:

```
X = [ sqrt(α) · N ,  sqrt(β) · G ]      α + β = 1
```

- `N` = 10 standardised numeric features, row-normalised to unit L2 norm.
- `G` = 114-column multi-label genre matrix, row-normalised to unit L2 norm
  (equivalent to the 1/|genres| rule above after normalisation).
- Because both blocks are unit norm, **cosine similarity decomposes as**
  `α · S_numeric + β · S_genre` — easy to explain, and each block's influence is
  an explicit, testable number.

**Initial value: α = 0.7, β = 0.3.** This is a *starting hypothesis*, not a
finding: audio profile is the primary content signal, genre is a coarse label
with 114 categories. It will be tested by sweeping α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}
at Checkpoint C3 and by the qualitative multi-song review in Phase 6, and the
chosen value will be justified with that evidence. Nothing will claim genre
weighting is "optimal" without those results.

Encoding note: `α` and `β` are applied as square roots inside the matrix so that
the resulting cosine similarity equals the weighted sum above; applying raw
weights would change the norm and silently rescale similarity.

---

## 7. Outputs of the pipeline

| Output | Content |
|---|---|
| `catalogue` | one row per `track_id`: display fields (`track_name`, `artists`, `album_name`), `genre_set`, and the 10 numeric features |
| `X` | `scipy.sparse` matrix, `n_tracks × 123` (9 numeric + 114 genre), rows unit-norm overall |
| `track_ids` | index array mapping matrix row → `track_id` (the identity key) |
| `scaler` | the fitted `StandardScaler`, reused by the app and evaluation |

Expected size after cleaning: **89,741 − (junk row) − (zero-coded rows, if they
survive de-duplication) ≈ 89,600 tracks**. The exact post-cleaning count will be
printed and recorded when the pipeline is implemented — not assumed here.

---

## 8. What this design deliberately does not do

- No imputation of audio features (fabrication risk).
- No averaging of conflicting duplicate rows (they do not actually conflict).
- No arbitrary single-genre choice (multi-label is preserved).
- No PCA/binning in v1 (explainability first).
- No modelling of user history, no collaborative filtering.
- No claim that α = 0.7 is optimal — it is a hypothesis to be tested.

## 9. Open questions for your review

1. **Zero-coded rows:** approve dropping the 157 rows, accepting that `sleep`
   drops to 862 usable tracks? (Alternative: keep them with those four features
   set to the catalogue median — explicitly rejected above.)
2. **Dedup rule:** approve "first occurrence in file order"?
3. **α = 0.7 / β = 0.3** as the starting weighting to be swept at C3?
4. **`log1p(duration_ms)`** approved, or would you rather exclude duration
   entirely from similarity (it is arguably a format property, not a musical one)?

---

## 10. Implementation record (2026-11-18)

Implemented in `preprocessing.py`, tested in `tests/test_preprocessing.py`.
Duration was **excluded** from the vector per the approved Decision 4, leaving
**9** numeric features.

### Real-data results

| Quantity | Value |
|---|---|
| Rows loaded | 114,000 |
| Junk rows dropped | **1** |
| Duplicate rows dropped (keep first) | **24,259** |
| Zero-coded tracks excluded | **157** |
| **Tracks in the final catalogue** | **89,583** |
| `sleep` tracks, before → after | **1,000 → 862** (membership count) |
| Numeric block | **(89,583 × 9)** |
| Genre block | **(89,583 × 114)** |
| Final matrix | **(89,583 × 123)** |
| Non-zero entries | 919,639 (density **8.35%**) |
| Row norms | 1.000000000000 … 1.000000000000 |
| Non-finite values | **0** |
| Build time | ~2.0 s |
| Raw CSV SHA256 before/after | identical (`b202fa49…177bd`) |

The catalogue arithmetic closes exactly:
`114,000 − 1 junk − 24,259 duplicates − 157 zero-coded = 89,583`.

### Verification performed

- 19 pytest tests pass, including one integration test against the real dataset.
- Tests cover: junk-row removal, zero-coded detection, duplicate "first
  occurrence" rule, multi-label genre preservation, reported counts, scaler
  fitting, zero-variance features, matrix dimensions, duration exclusion, unit
  row norms, block weighting, invalid weights, reproducibility, unknown-track
  lookup, loader errors, and raw-data immutability.
- The pipeline is deterministic: two consecutive builds produce byte-identical
  matrices.

### Deviations and corrections made during implementation

1. **Duration excluded**, per approved Decision 4 → 9 numeric features, no
   `log1p`.
2. **Sleep accounting** uses genre membership (1000 → 862), not the label on the
   surviving row (which would report 997 → 859). Explained in §1b.
3. **Genre block** normalised once (L2), not divided by `k` and normalised.
4. **Constant-feature handling** is sklearn's: `scale_ = 1.0` when variance is
   zero. The real catalogue has no constant feature, but the test fixture does,
   and the behaviour is now asserted explicitly.

### Not done (deliberately)

- No `PCA`, no binning, no imputation (see §8).
- No optional caching of the matrix (task 4.8) and no text/TF-IDF block (4.7) —
  both optional and unnecessary for the current scope.
- No recommender (Phase 5) — the matrix exists, nothing ranks with it yet.
