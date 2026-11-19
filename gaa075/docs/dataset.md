# Dataset documentation

Everything in this file was verified against the actual downloaded file on
2026-11-18. It contains no numbers taken from the dataset's description page.
Where the owner's documentation disagrees with the file, the file wins and the
disagreement is recorded.

## 1. Source and attribution

| Item | Value |
|---|---|
| Dataset | Spotify Tracks Dataset |
| Author | MaharshiPandya (Kaggle) |
| Source used | `https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset` |
| Original mirror named in the brief | `https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset` — **unreachable from this machine** (DNS interference; see `docs/dataset-license-evidence.md`) |
| Access date | 2026-11-18 |
| Licence | **ODbL v1.0** for the database; contents © the original authors |

**Required attribution (to be used in the README and the report):**

> Dataset: "Spotify Tracks Dataset" by MaharshiPandya, obtained from Kaggle
> (https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) on
> 2026-11-18. The database is licensed under the Open Database License (ODbL)
> v1.0; copyright in the contents remains with the original authors.

## 2. Licence and usage conditions

Verified from Kaggle's metadata (`licenseName`) and the dataset page's schema.org
`license` entry, which points at `http://opendatacommons.org/licenses/odbl/1.0/`.
The ODbL 1.0 text was downloaded and read. Obligations that apply here:

- **§4.2 Notices** — attribution must be preserved and displayed wherever the
  database or results based on it are published. The block quote above satisfies
  this for the README and the report.
- **§4.4 Share alike** — a publicly conveyed *Derivative Database* must be
  offered under ODbL (or a compatible licence). Analysing the data and
  publishing code/results is not the same act as conveying a derivative
  database, but this project will not publish a redistributable copy of the data
  at all, which avoids the question entirely.
- **§4.6** — no technological measures may restrict the licensed rights.

**Project rules that follow:**

1. The dataset stays **local**. `data/*.csv` and `data/*.zip` are excluded by
   `.gitignore` (verified with `git check-ignore`).
2. Nothing derived from the data is redistributed; only code, documentation and
   aggregate findings are versioned.
3. The attribution above appears in the README and in the report's references.

## 3. Retrieval instructions (reproducible)

```powershell
# ~8.2 MB compressed, no login required
curl.exe -L -o data/spotify-tracks-dataset.zip `
  "https://www.kaggle.com/api/v1/datasets/download/maharshipandya/-spotify-tracks-dataset"
# verify: 8,571,542 bytes
# sha256 zip : BE5216C0C811283C5DB3E3AF83CFFA59873CC5FA375222F3280782776E289176
Expand-Archive data/spotify-tracks-dataset.zip -DestinationPath data
# produces data/dataset.csv, 20,118,244 bytes
# sha256 csv : b202fa49909b2d5cef71a04b1d21243cfeb36414535f2ca9272aa646721177bd
```

Verified at download time:

| Check | Result |
|---|---|
| HTTP status | `200`, `application/zip` |
| Bytes downloaded | `8,571,542` — exact match to `Content-Range` |
| ZIP entries | `1` (`dataset.csv`), method 8 (deflate) |
| CRC integrity (`zipfile.testzip()`) | **PASS** (entry CRC `B8E1F807`) |
| Extracted CSV size | `20,118,244` bytes — matches the ZIP's uncompressed size |

## 4. Verified schema

- **Rows:** 114,000
- **Columns:** 21 total — **20 real columns plus one unnamed integer index**
  (column 0, named `Unnamed: 0` by pandas, values exactly `0…113,999`)
- **Memory as loaded:** 49.7 MB

| # | Column | dtype | Verified range / cardinality | Decision |
|---|---|---|---|---|
| 0 | `Unnamed: 0` | int64 | `0…113,999`, a duplicate row index | **Drop** — redundant |
| 1 | `track_id` | object | 22-char Spotify IDs, 89,741 unique | **Identity key**, not similarity |
| 2 | `artists` | object | 31,437 unique; 30,075 rows (26.38%) list several artists separated by `;` | Display + optional **preference filter/boost** only |
| 3 | `album_name` | object | 46,589 unique | Display only — too sparse for similarity |
| 4 | `track_name` | object | free text | Display only |
| 5 | `popularity` | int64 | `0…100`, mean 33.24 | **Exclude** from similarity (popularity, not content) |
| 6 | `duration_ms` | int64 | `0…5,237,295` (max ≈ 87 min) | Include, **log-scaled** |
| 7 | `explicit` | bool | 9,747 true | **Exclude** — binary, weak, biasing |
| 8 | `danceability` | float64 | `0.000…0.985` | **Include** |
| 9 | `energy` | float64 | `0.000…1.000` | **Include** |
| 10 | `key` | int64 | 12 values `0…11`, no `-1` present | **Exclude** — circular, weak |
| 11 | `loudness` | float64 | `-49.531…4.532` dB (90 rows > 0 dB) | **Include** (scaled) |
| 12 | `mode` | int64 | only `0`/`1` (63.8% major) | **Exclude** — binary, weak |
| 13 | `speechiness` | float64 | `0.000…0.965` | **Include** |
| 14 | `acousticness` | float64 | `0.000…0.996` | **Include** |
| 15 | `instrumentalness` | float64 | `0.000…1.000` (median ≈ 0.00004) | **Include** |
| 16 | `liveness` | float64 | `0.000…1.000` | **Include** |
| 17 | `valence` | float64 | `0.000…0.995` | **Include** |
| 18 | `tempo` | float64 | `0.000…243.372` BPM, 157 rows at 0 | **Include**, but 0 is treated as missing |
| 19 | `time_signature` | int64 | `{0:163, 1:973, 3:9195, 4:101843, 5:1826}` | **Exclude** — one value covers 89% |
| 20 | `track_genre` | object | **114** genres, exactly 1,000 tracks each | **Include** as weighted one-hot |

**Documented vs verified:** the owner's description claims *125 genres*; the file
contains **114**. The description's 18 core fields all exist. `key = -1`
("no key detected") is documented but **never occurs** in this file.

## 5. Data-quality findings (all counted, not estimated)

| Finding | Count | Notes |
|---|---|---|
| Missing values | **3 cells** in **1 row** (0.0009%) | `artists`, `album_name`, `track_name` are NaN in row index `65900` (`track_id 1kR4gIb7nGxHPI3D2ifs59`, genre `k-pop`) — and that same row also has `duration_ms == 0`. A junk row. |
| Exact duplicate rows | **0** | |
| Rows sharing a `track_id` with an earlier row | **24,259** | 40,900 rows involved in total |
| Distinct `track_id` values | **89,741** | i.e. the catalogue is 89,741 unique tracks spread over 114,000 rows |
| Repeated ids spanning **more than one genre** | **16,299** of 16,641 | The same song is listed in several genre buckets |
| Repeated ids with identical feature values | **15,921** | True cross-genre duplicates |
| Repeated ids with at least one differing feature | **720** | Same id, same artists, same duration, but a feature differs |
| `duration_ms == 0` | **1** | The junk row above |
| `duration_ms > 1 hour` | **16** | DJ continuous mixes, sleep/white-noise tracks, a 1h remastered guitar piece |
| `tempo == 0` | **157** | All 157 also have `danceability == 0.0` exactly, and 1 has `energy == 0`. Almost all are `ambient`/classical pieces — a known artefact where the audio-analysis values are absent but stored as zero instead of null |
| `danceability == 0.0` exactly | **157** | Identical set to `tempo == 0` |
| `loudness > 0 dB` | **90** | Max `+4.532` dB |
| `time_signature == 0` (undefined metre) | **163** | |
| Out-of-range unit features | **0** | All seven 0–1 features are inside `[0, 1]` |
| Empty/whitespace strings in text fields | **0** | |
| `track_id` length | **22 characters for all 114,000 rows** | Consistent |

**Genre balance:** exactly **1,000 tracks per genre × 114 genres = 114,000**. The
dataset is perfectly balanced *by design*, not by nature. That has two
consequences: (a) accuracy-style metrics would be artificially easy to inflate if
the genre labels were ever used as a target, and (b) genre is encoded as a
single label per row even though ~18% of the rows share a `track_id` that appears
in other genres — so a track's "genre" here is a bucket assignment, not a
property of the song.

## 6. Feature selection for the content-based recommender

Verified columns only. Nothing here is assumed.

**Similarity vector (verified, 9 numeric + 1 categorical block):**

1. `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`,
   `liveness`, `valence` — already on a comparable 0–1 scale.
2. `loudness` — dB, roughly `-50…+5`; must be scaled.
3. `tempo` — BPM, `0…243` with 157 zeros that are *missing*, not real; zeros
   must be treated as missing before scaling, otherwise they become the minimum.
4. `duration_ms` — heavy right tail (16 tracks over an hour); log-transform
   before scaling.
5. `track_genre` — one-hot over 114 categories, with a **weight** applied to the
   block so a 114-dimension one-hot does not swamp the 9 numeric dimensions.

**Excluded, with reasons:** `Unnamed: 0` (duplicate index); `track_id`,
`track_name`, `album_name`, `artists` (identity/display — `album_name` and
`artists` are far too sparse to encode: 46,589 and 31,437 distinct values);
`popularity` (a behavioural measure, not content; would bake in popularity bias);
`explicit` (binary, weak, potentially biasing); `key` (circular: 11 and 0 are
adjacent, so naive scaling is wrong); `mode` (binary, weak); `time_signature`
(89% a single value).

**Working catalogue implication:** because 24,259 rows repeat a `track_id`, the
recommender must collapse rows to one entry per track before ranking (dedupe by
`track_id`), otherwise the same song would fill several recommendation slots.
Genre for such a track is multi-valued; storing the first or the most specific is
a decision for the preprocessing task, not for this document.

## 7. Known limitations

- **Zero-encoded missingness.** 157 rows carry `tempo = 0` and
  `danceability = 0`, which are not plausible musical values but zeros. Any
  scaling that includes them as real values distorts the feature space.
- **Cross-genre duplication.** 16,299 repeated ids across genres mean the
  catalogue is not 114,000 distinct songs; treating rows as songs would inflate
  the apparent catalogue by ~27%.
- **Synthetic balance.** 1,000 tracks per genre is a sampling artefact of how the
  dataset was assembled, so genre frequencies say nothing about real music.
- **No user data.** There are no ratings, plays or user–item interactions, so no
  collaborative signal and no relevance labels exist. Evaluation therefore
  cannot use precision/recall against ground truth and must rely on structural
  checks and qualitative inspection (planned in Phase 6).
- **Popularity bias.** `popularity` exists but is not used for similarity; if it
  were used for re-ranking, it would amplify already-popular tracks.
- **Audio-feature provenance.** The features are provided by the dataset author
  (derived from Spotify's audio analysis); their accuracy for any individual
  track is not guaranteed and cannot be re-verified here.
- **Sleep/white-noise and DJ-mix tracks** (~16 very long items) are legitimate
  rows but behave very differently from songs; they will produce odd but honest
  neighbours unless filtered.

## 8. What is still not done (deliberately)

No loader function has been written yet — task 2.3 (`load_raw_data()` in
`preprocessing.py`) is **not** complete, because the current work order forbids
modifying `preprocessing.py`. The verified facts above are ready for it, and the
file's first column being a redundant index is the one non-obvious detail that
loader must handle (`index_col=0`, or drop column 0).
