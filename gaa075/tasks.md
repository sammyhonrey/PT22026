# TASK.md — Execution Plan for the Advanced Music Recommendation System

**Derived from:** `Music_Recommendation_System_Project_Brief.md`
**Project type:** Individual university ML project (mid-term, intermediate level)
**Primary language / framework:** Python + Streamlit
**Status:** Planning only. **No task is started.** Every checkbox is unchecked by design.

> **Filename note:** the brief asked for `TASK.md`; you asked for `tasks.md`, so this file is
> `tasks.md`. Rename it if your instructor expects the exact briefed name — nothing else depends on it.

---

## How to use this file

- **Task IDs** (`1.3`, `5.2`, …) are stable. Reference them when you ask me to start work
  ("start phase 2" or "do tasks 3.1–3.4").
- **`R` = Required by the assignment.** Must be finished before submission.
  **`O` = Optional improvement.** Only after every `R` task is done; safe to skip.
- **Dependencies** are stated per phase, and per task where the order is not obvious. Never
  start a task whose dependency is unchecked.
- **`Done when:`** is the acceptance criterion. A task is complete only when that statement is
  literally true — not "mostly working".
- **⏸ CHECKPOINT** blocks are for *you*: run something, look at it, confirm before we move on.
- **Effort** is planning guidance only (`S` ≈ under an hour, `M` ≈ half a day, `L` ≈ a day+),
  not a promise.
- **No results, metrics, screenshots, or dataset facts may be written before they exist.**
  Anything numeric in the report must come from a command that was actually run.

---

## 1. What is being built (analysis summary)

A Streamlit app where a user picks a reference song (and optional genre/artist preferences) and
receives a ranked list of 5–10 genuinely computed similar songs, using **content-based filtering
with cosine similarity** over song metadata and audio features that *actually exist* in the chosen
dataset. Scores are shown with an honest explanation, and the app, tests, README, report, and
demo are all deliverable artifacts.

**In scope (required):** dataset verification → preprocessing → content-based model → cosine
similarity ranking → preference-weighted reranking → evaluation → tests → Streamlit UI →
documentation → GitHub PR → ≥10-page PDF report → 3–5 min demo.

**Explicitly out of scope unless a written, justified change is approved:** collaborative
filtering, hybrid models, deep learning recommenders, user-account/history persistence,
"learning user taste" claims, live Spotify API integration.

### Deliverables checklist (the only things that get graded)

| # | Deliverable | Covered by |
|---|---|---|
| D1 | Working Streamlit application | Phase 8 |
| D2 | Recommendation algorithm + explanation | Phases 4–5 |
| D3 | Test suite and evaluation evidence | Phases 6–7 |
| D4 | `README.md`, `requirements.txt`, `.gitignore` | Phases 0, 9 |
| D5 | Dataset source + obtain instructions | Phases 2–3, 9 |
| D6 | Git history and Pull Request in the correct folder | Phase 10 |
| D7 | PDF report ≥10 pages | Phase 11 |
| D8 | 3–5 minute demonstration | Phase 12 |

---

## 2. Assumptions

Recorded instead of inventing requirements, per the brief. Correct any of these before Phase 1 ends.

- **A1** — The Spotify Tracks Dataset (Hugging Face, `maharshipandya/spotify-tracks-dataset`)
  will be used, subject to verification in Phase 2. If verification fails, Phase 2 includes a
  fallback decision point rather than a silent swap.
- **A2** — The dataset contains both metadata (track, artist, genre) and numeric audio features.
  This is an assumption to be *checked*, not a fact. Feature lists stay provisional until 2.4.
- **A3** — The dataset is large enough that it should **not** be committed to Git; it will be
  downloaded locally and fetched via instructions / small script.
- **A4** — Single-user, stateless app: no login, no stored history, no feedback learning.
  Personalization = preferences chosen in the current session only.
- **A5** — The repo folder `gaa075` (current working directory) is the assigned student folder,
  because it matches the "first three letters of first name + last three digits of student ID"
  convention. **This is a guess and must be confirmed (Q1).**
- **A6** — Content-based filtering with cosine similarity satisfies the "machine-learning
  algorithm" requirement, since preprocessing + vector representation + similarity ranking is a
  standard ML retrieval approach. The justification text belongs in the README and report.
- **A7** — Development happens on a branch and is merged by PR; `main` is never pushed to directly.
- **A8** — **Git remote policy (set by the student, 18 Nov 2026):** commits are made locally by the
  agent as each task finishes, but **nothing is pushed to GitHub — to any remote — unless the
  student explicitly asks for it in that turn.** Remote configuration (`origin` = the fork
  `qsymnswr870-pixel/PT22026`, `upstream` = `xahiru/PT22026`) may be prepared in advance; pushing
  may not. Task 10.4 (open the PR) and any push therefore require a fresh instruction.

---

## 3. Open questions — decisions needed from you

Answer these in one line each; they unblock the marked tasks. Unanswered questions do **not** block
Phases 2–7, but they do block Phase 9 (submission) and Phase 10 (report front matter).

- **Q1 (blocks 9.1, 9.2, 10.2)** — Is `gaa075` your assigned folder in
  `https://github.com/xahiru/PT22026`? Confirm from the repo/TA, not from this guess.
  **Answered 17 Nov 2026 (from the repository itself):** yes — the folder `gaa075/` is committed in
  the shared repository alongside 36 student folders, and a per-student branch `origin/gaa075`
  already exists. Assumption A5 confirmed.
- **Q2 (blocks 10.4)** — Do you already have write access to the shared repo (or a fork + PR
  workflow)? Which one is expected by the TAs?
  **Answered 17 Nov 2026:** no write access. `git push` was rejected with
  `403 Permission to xahiru/PT22026.git denied to qsymnswr870-pixel`. A fork + Pull Request (or
  collaborator access granted by the TA) is required before anything can be pushed. **Blocking.**
- **Q3 (blocks 11.3)** — Full name, student ID, course/module code, and submission date for the
  report title page.
- **Q4 (blocks 2.2)** — Any constraint on downloading datasets (offline lab machines, size limits,
  university network)? If the Hugging Face download is blocked, do we fall back to a local CSV?
- **Q5 (blocks 8.3)** — Do you want the app to bundle a small demo subset of the data (a few
  hundred rows, committed) so it runs without a download? Default: **no** — commit instructions
  only, keep the repo clean.
- **Q6 (blocks 9.1)** — Do you have a preferred language for the app/UI text and the report
  (English assumed)?
- **Q7 (blocks 11.12)** — Is a submitted `report/report.pdf` inside the repo expected, or is the
  PDF uploaded separately to the LMS? Default: commit the PDF under `report/`.

---

## 4. Milestones

| Milestone | Meaning | Exit gate |
|---|---|---|
| **M1 — Foundation** | Repo, env, dataset verified, EDA recorded | Checkpoint C1 (a documented dataset schema) |
| **M2 — Data ready** | Clean dataframe + feature matrix build reproducibly | Checkpoint C2 (preprocessing runs end to end) |
| **M3 — Recommender works** | Ranked, de-duplicated, preference-aware recommendations + scores | Checkpoint C3 (CLI demo of recommendations) |
| **M4 — Evidence** | Tests green + honest evaluation numbers produced | Checkpoint C4 (test + eval output saved) |
| **M5 — Product** | Streamlit app passes all error cases and looks clean | Checkpoint C5 (manual UI walkthrough) |
| **M6 — Delivered** | README, report, demo, PR all complete | Checkpoint C6 (final submission audit) |

---

## Phase 0 — Project setup and ground rules
**Goal:** A working Python environment and repo skeleton so nothing built later is wasted.
**Dependencies:** none. **Effort:** S–M.

- [x] **0.1 — Confirm the scope boundary** (`R`) — Read sections 5 and 13 of the brief with me and
  confirm: content-based only, no collaborative/hybrid work in the required path.
  *Done when:* you have said "confirmed" or listed adjustments. Unblocks all later phases.
- [x] **0.2 — Verify Python toolchain** (`R`) — Check `python --version` (3.10+ preferred), that
  `pip` works, and that a virtual environment can be created.
  *Done when:* version output is recorded and a `.venv` exists without errors.
- [x] **0.3 — Create the repo skeleton** (`R`) — `app.py`, `recommender.py`, `preprocessing.py`,
  `evaluation.py`, `data/.gitkeep`, `assets/`, `tests/`, `report/`, per brief section 11. Use
  placeholder docstrings only — no logic yet.
  *Done when:* the tree matches the brief (or a deviation is documented in the README).
- [x] **0.4 — Write `.gitignore`** (`R`) — Exclude `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`,
  `data/*.csv` (and dataset archive types), `.env`, OS junk files, `report/*.aux`-style scratch files.
  *Done when:* `git status` shows no dataset, venv, or cache files as untracked.
- [x] **0.5 — Repository and branch setup** (`R`) — **Done 17 Nov 2026, and the plan was
  corrected:** a `git init` would have been wrong. The shared course repository `xahiru/PT22026`
  is already cloned at `C:\professionel training\PT22026`, the student folder `gaa075/` already
  exists in it, and the repository uses **one branch per student** (`origin/gaa075`), not a single
  shared `main`. Work therefore happens on the local branch `gaa075`, which tracks
  `origin/gaa075`; `main` is left untouched. Pushing is currently blocked (see Q2), so commits are
  local until remote access is resolved.
  *Done when:* `git branch --show-current` shows the `gaa075` branch and `main` has no new commits.
- [ ] **0.6 — Create `requirements.txt`** (`R`) — Pin `streamlit`, `pandas`, `numpy`,
  `scikit-learn`, `matplotlib`, `pytest` (version pins added once 0.2 confirms what installs).
  *Done when:* a fresh `pip install -r requirements.txt` succeeds in the venv.
- [ ] **0.7 — Decide the feature-branch/commit convention** (`O`) — e.g. `feat:`, `fix:`, `docs:`
  prefixes and one topic per commit, to produce a meaningful history for the grading rubric.
  *Done when:* convention noted in `README.md` or `CONTRIBUTING.md` (one short paragraph is enough).

**Phase 0 exit criteria:** environment installs cleanly, repo tree matches the brief, first commit
exists, and nothing unignoreable is staged.

---

## Phase 1 — Requirements traceability
**Goal:** Turn the assignment's grading criteria into a checklist we can audit at the end, so no
mandatory deliverable is missed.
**Dependencies:** Phase 0 (0.1). **Effort:** S.

- [x] **1.1 — Build the requirement matrix** (`R`) — A table mapping each assignment requirement
  (brief §12) to the file/artifact that satisfies it and the task ID that produces it — written to
  `docs/requirements-matrix.md` (I1–I7, F1–F6, G1–G6, R1–R10, D1–D6, O1–O2).
  *Done when:* every §12 bullet has a row and no row is empty.
- [x] **1.2 — Map tasks to the mark distribution** (`R`) — Note which phases feed the 30/20/30/10/10
  split (functionality, code+repo, report, depth, demo) so effort lands where marks are.
  *Done when:* the mapping is recorded and you have reviewed it. **Done 17 Nov 2026:** recorded in
  `docs/requirements-matrix.md` under "Mark distribution → where the effort goes".

**Phase 1 exit criteria:** the matrix exists and a reader can trace any mark back to real work.

---

## Phase 2 — Dataset acquisition, licensing, and verification
**Goal:** Prove the dataset is usable *and* that its schema matches what the plan assumes, before
writing model code. Nothing in Phase 4 may reference a column that is not verified here.
**Dependencies:** Phase 0 (0.2, 0.6). **Effort:** M–L.

- [x] **2.1 — Access and license check** (`R`) — Open the candidate dataset page
  (`https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset`), read the dataset card,
  and record: license, attribution requirements, commercial/redistribution terms, source-origin
  notes (Spotify-derived), and the exact access date.
  *Done when:* a `report/` or `docs/dataset.md` note lists license + attribution text verbatim-ish
  and the access date. If the license forbids redistribution, that is written down too.
  **Done 2026-11-18:** the Hugging Face page is **unreachable from this machine** (DNS interference:
  443 closed, `157.240.3.50` returned for `datasets-server.huggingface.co`; PyPI responds in 0.4 s as
  a control). With the student's approval the **Kaggle original** was used instead. Licence verified
  from Kaggle metadata (`licenseName`) and the page's schema.org `license` URL:
  **ODbL v1.0**, contents © original authors. Attribution block written in `docs/dataset.md` §1;
  full evidence in `docs/dataset-license-evidence.md`.
- [x] **2.2 — Download and record the retrieval method** (`R`) — Fetch the CSV (dataset viewer /
  `huggingface_hub` / direct link) into `data/`, and record the exact command or link used.
  *Done when:* the file exists locally and the retrieval command is reproducible from the note.
  If the download is blocked (Q4), record the failure and switch to the agreed fallback here.
  **Done 2026-11-18:** downloaded from Kaggle with the student's approval (8.2 MB approved).
  `curl` fetched **8,571,542 bytes**, an exact match to the server's `Content-Range`. ZIP CRC test
  passed (`zipfile.testzip()` → PASS, entry CRC `B8E1F807`), one entry `dataset.csv`, extracted to
  **20,118,244 bytes**. Checksums and the reproducible command are in `docs/dataset.md` §3. Both the
  zip and the CSV are confirmed ignored by Git (`git check-ignore`).
- [x] **2.3 — Add a small, documented loader** (`R`) — `preprocessing.load_raw_data()` that reads
  the local CSV, raises a clear error if the file is missing, and does not hardcode absolute paths.
  *Done when:* calling it prints a shape and succeeds from a clean checkout with data present, and
  gives a readable message with data absent.
  **Done 2026-11-18:** implemented in `preprocessing.py`. Verified: default load returns
  **(114,000, 20)** with the redundant index consumed as the DataFrame index (no `Unnamed: 0`
  column); a missing path raises `FileNotFoundError` quoting the expected path and the download
  commands; a scan of the module finds **0** hardcoded absolute paths (the default resolves
  relative to `__file__`). It deliberately does not clean or transform anything.
- [x] **2.4 — Inspect the real schema** (`R`) — Record column names, dtypes, row count, memory size,
  and a `head()` sample — **as printed by the actual file**, not from memory or the dataset card.
  *Done when:* the schema table is pasted into `docs/dataset.md` from real output.
  **Done 2026-11-18:** **114,000 rows × 21 columns** (20 real + one redundant `Unnamed: 0` index),
  49.7 MB in memory. All 21 columns, dtypes and verified ranges are tabulated in
  `docs/dataset.md` §4. Notable: the owner's description claims **125 genres, the file has 114**.
- [x] **2.5 — Data-quality audit** (`R`) — Quantify per column: missing-value counts/percentages,
  duplicates (full-row and by track identifier), implausible ranges (e.g. audio features outside
  0–1 if that is their documented range), constant-value columns, whitespace/encoding issues, and
  class balance of `genre`.
  *Done when:* each finding has a number and a planned action (use / clean / exclude / keep-and-flag).
  **Done 2026-11-18:** every finding has a count and an action in `docs/dataset.md` §5 — 3 missing
  cells in 1 junk row; 0 exact duplicate rows but 24,259 repeated `track_id` rows (89,741 unique
  tracks), 16,299 of the repeated ids spanning several genres; 157 rows with `tempo == 0` **and**
  `danceability == 0.0`; 1 row with `duration_ms == 0`; 16 tracks over 1 hour; 90 rows above 0 dB;
  163 rows with `time_signature == 0`; 0 out-of-range unit features; 114 genres at exactly 1,000
  tracks each.
- [x] **2.6 — Confirm the feature list** (`R`) — Final table: `feature → available? → used or
  excluded → reason`. Covers genre, artist, track name, danceability, energy, acousticness,
  instrumentalness, valence, tempo, loudness, speechiness, popularity, and anything else found.
  Anything listed in brief §5 but **absent** must be marked absent — do not silently pretend.
  *Done when:* the table is complete and every "excluded" has a one-line justification.
  **Done 2026-11-18:** decision table with reasons in `docs/dataset.md` §4 and §6. **All** features
  named in the brief actually exist in the file (danceability, energy, acousticness,
  instrumentalness, valence, tempo, loudness, speechiness, plus liveness, duration_ms, genre,
  artist). Included in the similarity vector: 7 unit-scale features + loudness + tempo +
  log-duration + weighted one-hot genre. Excluded with reasons: popularity, explicit, key, mode,
  time_signature, track_id/track_name/album_name/artists, and the redundant index column.
- [x] **2.7 — Record dataset facts for reuse** (`R`) — Save the canonical numbers (rows, columns,
  genres, license, access date, file size, checksum if cheap) into `docs/dataset.md` so the report
  quotes one source of truth.
  *Done when:* the README/report sections can copy them without re-running anything.
  **Done 2026-11-18:** `docs/dataset.md` holds the canonical numbers, both SHA256 checksums, the
  licence line, the attribution block and the access date. The report can quote them verbatim.
- [ ] **2.8 — Sanity-check genre/artist distributions** (`O`) — Quick counts: how many genres,
  long-tail shape, how many tracks per artist. Useful later for the limitations section.
  *Done when:* a short numbers-only note exists (no plots required yet).

**⏸ CHECKPOINT C1 — Dataset verified.** Show me the schema table, the license line, and the
feature-decision table. Confirm: "these are the fields we are using." Only then does Phase 3 start.

**Phase 2 exit criteria:** dataset is legal to use, loads through one documented function, and the
used-feature list is evidence-based.

---

## Phase 3 — Exploratory data analysis (EDA)
**Goal:** Understand the data well enough to choose preprocessing, and to have honest figures for
the report.
**Dependencies:** Phase 2 (2.4–2.6), C1. **Effort:** M.

- [ ] **3.1 — Numeric feature summaries** (`R`) — Distribution summaries (min/max/mean/median/
  quantiles, skew) for each used numeric feature; note outliers that would distort distances.
  *Done when:* summary tables are saved from real output.
- [ ] **3.2 — Categorical/missingness summaries** (`R`) — Top genres, top artists, and a
  missing-value heatmap or table.
  *Done when:* figures/tables saved under `assets/` or `report/figures/` with descriptive names.
- [ ] **3.3 — Correlation check** (`R`) — Correlation matrix of numeric features; flag pairs that
  are near-duplicates (common for energy/loudness) and decide whether to keep both.
  *Done when:* the matrix exists and a keep/drop decision per correlated pair is written down.
- [ ] **3.4 — Decide dataset scope for the app** (`R`) — Full catalog vs. a filtered subset
  (e.g. minimum popularity, removing severely unbalanced ultra-long-tail genres) chosen for
  speed and quality; state the tradeoff and the exact rule used.
  *Done when:* the rule is a single documented function/config value, and the resulting row count
  is recorded.
- [ ] **3.5 — Write the EDA summary** (`R`) — 1 page of findings: shape, quality issues, notable
  distributions, and the implications for preprocessing.
  *Done when:* it can be pasted into report sections 5 and 7 without edits.
- [ ] **3.6 — Similarity spot-check plan** (`O`) — Before building the model, list 3–5 reference
  songs and what *you* would consider a sensible neighbour, so results can be judged qualitatively
  later rather than only numerically.
  *Done when:* the list exists with your expectation per song.

**Phase 3 exit criteria:** every preprocessing choice in Phase 4 can point at an EDA finding.

---

## Phase 4 — Preprocessing and feature representation
**Goal:** One reproducible function chain that turns the raw CSV into (a) a clean display dataframe
and (b) a numeric feature matrix ready for cosine similarity.
**Dependencies:** Phase 3, C1. **Effort:** L.

- [x] **4.1 — Cleaning pipeline** (`R`) — Handle missing values (drop vs. impute, decided and
  justified per field), remove duplicate tracks (define which identity counts as "the same song"),
  strip whitespace, coerce types, and drop rows that cannot be represented.
  *Done when:* input rows → output rows is a printed, recorded number and every removal has a reason.
  **Done 2026-11-18:** implemented in `preprocessing.drop_junk_rows()`, `zero_coded_mask()`,
  `build_catalogue()`. Recorded arithmetic: **114,000 − 1 junk − 24,259 duplicates − 157
  zero-coded = 89,583 tracks**. No imputation of audio features was performed (approved decision);
  the zero-coded rows are removed from the catalogue, not from the CSV.
- [x] **4.2 — Choose the track identity** (`R`) — Decide what makes two rows "the same
  recommendation" (track+artist string vs. a dataset track id vs. normalized name) so that 4.1's
  dedup and Phase 4.5's exclusion both use the same key.
  *Done when:* the key is implemented in one helper and documented.
  **Done 2026-11-18:** identity is `track_id` (`ID_COLUMN`). De-dup keeps the **first occurrence in
  file order** (approved Decision 2); every stage uses the same key, and `FeatureMatrix.row_index()`
  resolves a `track_id` to its matrix row.
- [x] **4.3 — Scale numeric features** (`R`) — `StandardScaler` or `MinMaxScaler`, fitted on the
  full catalog once and reused; justify the choice; describe what happens to sklearn's sparse
  output if relevant.
  *Done when:* matrix stats (shape, mean≈0/std≈1 or 0–1 bounds) verified on real output.
  **Done 2026-11-18:** `StandardScaler` fitted once in `build_numeric_block()`, then rows L2
  normalised. Verified on real output: scaler mean equals catalogue column means; the numeric block
  is `(89,583 × 9)` and every final matrix row has norm `1.000000000000`. `StandardScaler` is dense
  (9 columns, ~6.4 MB) and is converted to sparse on assembly.
- [x] **4.4 — Encode categorical features** (`R`) — One-hot for genre (and artist if used), with
  rarity handling (min frequency) so the matrix stays sane; decide and justify weights so numeric
  and categorical blocks do not silently dominate.
  *Done when:* block sizes (how many dimensions each block contributes) are printed and the
  weighting rationale is written down.
  **Done 2026-11-18:** multi-label one-hot via `MultiLabelBinarizer` (114 genres → 114 columns),
  row-normalised; no rarity cut because every genre has ≥862 tracks. Block sizes: **9 numeric +
  114 genre = 123**. Weights: `√α = √0.7` and `√β = √0.3` applied inside the matrix so
  `cosine = 0.7·S_numeric + 0.3·S_genre`; rationale in `docs/preprocessing-design.md` §6, and the
  values remain testable at Checkpoint C3.
- [x] **4.5 — Assemble the feature matrix** (`R`) — Combine blocks (`scipy.sparse.hstack` or
  equivalent), keep a row-aligned index mapping back to track identity, and persist the build as a
  cached function so the app does not rebuild on every keystroke.
  *Done when:* one call yields matrix + aligned dataframe + identity lookup, and a repeat call
  reuses the cache.
  **Done 2026-11-18:** `build_feature_matrix()` returns a `FeatureMatrix` holding the weighted
  `csr` matrix `(89,583 × 123)`, the row-aligned catalogue indexed by `track_id`, both blocks, the
  genre vocabulary and the fitted scaler; 919,639 non-zeros (8.35% dense), ~2.0 s to build. The
  *cache* requirement is met at call level by passing the returned `scaler`/`vocabulary` back in;
  on-disk caching is deferred to optional task 4.8.
- [x] **4.6 — Guard against leakage/drift** (`R`) — Ensure scalers/encoders are fitted consistently
  and that the app path and the evaluation path build the matrix with the *same* code path.
  *Done when:* both paths call the same builder function (no duplicated preprocessing).
  **Done 2026-11-18:** there is exactly one builder, `build_feature_matrix()`; the scaler and
  vocabulary can be re-supplied, so refitting is explicit. The integration test asserts the
  documented counts on the real file, which would fail loudly if the fit drifted.
- [ ] **4.7 — Text features** (`O`) — Only if it demonstrably improves results: TF-IDF over
  track/artist/album names, weighted low. Do not include without measuring (see 5.4).
- [ ] **4.8 — Persist intermediates** (`O`) — Cache the built matrix to disk (e.g. `.npz`) keyed by
  a fingerprint of the cleaning rules, with an ignore rule in `.gitignore`.
  *Done when:* a cold start is measurably faster and a stale cache is detected safely.

**⏸ CHECKPOINT C2 — Preprocessing runs end to end.** I run the pipeline from raw CSV to matrix;
we look at shapes, rows dropped, and a random sample of neighbours from a raw cosine calculation.
Confirm the matrix "looks reasonable" before writing the recommender API.

**Phase 4 exit criteria:** a single documented pipeline produces a matrix whose dimensions and
semantics are explainable in one paragraph.

---

## Phase 5 — Recommender: similarity, ranking, and personalization
**Goal:** `recommend(track, preferences, k)` returns ranked, de-duplicated recommendations with
honest scores.
**Dependencies:** Phase 4, C2. **Effort:** L.

- [x] **5.1 — Cosine similarity over the matrix** (`R`) — Compute similarity between the reference
  vector and all candidates (`sklearn.metrics.pairwise.cosine_similarity`), using efficient
  row-wise dot products rather than a full N×N matrix if memory is a concern.
  *Done when:* scores for a known reference are in [−1, 1] (or [0, 1] for non-negative features)
  and documented as such.
  **Done 2026-11-18:** `Recommender.scores()` computes each block as a matrix–vector product
  (no N×N matrix) and combines `0.9·S_num + 0.1·S_genre`. Scores are clipped to [-1, 1] because
  float arithmetic overshot by 2e-16 in testing; the integration test asserts
  `score.between(0, 1).all()`.
- [x] **5.2 — Ranking and top-k selection** (`R`) — Sort descending, take `k` (default 10,
  user-adjustable 5–10), and return a tidy structure (title, artist, genre, score).
  *Done when:* calling it with `k=5` and `k=10` returns exactly that many rows when available.
  **Done 2026-11-18:** `Recommender.recommend(track_id, k=10)` returns a DataFrame
  (`rank, track_id, track_name, artists, album_name, genres, duration_ms, score, numeric_score,
  genre_score`); `k` validated 1–50. Tests: `k=1/5` exact length, and `k=10` returning all 9
  distinct candidates when the pool is smaller.
- [x] **5.3 — Exclusion and de-duplication** (`R`) — Never return the reference track itself (same
  identity key as 4.2) and never return the same track twice; handle "fewer candidates than k".
  *Done when:* a test proves the reference track is absent and no identity repeats (see 6.3).
  **Done 2026-11-18:** reference row always excluded **and** its duplicate-recording signature is
  pre-seeded into the suppression set, so another `track_id` holding the same recording cannot
  reappear at rank 1 (test: `test_duplicate_of_the_reference_recording_is_not_returned`).
  Approved signature: normalized title + artist + 9 audio features (9 dp) + exact duration; the
  highest-scoring duplicate is kept. Not suppressing different audio/duration variants was verified
  by dedicated tests.
- [x] **5.4 — Preference-weighted reranking** (`R`) — Implement genre and artist preferences as an
  explicit, documented boost (e.g. final = similarity × weight, or similarity + α·match), with the
  weight configurable and defaulting to a neutral value. **State plainly** what it does and does
  not do — it does not learn the user.
  *Done when:* with the same reference song, changing a preference changes the ranking in a way
  that is demonstrated side by side.
  **Done 2026-11-18:** `Preferences.build(genres=…, artists=…)` + `final = 0.80·base + 0.20·pref`;
  genre component = share of requested genres matched (multi-label aware), artist component = share
  of requested artists credited, mean when both are given. With no preferences the ranking and
  scores are byte-identical to the content-based result (asserted). Side-by-side demonstration in
  `analysis/sample_recommendations.py`: *I Ain't Worried* returns *Bad Blood* with no preferences but
  *Crocodile Rock* with a piano or Elton John preference. Unknown genres raise
  `InvalidRequestError`; `Recommender.genres` exposes the vocabulary.
- [x] **5.5 — Similarity explanation layer** (`R`) — Produce a short, true explanation per result
  (e.g. top contributing features, or a plain statement of shared genre/close audio profile).
  Must be derived from real feature values, never invented prose.
  *Done when:* explanations are traceable to a computation, and a comment says so.
  **Done 2026-11-18:** every result carries an `explanation` built from real values: shared genres
  (sorted intersection), the 1–2 audio features with the largest positive `z_ref·z_cand`
  contribution (these products sum exactly to `S_num` because rows are unit-normalised), and any
  matched preferred genre/artist. Deterministic ordering; asserted by
  `test_explanations_are_deterministic`.
- [x] **5.6 — Score semantics and honesty** (`R`) — Decide how scores are displayed (raw cosine vs.
  rescaled to 0–100) and write the caveat text: algorithmic similarity ≠ taste prediction.
  *Done when:* the caveat string exists once in code/config and appears wherever scores appear.
  **Done 2026-11-18:** raw cosine is displayed (never rescaled to a percentage) and the single
  constant `recommender.SCORE_INTERPRETATION_WARNING` carries the caveat. It is attached to every
  returned DataFrame via `attrs["score_warning"]` (empty results included), alongside
  `attrs["preferences_applied"]`, so the UI has one source for the warning.
- [x] **5.7 — Determinism and edge cases** (`R`) — Same inputs → same output order (stable
  tie-breaking); empty candidate pool, unknown track, and all-filtered-out cases return a clear
  signal rather than an exception traceback.
  *Done when:* each edge case is exercised once and behaves as described.
  **Done 2026-11-18:** ordering is score-descending with `track_id` ascending tie-break
  (`np.lexsort`), asserted deterministic. Errors: `UnknownTrackError`, `InvalidRequestError` for
  bad ids, bad `k`, bad weights; empty DataFrame for a single-track catalogue; fewer rows when the
  candidate pool is smaller than `k`.
- [x] **5.8 — CLI smoke demo** (`R`) — A tiny script or `python -m` entry that prints
  recommendations for 3 reference songs without Streamlit, for fast iteration and demo confidence.
  *Done when:* it runs and prints a legible table.
  **Done 2026-11-18:** `analysis/sample_recommendations.py` prints top-5 for six reference profiles
  (rock, hip-hop, classical, sleep, multi-genre, most-multi-genre) plus duplicate diagnostics, with
  no Streamlit involved.

**⏸ CHECKPOINT C3 — Recommender works.** You pick 2–3 songs you know, I run the CLI, and we judge
whether the neighbours are plausibly similar. If they are not, we fix features (Phase 4) *before*
building UI. This is the single most important checkpoint in the plan.

**C3 evidence gathered 2026-11-18 (student-defined scope: weight sensitivity).** Analysis in
`analysis/weight_sensitivity.py`, full results in `docs/weight-sensitivity.md`:
- Decomposition verified numerically: `cosine(weighted matrix) = 0.7·S_num + 0.3·S_genre` to
  **4.4e-16**.
- Over `alpha ∈ {0.5 … 0.9}` on 300 sampled tracks: genre-match in the top-10 is **1.000** at
  0.5/0.5, **1.000** at 0.7/0.3 and **0.839** at 0.9/0.1 — the genre block dominates until its
  weight drops to 0.1. No weight is claimed optimal (no relevance labels exist).
- **Defect found:** de-duplication by `track_id` does not stop the same recording appearing twice
  under different ids — 4,647 title+artist groups / 13,032 rows, and a real rock top-10 contained
  *Follow You* twice. Phase 5 must add an explicit duplicate-suppression rule.
- Reference tracks, methodology and all per-weight top-10 tables are recorded in the doc so the
  analysis is reproducible.

**Phase 5 exit criteria:** recommendations are computed, ranked, de-duplicated, preference-aware,
and explainable.

---

## Phase 6 — Evaluation
**Goal:** Honest, reproducible evidence of behaviour quality — without inventing labels or metrics.
**Dependencies:** Phase 5, C3. **Effort:** L.

- [ ] **6.1 — Decide the evaluation design** (`R`) — Because there is no user-relevance ground
  truth, use defensible proxies only: qualitative neighbour inspection across N reference songs,
  self-consistency checks, and agreement between independent similarity signals (e.g. does the
  cosine ranking agree with genre overlap / a hand-built feature-distance ranking?). Write the
  design *before* computing numbers.
  *Done when:* the design paragraph exists and states explicitly which classic metrics (precision,
  recall, accuracy, NDCG) are **not** computed and why.
- [x] **6.2 — Multi-song qualitative report** (`R`) — For 8–12 varied reference songs, dump top-5
  results with genre/artist and a 1-line human judgement (sensible / questionable / wrong).
  *Done when:* the table is saved as CSV/markdown with real output.
  **Done 2026-11-18:** `report/evaluation_results.md` now contains a *Sampled top-5 lists* section
  for all **17** reference tracks (title, artist, genres, score), and `report/evaluation_results.csv`
  holds the per-reference metrics. Honest gap: the **human judgement column is intentionally
  empty** — only the student can judge whether a neighbour is musically sensible, and inventing
  those labels is forbidden. The tables are ready for that judgment pass.
- [x] **6.3 — Structural correctness checks** (`R`) — Verify across the sample: no self-recommendation,
  no duplicates, no NaN/inf scores, and every returned score within its documented range.
  *Done when:* results table shows 100% pass, and it is implemented as automated tests (6.5).
  **Done 2026-11-18:** measured over 17 references × top-10: **0 self-match violations**, **0
  full-signature duplicate repeats**, 0 non-finite scores, all scores inside [0, 1]. Enforced by
  `tests/test_evaluation.py` (`test_evaluate_all_reports_structural_invariants`) and by the
  recommender suite (`test_reference_is_never_recommended`, duplicate tests, `between(0,1)` checks).
- [x] **6.4 — Preference-effect analysis** (`R`) — Same reference song, preference on vs. off:
  how many results change, and does the change match the documented rule? Include a case where the
  preference is impossible (no matching genre) to show honest degradation.
  *Done when:* before/after tables exist and the explanation matches the implementation.
  **Done 2026-11-18:** three deterministic scenarios over all 17 references (aligned genre, cross
  genre `classical`, artist). Measured: aligned genre changes **8.59/10** items and the #1 in **94%**
  of cases (preference match 0.876); cross genre changes **3.59/10** (match 0.400); artist changes
  **3.82/10** (match 0.394). The "impossible preference" case is the cross-genre scenario: when
  nothing matches, all scores scale by 0.8 and the order is preserved — reported, not hidden.
- [x] **6.5 — Quantitative proxies, only if valid** (`O`) — e.g. mean intra-list genre homogeneity,
  mean pairwise audio-feature distance, or genre-overlap hit rate at k. Label them clearly as
  proxies with a definition, never as precision/recall.
  *Done when:* each metric has a written definition and a computed value from real data.
  **Done 2026-11-18:** definitions and values in `docs/evaluation.md` §2 and §5 — intra-list
  similarity (numeric 0.9143 / genre 0.5354 / combined 0.8764), distinct genres per list (4.59),
  genre consistency (0.806), artist diversity (0.900), catalogue coverage (0.1898%). Each is
  labelled a descriptive proxy; a test asserts no precision/recall/accuracy/NDCG function exists.
- [x] **6.6 — Evaluation artifact** (`R`) — `evaluation.py` runs the above and writes results to a
  saved file (`report/evaluation_results.md` or CSV) plus figures if useful.
  *Done when:* one command regenerates all reported numbers.
  **Done 2026-11-18:** `python evaluation.py` writes `report/evaluation_results.md` (12,087 B) and
  `report/evaluation_results.csv` (17 rows) in one deterministic run; no randomness, no seeds.
- [x] **6.7 — Limitations write-up** (`R`) — Dataset-feature dependence, no taste model, catalog and
  popularity bias, cold-start, low diversity of content-based neighbours, and small-sample caveats.
  *Done when:* it is written and each limitation is true of *this* implementation.
  **Done 2026-11-18:** `docs/evaluation.md` §7 and `report/evaluation_results.md` *Limitations*
  section: no relevance labels, 17 references only, k=10 only, coverage measured over these lists,
  look-alike repeats permitted by the approved rule, ILS sensitive to catalogue size and the
  114-genre encoding, human judgement still outstanding.

**⏸ CHECKPOINT C4 — Evidence exists.** Run the evaluation command; check the pass rate and read
2–3 qualitative rows critically. Confirm the numbers are trustworthy enough to write into the report.

**Phase 6 exit criteria:** every claim in the report's results section traces to a saved artifact.

---

## Phase 7 — Automated tests
**Goal:** A fast test suite that catches the regressions that would break the demo.
**Dependencies:** Phases 4–6. **Effort:** M.

- [x] **7.1 — Test layout and fixtures** (`R`) — `tests/` with a tiny synthetic dataframe fixture
  (5–10 hand-made tracks) so tests run in under a few seconds without the real dataset.
  *Done when:* `pytest -q` passes with zero dataset dependency.
  **Done 2026-11-18:** `tests/conftest.py` plus four test modules; the synthetic fixtures are 3–11
  hand-made rows. 102 tests run in ~21 s on the development machine, and the dataset-dependent
  tests are isolated behind `@requires_dataset`, so the rest of the suite is dataset-free.
- [x] **7.2 — Preprocessing tests** (`R`) — Missing values, duplicates, scaling bounds/shape,
  encoding of unseen categories, and missing-file error message.
  *Done when:* each behaviour has at least one asserting test.
  **Done 2026-11-18:** 20 tests in `tests/test_preprocessing.py` cover junk rows, zero-coded rows,
  dedup semantics, multi-label genre preservation, scaler fitting, zero-variance features, matrix
  dimensions, duration exclusion, unit row norms, weight validation, reproducibility, the loader's
  missing-file message and the dense/sparse type annotations.
- [x] **7.3 — Recommender tests** (`R`) — Reference exclusion, duplicate-free output, k respected,
  score range, deterministic order, unknown-track handling, and empty-pool behaviour.
  *Done when:* all listed cases assert, and a deliberately broken implementation fails the suite.
  **Done 2026-11-18:** 53 tests in `tests/test_recommender.py`. The suite has caught real defects:
  the unclipped score (`1.0000000000000002`) and the `k`-exceeds-candidates assumption; the
  reference-duplicate defect found in Phase 5 is now locked by
  `test_duplicate_of_the_reference_recording_is_not_returned`. Details in `docs/testing.md` §3.
- [x] **7.4 — Preference-logic tests** (`R`) — A matching preference promotes a low-similarity item;
  a neutral preference changes nothing; impossible preferences degrade gracefully.
  *Done when:* the three cases pass.
  **Done 2026-11-18:** `test_genre_preference_changes_the_ranking` (promotion),
  `test_no_preferences_preserves_ranking_and_scores` (neutral changes nothing, byte-identical), and
  `test_unmatched_artist_preference_preserves_order` (impossible preference: order preserved,
  scores scaled by 0.80, `preference_score == 0`).
- [x] **7.5 — Optional real-data integration test** (`O`) — Skipped automatically when the dataset
  is absent (`pytest.mark.skipif`), so CI/fresh clones stay green.
  *Done when:* the suite is green both with and without the data file.
  **Done 2026-11-18:** `@requires_dataset` guards 6 real-data tests across preprocessing,
  recommender and evaluation. Verified green **with** the dataset present; the guard is a
  `skipif` on `pp.DEFAULT_DATA_PATH.is_file()`, so absence skips rather than fails.
- [x] **7.6 — Acceptance criteria traceability** (`R`) — Link each test to the acceptance criteria
  it proves (docstring or README table).
  *Done when:* every "Done when" in Phases 4–6 has a corresponding check.
  **Done 2026-11-18:** `docs/testing.md` contains a criterion → test table covering tasks 4.1–4.6,
  5.1–5.7, 6.1/6.3/6.5, error handling, integration and graceful degradation. Every "Done when" in
  Phases 4–6 maps to at least one test.

**Phase 7 exit criteria:** `pytest -q` is green, fast, and independent of the big CSV.

---

## Phase 8 — Streamlit application
**Goal:** A minimal, uncrowded UI that makes the model usable and demonstrates input → output.
**Dependencies:** Phase 5 + 7, C3. **Effort:** L.

- [x] **8.1 — App skeleton and navigation** (`R`) — Multi-page or single page with clear sections:
  Home, Recommend, About/Model. Cached data/model loading (`st.cache_data` / `st.cache_resource`).
  *Done when:* the app starts with `streamlit run app.py` in a few seconds after cache warm-up.
  **Done 2026-11-18:** sidebar navigation with **Home / Discover / Recommendations / About**;
  `load_engine()` is `@st.cache_resource`, so the matrix is built once (~2 s) and reused. Verified
  live: `python -m streamlit run app.py` served HTTP 200 and `/stcore/health` → `ok`.
- [x] **8.2 — Home / introduction** (`R`) — Project name, one-paragraph problem statement, and a
  short "how recommendations are generated" note.
  *Done when:* a non-technical reader understands the app in under 30 seconds.
  **Done 2026-11-18:** hero reads *"Discover Your Next Favorite Song"* with a one-paragraph
  explanation, three "how it works" steps, and live catalogue facts (track count, genre count,
  feature count) read from the engine rather than hardcoded.
- [x] **8.3 — Song selection UX** (`R`) — Searchable selectbox or text search over "Title — Artist",
  showing the chosen track's identity clearly; handles no-match input.
  *Done when:* a track can be found by typing a partial name.
  **Done 2026-11-18:** case-insensitive title/artist search with a selectbox of up to 40 hits; an
  empty query lists the most popular tracks so the page is never dead. Verified in the UI harness:
  searching "Elton John" populates 40 hits; a nonsense query shows *"No song matches …"*.
- [x] **8.4 — Preference controls** (`R`) — Optional genre multi-select and optional artist field,
  plus a "how strongly preferences matter" slider; optional fields are visibly marked optional
  and are not applied unless set.
  *Done when:* leaving them empty reproduces pure similarity results exactly (matches 7.4).
  **Done 2026-11-18:** a *Personalize* section with a genre multi-select, an artist text field and a
  5/10 result radio, all labelled optional. When empty, `Preferences.build()` yields an empty
  preference object and the engine returns pure-similarity results (behaviour already asserted by
  `test_no_preferences_preserves_ranking_and_scores`). The brief's "strength slider" is replaced by
  the fixed approved 0.80/0.20 formula — deliberately, so the UI cannot drift from the engine.
- [x] **8.5 — Results presentation** (`R`) — 5–10 recommendations as clean cards/rows: rank, title,
  artist, genre, score, optional short explanation; score caveat text shown.
  *Done when:* results render without clutter and the numbers displayed match the CLI output.
  **Done 2026-11-18:** recommendation cards show rank, generated cover, title, artists, genre
  chips, duration, raw cosine score with a subtle bar, and the engine's `explanation` string. The
  warning constant is rendered once per list. Verified through Streamlit's `AppTest`: 10 cards,
  explanations and warning present; the Recommendations page shows the same stored list.
- [x] **8.6 — Error handling in the UI** (`R`) — Friendly, specific messages for: empty/invalid
  input, song not found, dataset missing, missing values in used fields, zero results after
  filtering, and an unexpected-exception catch-all.
  *Done when:* each of the six cases is triggered manually and shows a helpful message, not a traceback.
  **Done 2026-11-18:** missing dataset → instruction panel with the download commands; no search
  hits → warning naming the query; `UnknownTrackError` → "track is no longer in the catalogue";
  `InvalidRequestError` → message with the reason; empty result list → info panel; unmatched
  preferences → explicit "none matched your preferences" notice; any other exception → generic
  error instead of a traceback.
- [x] **8.7 — Visual design pass** (`R`) — Neutral background, restrained accent color, consistent
  spacing/typography, custom CSS limited to a small, commented block, no decorative animation.
  *Done when:* screenshot reviewed against the brief's §8 list.
  **Done 2026-11-18:** dark charcoal theme (`#0A0C0B`), green accent (`#35E27B`), rounded 16 px
  cards, subtle 1 px borders, hover lift on cards, one CSS block at the top of `app.py`. No
  animation beyond a 140 ms hover transition, no decorative images, no fake data. A screenshot was
  **not** possible in this environment (no browser automation); the interface was verified
  structurally through `AppTest` and by running the live server — see the checkpoint report.
- [x] **8.8 — About / Model section** (`R`) — Content-based filtering explained, preprocessing and
  cosine similarity explained in plain language, how preferences affect ranking, and limitations.
  *Done when:* the section is accurate to the code (no aspirational claims).
  **Done 2026-11-18:** About page explains the 0.90/0.10 content weights with the reason (the C3
  study), the 0.80/0.20 preference formula, why the score is not a percentage, the ODbL attribution,
  and the real limitations (no user data, 157 unreachable tracks, low diversity, popularity bias,
  look-alike duplicates).
- [ ] **8.9 — Responsive check** (`O`) — Narrow-window / mobile-width layout sanity; two-column →
  stacked.
  *Done when:* checked at a narrow viewport.
- [ ] **8.10 — Demo-ability polish** (`O`) — Fast first paint with cached model, sensible defaults,
  and a "surprise me" example preset to speed up the live demonstration.
  *Done when:* a cold start to first recommendation takes an acceptable time in the demo environment.

**⏸ CHECKPOINT C5 — Manual UI walkthrough.** We run the app together: pick a song, add a genre
preference, force an error, open About. You confirm it is clean, honest, and demo-ready.

---

## Phase 9 — Single-page interface and audio preview player (added 2026-11-18)

**Why:** the student reviewed Phase 8 and asked for one simple page instead of four, using the
official project name, plus legitimate 30-second previews with playback controls.

- [x] **9.1 — Official name** (`R`) — "Music Recommendation System" everywhere; no invented brand.
  **Done:** app header, `st.set_page_config` title, README and this plan. The earlier "SONORA"
  brand was removed from `app.py`; no occurrence remains.
- [x] **9.2 — Single page, no navigation** (`R`) — Header, search, selected song, optional
  preferences, 5/10 selector, one button, results — all on one page, no sidebar.
  **Done:** `app.py` renders sections 1–5 in that order; the only `st.radio` is the 5/10 chooser
  (asserted by `test_single_page_renders_without_navigation`).
- [x] **9.3 — Selected song panel** (`R`) — title, artists, genres, artwork only when legitimate.
  **Done:** artwork appears only when the preview resolver returns Apple's official artwork for a
  verified match; otherwise a neutral placeholder. No generated/fake covers remain.
- [x] **9.4 — Optional preferences with clear wording** (`R`).
  **Done:** "Preferences are optional — you can press “Find Similar Songs” without choosing
  anything", plus a note under the controls. Empty preferences reproduce pure similarity.
- [x] **9.5 — Audio previews (legitimate source)** (`R`) — investigate providers, no invented URLs.
  **Done:** `previews.py` uses the public **iTunes Search API** (no key/OAuth). Verified a real
  `m4a` (`audio/x-m4p`, 1,016,056 bytes) from `audio-ssl.itunes.apple.com`. Deezer timed out on this
  network; Spotify requires OAuth and no longer exposes 30 s preview URLs. Strict artist+title
  verification; cover/tribute acts rejected; no audio stored in the repo.
- [x] **9.6 — Player controls** (`R`) — play, pause, resume, replay from start, next, 30 s auto-stop,
  single playback, now-playing indicator.
  **Done:** one `Audio` object inside a single component; verified in a real headless Chrome session:
  play (`Pause`, progress 3.7%) → pause → resume (7.8%) → replay (back to 2.4%) → next (title
  changed) → automatic stop at the end of the 30 s window (status "Preview finished. Press Replay or
  Next.", progress reset). Exactly one card highlighted at all times.
- [x] **9.7 — Unavailable previews handled honestly** (`R`).
  **Done:** cards without a verified preview show "Preview unavailable" + "Listen on Apple Music"
  (a search link, not audio). Displayed in the captured results screenshot.
- [x] **9.8 — Light, simple design** (`R`) — clean light background, green accents, readable type,
  no sidebar/dark dashboard/statistics.
  **Done:** `.streamlit/config.toml` light theme + one CSS block; Streamlit's developer toolbar is
  hidden. Screenshots in `assets/` (`ui-landing.png`, `ui-results.png`, `ui-playing.png`).
- [x] **9.9 — Regression safety** (`R`) — engine untouched, tests preserved.
  **Done:** `recommender.py`, `preprocessing.py`, `evaluation.py` byte-identical (SHA256 recorded in
  the Phase 9 report); full suite green; UI tests rewritten only where they asserted the removed
  page navigation.

**Phase 9 exit criteria:** one page, real previews with real controls, protected engine unchanged,
screenshots captured, no commits or pushes.

---

## Phase 10 — Premium Now Playing experience (added 2026-11-18)

**Why:** the student asked for a dark violet Now Playing experience with verified
circular artwork, real preview progress and working playback controls.

- [x] **10.1 — Dark violet theme** (`R`) — `#090611` background, `#170D2B` panels, `#963BFF`
  accent, `#B66BFF` glow, white/lavender text; gradients, rounded controls, soft shadows.
  **Done:** page CSS in `app.py` + `.streamlit/config.toml` (`base = "dark"`).
- [x] **10.2 — Expanded Now Playing view** (`R`) — opens on "Play preview"; back control,
  "NOW PLAYING", large circular artwork with a glowing ring, title, artist, genres, progress with
  elapsed/remaining, controls and status.
  **Done:** `#np` overlay in `render_results_html()`. Verified in a real browser: overlay hidden at
  first, opens on play, artwork `<img>` loads from `is1-ssl.mzstatic.com` (Apple CDN).
- [x] **10.3 — Verified artwork only** (`R`) — no generated covers.
  **Done:** the overlay shows `preview.artwork_url`; when absent it shows a neutral note, and the
  test asserts no `<img class="art">` is emitted without a verified preview.
- [x] **10.4 — Real 30 s preview progress** (`R`) — elapsed/remaining from `audio.duration`,
  capped at the 30 s preview; replay restarts at 0; next advances; auto-stop at the end.
  **Done:** verified live — play at 0:02/-0:28, pause froze the bar, resume advanced it, replay
  reset the bar to 2.4 %, next switched to *Shower*, and the preview stopped on its own with
  "Preview finished…".
- [x] **10.5 — Replay icon, not text** (`R`).
  **Done:** SVG circular arrow in both the overlay and the compact bar; a test asserts no visible
  "Replay" label exists.
- [x] **10.6 — One stream at a time; unavailable handled** (`R`).
  **Done:** a single `Audio` object; at most one `.card.playing` at any time. Cards without a
  verified preview have no play button, show "Preview unavailable" and a legitimate external link;
  the player never claims to be playing.
- [x] **10.7 — Return without losing state** (`R`).
  **Done:** closing the overlay keeps playback running (verified: progress advanced 5.9 % → 12.1 %
  after closing) and the compact bar reopens it.
- [x] **10.8 — Responsive + light animations** (`R`).
  **Done:** mobile layout captured at 390×844 (`assets/p10-mobile.png`); 200 ms fades, a slow ring
  rotation, and `prefers-reduced-motion` disables them.
- [x] **10.9 — Regression safety** (`R`).
  **Done:** `recommender.py`, `preprocessing.py`, `evaluation.py` byte-identical (SHA256 in the
  Phase 10 report); full suite green.

**Phase 10 exit criteria:** premium player with real controls, verified artwork, dark violet theme,
screenshots captured, protected files unchanged, nothing committed.

**Phase 8 exit criteria:** the app satisfies every §7 feature and §8 design point, and survives the
six error cases.

---

## Phase 9 — Documentation
**Goal:** An instructor can clone, install, and run this without asking questions.
**Dependencies:** Phases 2–8 (final behaviour frozen). **Effort:** M.

- [ ] **9.1 — `README.md` structure** (`R`) — Title, one-line description, screenshot/GIF placeholder
  (added only when a real one exists), features, dataset source + license + how to obtain,
  installation, usage, project structure, methodology summary, evaluation summary, limitations,
  future work, acknowledgements, license.
  *Done when:* every section is filled with real content.
- [ ] **9.2 — Reproduce-from-scratch test** (`R`) — In a clean directory/venv: follow only the README
  and get to a working app. Fix everything that breaks.
  *Done when:* the walkthrough succeeds and the steps are corrected to match reality.
- [ ] **9.3 — `docs/dataset.md`** (`R`) — Source, license/attribution, access date, schema table,
  quality findings, cleaning rules, and the used/excluded feature table from 2.6.
  *Done when:* it is self-contained and consistent with the code.
- [ ] **9.4 — Algorithm explanation** (`R`) — A short "how it works" write-up with the math in plain
  form (vector representation, cosine formula, preference weighting, complexity and runtime notes).
  *Done when:* a classmate could reimplement it from this text.
- [ ] **9.5 — Final structure ↔ README consistency** (`R`) — The README tree matches the actual tree.
  *Done when:* diffing the listed files against the repo yields no surprises.
- [ ] **9.6 — Code docstrings/comments pass** (`R`) — Public functions documented, non-obvious
  decisions explained, no dead code, no commented-out blocks left behind.
  *Done when:* a read-through of every module finds no confusing section.
- [ ] **9.7 — `LICENSE` / acknowledgements** (`O`) — Acknowledge dataset, libraries, and any
  external code or references (also required for original-work rules).
  *Done when:* external sources are credited.

**Phase 9 exit criteria:** README + docs fully describe reality, verified by a clean-slate run.

---

## Phase 10 — GitHub workflow and submission
**Goal:** Correct folder, meaningful history, a real Pull Request, no contamination of other
students' work.
**Dependencies:** Q1/Q2 answered; Phases 0–9. **Effort:** M.

- [ ] **10.1 — Confirm folder ownership** (`R`) — Verify `gaa075` (Q1) against the shared repo and TA
  instructions before touching anything remote.
  *Done when:* Q1 is answered in writing and matches the local directory name.
- [ ] **10.2 — Clone/sync the shared repo safely** (`R`) — Fork or branch per TA instructions; add
  your folder only; never modify or delete other students' directories.
  *Done when:* `git status` shows changes only inside the assigned folder.
- [ ] **10.3 — Commit hygiene pass** (`R`) — Review history for: no dataset/large files, no venv,
  no secrets, coherent messages, commits grouped by topic. Use `git log --stat` and a size check.
  *Done when:* the largest blob in history is small and the log reads like a narrative.
- [ ] **10.4 — Push and open the Pull Request** (`R`) — PR title and description summarizing scope,
  approach, evidence (tests/eval), and known limitations; link the report.
  *Done when:* the PR exists and shows only intended file changes.
- [ ] **10.5 — PR self-review** (`R`) — Review your own diff as if by a grader: leftover TODOs,
  debug prints, unused imports, inconsistent naming.
  *Done when:* the diff is clean and any reviewer comment you would raise is fixed.
- [ ] **10.6 — Repository presentation** (`O`) — Repo description/topics, pinned screenshot, clean
  commit titles — cheap marks on the 20-point code+repo criterion.
  *Done when:* a grader landing on the repo understands it in 10 seconds.

**Phase 10 exit criteria:** PR open against the correct folder, nothing outside it touched, history
clean.

---

## Phase 11 — Technical report (PDF, ≥10 pages)
**Goal:** A report describing the *actual* implementation and *actual* results.
**Dependencies:** Phases 5–8 (results must exist), C4. **Effort:** L–XL.

- [ ] **11.1 — Reference collection** (`R`) — 8–12 real, verifiable sources (content-based
  recommendation, cosine similarity in IR, music recommender systems and their known limitations,
  the dataset's own citation, sklearn documentation as a tool reference). Every entry needs a real
  URL/DOI and must actually be read — no fabricated citations.
  *Done when:* the bibliography exists with URLs/DOIs and each source is quoted/used somewhere.
- [ ] **11.2 — Outline and page budget** (`R`) — Map the 9 required sections onto ≥10 pages, with
  figures/tables planned per section.
  *Done when:* an outline with page estimates exists.
- [ ] **11.3 — Title page + abstract** (`R`) — From Q3; abstract = problem, method, key real results,
  limitations, in ~200 words.
  *Done when:* no claim in the abstract lacks a matching section.
- [ ] **11.4 — Introduction + related work** (`R`) — Motivation, problem statement, objectives, and
  a literature review positioning content-based filtering against alternatives (citing 11.1).
  *Done when:* every citation is used to make a point, not just listed.
- [ ] **11.5 — Design and methodology** (`R`) — Pipeline diagram, feature decisions from 2.6, scaling
  and encoding, cosine similarity math, preference weighting, and the reasoning for each choice.
  *Done when:* it matches the code exactly.
- [ ] **11.6 — Implementation** (`R`) — Tech stack, module responsibilities, key code excerpts
  (short, real), UI description with real screenshots, error-handling approach.
  *Done when:* screenshots are of the actual running app.
- [ ] **11.7 — Results and discussion** (`R`) — Evaluation design, the real tables from Phase 6,
  qualitative examples, and honest discussion of what failed or is weak.
  *Done when:* every number matches a saved artifact; no metric is reported that was not computed.
- [ ] **11.8 — Conclusion and future work** (`R`) — What was achieved vs. objectives, limitations,
  and concrete next steps (including collaborative-filtering extension as future work only).
  *Done when:* future work is not written as if already built.
- [ ] **11.9 — References** (`R`) — Consistent citation style (e.g. IEEE/APA), complete entries.
  *Done when:* every in-text citation appears in the list and vice versa.
- [ ] **11.10 — Formatting and export** (`R`) — Page numbers, captions, consistent fonts/headings,
  figure numbering, spell-check, export to PDF, verify page count ≥10.
  *Done when:* the PDF opens cleanly, is ≥10 pages, and reads without layout breakage.
- [ ] **11.11 — Proofread pass** (`R`) — Read once for correctness (claims vs. reality) and once for
  language/clarity.
  *Done when:* both passes are done and any overstatement removed.
- [ ] **11.12 — Commit the report** (`R`) — Place under `report/` per Q7 with a sensible name.
  *Done when:* the file is in the PR and opens from a fresh clone.

**Phase 11 exit criteria:** a ≥10-page PDF with all nine required sections, real results, valid
references, and no fabricated content.

---

## Phase 12 — Demonstration preparation (3–5 minutes)
**Goal:** A rehearsed demo that also prepares you for technical questions.
**Dependencies:** Phase 8 (working app), Phase 11 (numbers to cite). **Effort:** M.

- [ ] **12.1 — Demo script with timings** (`R`) — Problem (30 s) → algorithm (60 s) → live app:
  select song, show results (90 s) → preference toggle showing the change (45 s) → evaluation
  results (30 s) → limitations + wrap-up (30 s).
  *Done when:* a timed script exists within 3–5 minutes.
- [ ] **12.2 — Prepare the demo data state** (`R`) — Warm caches, chosen example songs that
  demonstrably work well, and a fallback screenshot set in case of environment failure.
  *Done when:* the demo runs offline from a cold start in the real demo environment.
- [ ] **12.3 — Rehearse the live path** (`R`) — Run the exact demo 3× with a timer, including the
  preference-change moment.
  *Done when:* three clean runs inside the time window.
- [ ] **12.4 — Q&A preparation** (`R`) — Write answers to likely questions: why cosine and not
  Euclidean; why scaling matters; what the score means; why not collaborative filtering; what the
  limitations are; how you would evaluate with real user data; how the preference weight works.
  *Done when:* each answer is 3–5 accurate sentences matching the code.
- [ ] **12.5 — Contingency plan** (`R`) — If the app dies mid-demo: screenshots + CLI output ready;
  know the one command to restart.
  *Done when:* the fallback path has been tested once.
- [ ] **12.6 — Slide deck** (`O`) — 5–6 minimal slides: title, problem, approach diagram, demo,
  results, limitations.
  *Done when:* it renders without text overflow.

**Phase 12 exit criteria:** a rehearsed, timed, failure-tolerant demonstration and prepared answers.

---

## Phase 13 — Final audit and hand-over
**Goal:** Nothing mandatory is missing at submission time.
**Dependencies:** all required phases. **Effort:** S–M.

- [ ] **13.1 — Run the definition-of-success audit** (`R`) — Walk brief §14 item by item and mark
  each as met/unmet with evidence (file, command output, PR link). Fix every "unmet".
  *Done when:* all §14 items are met or explicitly explained.
- [ ] **13.2 — Verify the grading matrix** (`R`) — Re-check the Phase 1 matrix: no empty rows.
  *Done when:* every requirement points to existing, working evidence.
- [ ] **13.3 — Final green run** (`R`) — Fresh clone + README-only install + `pytest -q` +
  `streamlit run app.py` all succeed.
  *Done when:* all three pass in one session from scratch.
- [ ] **13.4 — Hygiene sweep** (`R`) — No TODO/FIXME left, no debug prints, no secrets, no dataset
  in the diff, `.gitignore` effective, dependency list complete and minimal.
  *Done when:* a search for leftover markers returns nothing meaningful.
- [ ] **13.5 — Submission package** (`R`) — PR open and reviewed, PDF report uploaded per LMS rules,
  demo ready, repo clean and public/visible per instructions.
  *Done when:* the submission checklist is fully checked and you have the PR link saved.

**Phase 13 exit criteria:** the project meets brief §14 in full, evidenced.

---

## Optional extensions (only after every `R` task is complete)

Do not start these before submission-critical work is done. Each needs a short written justification
and a scope cap — they can sink an otherwise finished project.

- [ ] **O-1 — Collaborative filtering baseline** (`O`) — Requires user–item interaction data, which
  this dataset likely lacks. If absent, treat as future work only; do not fabricate interactions.
- [ ] **O-2 — Hybrid re-ranker** (`O`) — Combine content similarity with a popularity prior or a
  graph-based neighbourhood signal; measure whether it actually improves the qualitative results.
- [ ] **O-3 — Diversity-aware re-ranking** (`O`) — MMR-style penalty so a playlist is not ten near-
  identical tracks; report intra-list diversity before/after.
- [ ] **O-4 — Audio preview / artwork** (`O`) — Only with a legitimate API and clear attribution;
  adds demo polish, adds failure modes.
- [ ] **O-5 — User feedback capture** (`O`) — Thumbs up/down stored locally to adjust weights. Only
  if it is genuinely implemented; never describe it as "learning user taste" unless it does.
- [ ] **O-6 — Deployment** (`O`) — Streamlit Community Cloud with a data-download step; optional,
  and must not risk the graded local run.
- [ ] **O-7 — CI workflow** (`O`) — GitHub Actions running `pytest` on push; cheap credibility for
  the code-quality criterion.

---

## Risks and unknowns

| # | Risk | Impact | Mitigation (task) |
|---|---|---|---|
| R1 | Dataset columns differ from expectations (missing audio features, few genres) | Model design changes | Verify before coding: 2.4–2.6, gate C1 |
| R2 | Dataset license restricts redistribution or use | May need a different dataset | License check first: 2.1, fallback decision in 2.2 |
| R3 | Dataset download blocked on the university network | Blocks Phases 3–8 | Q4 answered; offline fallback plan in 2.2 |
| R4 | Similarity matrix too large for memory/time | Slow app, crash risk | Row-wise dot products (5.1), scope rule (3.4), caching (4.5/4.8) |
| R5 | Recommendations look "wrong" to a listener | Weak grade on functionality/demo | Early qualitative gate C3; tune features via 4.3/4.4/4.7 |
| R6 | Categorical one-hot swamps numeric features | Genre-only neighbours | Explicit block weighting justified in 4.4, checked at C3 |
| R7 | Duplicate tracks pollute results | Visible quality bug | Identity key 4.2 + dedup 5.3 + test 7.3 |
| R8 | No ground-truth labels → cannot compute precision/recall | Report credibility | Evaluation design 6.1 states the limitation explicitly |
| R9 | Wrong GitHub folder / touching others' files | Serious submission problem | Q1 confirmed, 10.1–10.2 guards |
| R10 | Report written last and rushed, fabricated numbers | 30-mark loss | Draft sections from real artifacts (11.5–11.7), never earlier |
| R11 | Scope creep into hybrid/CF | Core unfinished | Optional list is explicitly gated |
| R12 | Large dataset accidentally committed | Repo bloat, rubric penalty | `.gitignore` 0.4 + history audit 10.3 |

---

## Checkpoints at a glance

| Gate | Where | What you run / inspect | Passes when |
|---|---|---|---|
| **C1** | End of Phase 2 | Schema + license + feature table | You confirm the used fields |
| **C2** | End of Phase 4 | Full pipeline, matrix shape, sample neighbours | Matrix is sane and reproducible |
| **C3** | End of Phase 5 | CLI recommendations for songs you know | Neighbours are plausibly similar |
| **C4** | End of Phase 6 | Evaluation command + saved results | Numbers are real and trustworthy |
| **C5** | End of Phase 8 | Full app walkthrough incl. error cases | Clean, honest, demo-ready |
| **C6** | End of Phase 13 | Definition-of-success audit | All brief §14 items evidenced |

---

## Definition of done (mirrors brief §14)

- [ ] Dataset source, license, and used fields verified and documented (Phases 2–3).
- [ ] Data preparation reproducible from one documented pipeline (Phase 4).
- [ ] Recommendations computed from real features, ranked, de-duplicated, self-excluded (Phase 5).
- [ ] Preferences influence ranking exactly as documented (5.4, 6.4).
- [ ] Streamlit app runs and handles the six common error cases (Phase 8).
- [ ] Tests pass and evaluation results are documented honestly (Phases 6–7).
- [ ] Code modular, commented, dependencies listed, `.gitignore` correct (Phases 0, 9).
- [ ] README enables install-and-run from scratch (9.2).
- [ ] Git history shows meaningful progress; PR opened against the correct folder (Phase 10).
- [ ] PDF report ≥10 pages with all nine required sections and real results (Phase 11).
- [ ] 3–5 minute demonstration prepared and rehearsed (Phase 12).

---

## Suggested execution order (short version)

`0 → 1 → 2 (C1) → 3 → 4 (C2) → 5 (C3) → 6 (C4) → 7 → 8 (C5) → 9 → 10 → 11 → 12 → 13 (C6)`

Phases 6 and 7 can overlap with Phase 8 once C3 passes. Phase 9 (docs) should be written
incrementally alongside 4–8 and finalized after them. Phase 11 (report) starts only when Phase 6's
numbers exist. Phase 12 starts once the app is stable.

---

*End of plan. Nothing here has been started, no dataset facts have been asserted, and no results
have been invented. Tell me when to begin, and with which phase.*
