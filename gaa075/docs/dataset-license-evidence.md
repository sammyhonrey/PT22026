# Dataset access and licence evidence

Research notes for Phase 2 (tasks 2.1 and the pre-download part of 2.2).
Collected 2026-11-18 on the project machine. Nothing in this file is inferred:
each claim names the endpoint or file it came from, and the raw responses were
saved alongside this note.

## 1. Candidate sources

| Source | URL | Status on this machine |
|---|---|---|
| Hugging Face mirror (named in the brief) | `https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset` | **Unreachable** |
| Kaggle original (author: MaharshiPandya) | `https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset` | **Reachable (HTTP 200)** |

### Why the Hugging Face page could not be used

```
huggingface.co              -> 202.160.130.117   443 open: False
datasets-server.huggingface.co -> 157.240.3.50   443 open: False
curl https://huggingface.co -> http=000 after 15.0 s (curl: (28) timeout)
curl https://pypi.org       -> http=200 after 0.41 s   (control: network otherwise works)
```

Three separate attempts (page, dataset card, dataset viewer API) each timed out
with zero bytes. The DNS answers are themselves suspicious — `157.240.3.50`
belongs to Facebook's address space, not Hugging Face — which indicates
DNS-level interference on this network rather than an outage. **This is an
environment blocker, not a dataset problem.** The Hugging Face copy could not be
inspected, so no claim is made about its licence page.

## 2. Dataset identity (from Kaggle's public metadata API)

Endpoint: `https://www.kaggle.com/api/v1/datasets/view/maharshipandya/-spotify-tracks-dataset`
(HTTP 200, 14,509 bytes, saved as `kaggle-view.json`)

| Field | Value |
|---|---|
| Creator | `MaharshiPandya` |
| Title | `🎹 Spotify Tracks Dataset` |
| Subtitle | "A dataset of Spotify songs with different genres and their audio features" |
| Reported total size | `20,118,244` bytes (uncompressed total) |
| Usability rating | `1.0` |

## 3. Licence (verified, verbatim)

Kaggle metadata field `licenseNameNullable`:

> **Database: Open Database, Contents: © Original Authors**

The dataset page's embedded structured data (schema.org `license` object) gives
the same name together with the licence URL:

```json
"license":{"@type":"CreativeWork",
           "name":"Database: Open Database, Contents: © Original Authors",
           "url":"http://opendatacommons.org/licenses/odbl/1.0/"}
```

So the **database** is licensed under the **Open Database License (ODbL) v1.0**,
while copyright in the **contents** (the individual data) stays with the original
authors. The licence text was fetched directly from the Open Data Commons
(HTTP 200, 51,176 bytes, `<title>Open Data Commons Open Database License (ODbL) v1.0</title>`).

### Obligations that follow from ODbL 1.0

| Section | Obligation | What it means for this project |
|---|---|---|
| §4.2 Notices | Any public use/conveyance of the database must keep or display the required notices and attribution | The README and report must state the dataset name, the author (MaharshiPandya), the source, and that it is ODbL 1.0 |
| §4.4 Share alike | If a public Derivative Database is conveyed, it must be offered under ODbL (or a compatible licence) | If we ever publish a derived *database*, the same licence must apply to it. Publishing *code* and *analysis* is not the same as publishing a derivative database, but the question should be treated with care |
| §4.6 | Technological measures must not restrict exercise of the licensed rights | Do not add DRM/redistribution limits to the data |

**Usage conclusion (honest, narrow):** the dataset may be used in this academic
project with clear attribution. It is **not** "public domain" and **not**
unrestricted: attribution is mandatory, and share-alike applies to publicly
conveyed derivative databases. The dataset must therefore **not be committed to
Git** (which the project's `.gitignore` already enforces for `data/*.csv`); only
download instructions and the attribution text should be versioned.

## 4. Retrieval facts (measured, no bulk download yet)

| Fact | Evidence |
|---|---|
| Download endpoint | `https://www.kaggle.com/api/v1/datasets/download/maharshipandya/-spotify-tracks-dataset` |
| Authentication needed | **No** — a 1-byte range request returned `HTTP 206` with `Content-Type: application/zip` |
| Compressed size on the wire | `8,571,542` bytes (≈ **8.2 MB**) — from `Content-Range: bytes 0-0/8571542` |
| Archive content | contains `dataset.csv` — filename read from the ZIP central directory tail |
| Uncompressed total (owner-reported) | `20,118,244` bytes (≈ 19.2 MB) |

## 5. Still unverified (requires the file)

These items are **unknown** and must not be stated anywhere until the CSV is
downloaded and inspected:

- actual row count and column count;
- exact column names as they appear in the file (the owner *documents* 18 columns:
  `track_id, artists, album_name, track_name, popularity, duration_ms, explicit,
  danceability, energy, key, loudness, mode, speechiness, acousticness,
  instrumentalness, liveness, valence, tempo, time_signature, track_genre` —
  documentation is not verification);
- data types;
- missing-value counts and percentages;
- duplicate rows and duplicate track identities;
- invalid or suspicious values (e.g. tempo = 0, out-of-range audio features);
- genre distribution and class balance.

## 6. Raw evidence files

Saved during collection (outside the repository, in `%TEMP%\gaa075-phase2`):
`kaggle-view.json`, `kaggle-page.html`, `dl-headers.txt`, `zip-tail.bin`,
`odbl.html`, `probe.bin`, `data-view.json`.
