# `data/` — local dataset staging (never committed)

This directory is a **workspace for downloaded data**, not a data store. Everything in
it except this README is git-ignored.

```
data/
├── README.md                    this file
├── raw/                         downloaded archives (Audio_Speech_Actors_01-24.zip)
└── ravdess_manifest.csv         path,emotion rows used by the evaluation scripts
```

## Why nothing is committed

The assignment forbids bundling the dataset, and RAVDESS is ~200 MB of CC BY-NC-SA audio.
Committing it would bloat the repository, breach the licence's share-alike terms in a
context the authors did not intend, and make the clone unusable for the marker. The
repository therefore contains **zero dataset audio**; it contains links, a manifest
schema, and scripts that fetch on demand.

## Getting the data

```bash
python scripts/fetch_ravdess.py --print-links
python scripts/fetch_ravdess.py --download
python scripts/fetch_ravdess.py --from-zip data/raw/Audio_Speech_Actors_01-24.zip
```

`--from-zip` extracts the first file for each of six emotion codes into
`demo_samples/` and writes `PROVENANCE_RAVDESS.json` alongside them recording the DOI,
licence, citation and original member names.

## RAVDESS reference

| | |
|---|---|
| Record | <https://zenodo.org/records/1188976> |
| DOI | `10.5281/zenodo.1188976` |
| Licence | CC BY-NC-SA 4.0 |
| Speech archive | <https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1> |

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of
Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391.

## Filename grammar

```
03-01-05-01-02-01-12.wav
│  │  │  │  │  │  └── actor           01–24 (odd = male, even = female)
│  │  │  │  │  └───── repetition      01, 02
│  │  │  │  └──────── statement       01, 02
│  │  │  └─────────── intensity       01 normal, 02 strong
│  │  └────────────── emotion         01 neutral, 02 calm, 03 happy, 04 sad,
│  │                                  05 angry, 06 fearful, 07 disgust, 08 surprised
│  └───────────────── vocal channel   01 speech, 02 song
└──────────────────── modality        03 = audio only
```

## Manifest format

The evaluation scripts read a CSV with a header row. Paths may be absolute, or relative
to the manifest, `data/`, or the repository root.

```csv
path,emotion
raw/Audio_Speech_Actors_01-24/Actor_01/03-01-05-01-02-01-12.wav,angry
raw/Audio_Speech_Actors_01-24/Actor_02/03-01-03-01-01-02-13.wav,happy
raw/Audio_Speech_Actors_01-24/Actor_03/03-01-04-01-01-01-15.wav,sad
```

`emotion` must be one of the six canonical classes:
`happy, sad, angry, fear, disgust, neutral`.

Three mapping decisions the manifest must respect, because RAVDESS does not line up
one-to-one with the target taxonomy:

| RAVDESS | Canonical | Reasoning |
|---|---|---|
| `01 neutral` | `neutral` | direct |
| `02 calm` | `neutral` | calm is low-arousal neutrality, not a separate class |
| `03 happy` | `happy` | direct |
| `04 sad` | `sad` | direct |
| `05 angry` | `angry` | direct |
| `06 fearful` | `fear` | direct |
| `07 disgust` | `disgust` | direct |
| `08 surprised` | *excluded* | surprise has no canonical home; mapping it to `fear` would corrupt the labels |

Excluding `surprised` costs data but keeps the evaluation honest. Document the exclusion
in the report rather than quietly folding it into a neighbouring class.

## Privacy

The demo clips shipped in `demo_samples/` are synthetic. If you record your own audio for
demonstration, note that the file leaves your machine only if you choose to upload it
somewhere; this application analyses in memory and writes nothing to disk.
