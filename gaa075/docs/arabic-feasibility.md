# Arabic music expansion — Stage A feasibility report

Date: 2026-11-18. **No production code was changed for this report.** It records
the audit, the dataset research, and why implementation cannot proceed safely
without a decision from the student.

## 1. Current architecture (audited)

| Module | Role | Relevant facts |
|---|---|---|
| `preprocessing.py` | Builds the catalogue and feature matrix | 114,000 rows → **89,583 unique tracks**; 9 numeric audio features + **114** one-hot genre columns; requires exact columns `danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence, loudness, tempo` |
| `recommender.py` | Content-based ranking | `base = 0.90·S_audio + 0.10·S_genre`; optional `final = 0.80·base + 0.20·preference`; duplicate-recording suppression; explanations from real feature contributions |
| `evaluation.py` | Reproducible evaluation | 17 references, ILS/diversity/coverage; no relevance metrics (no labels) |
| `previews.py` | 30 s Apple previews | iTunes Search API; strict artist + title verification; rejects cover/tribute acts; caches lookups |
| `app.py` | Single-page UI | search → selected song → optional preferences → 5/10 → results; Now Playing overlay (Phase 10) |

**What the engine actually requires:** nine genuine Spotify-defined audio features
per track. A track without them cannot enter the matrix — there is no code path for
metadata-only rows, and adding one would change the similarity space.

**Arabic material in the current catalogue:** *none.* A scan for Arabic-script
characters (`U+0600–U+06FF`, `U+0750–U+077F`, `U+08A0–U+08FF`) over title, artist
and album found **6 rows, all in the `iranian` genre (Persian, not Arabic)**. A
check of 18 well-known Arabic artists under Latin spelling (Amr Diab, Fairuz,
Nancy Ajram, Kadim Al Sahir, Mohamed Hamaki, Elissa, Ragheb Alama, Saad Lamjarred,
Majid Al Muhandis, Hussain Al Jassmi, Abdul Majeed Abdullah, …) returned **0 real
matches** — the two hits were false positives (a Swedish act containing "elissa",
and the Brazilian singer Anitta).

So the system today recommends **zero Arabic songs**. The expansion is not a matter
of adding a label; a new catalogue with genuine features is required.

## 2. Dataset research

| # | Source | Access | Size | Arabic tracks | Columns | Spotify-style features? | Licence | Usable? |
|---|---|---|---|---|---|---|---|---|
| 1 | Hugging Face datasets (`huggingface.co`, `datasets-server.…`) | — | — | unknown | unknown | unknown | unknown | **No — host unreachable.** DNS-level interference on this network: TCP 443 closed, `datasets-server` resolves to a Facebook IP, every request times out. Re-tested in this phase. |
| 2 | Kaggle `araraltawil/audio-classifier` (**Ar-MGC**: Arabic Music Genre Classification) | public API, no auth for metadata; download needs Kaggle credentials | **7,985 MB** | 5 Arabic genres (mawwal, Andalusian muwashahat, rai, poetic qasida, oriental takht) | raw audio encoded in JSON | **No** — audio only, no tabular audio features | **CC0 1.0 (Public Domain)** | Metadata-verified; **not feature-compatible** |
| 3 | Kaggle `murtadhanajim/arabic-maqam-music-features` | same | **6,862 MB** | not stated | unknown (no unauthenticated file preview) | Likely maqam/MFCC-style features, **not** the nine Spotify fields | **Apache 2.0** | Cannot verify schema without a multi-GB download |
| 4 | Kaggle `dadalyndell/arabic-hip-hop` | same | 1.9 MB | 864 songs | lyrics, song name, author origin | **No** | **Unknown** | **No** — lyrics only + unclear licence |
| 5 | Kaggle `bassantsherif123/songs-clips` | same | 549 MB | 60 songs (Arabic + English), 600 clips | 5-second audio clips + male/female label | **No** | **Unknown** | **No** — copyrighted clips, unclear licence |
| 6 | MusicBrainz + AcousticBrainz (open computed features) | public APIs | API | probe below | Essentia high-level: mood, danceability, voice/instrumental, tonal | **Different definition** (not Spotify's nine) | CC0 data, API terms | **Not compatible** (see §3) |
| 7 | Spotify Web API | requires OAuth client credentials | API | — | exact nine features | **Yes, by definition** | Spotify Developer Terms | Blocked: no credentials here; audio-features access for new apps is restricted and cannot be verified |

### AcousticBrainz coverage probe (evidence, not assumption)

MusicBrainz lookups were partly throttled (`HTTP 503`), and for the artists that
resolved, AcousticBrainz had **no features**:

```
artist          MB recordings sampled   with AcousticBrainz features
Fairuz                       0                      0
Cheb Khaled                  5                      0   (404 per recording)
others                    not resolved (503)
```

Even where it exists, AcousticBrainz provides *Essentia* features (mood, voice,
tonal, its own "danceability" model), **not** Spotify's `danceability`, `energy`,
`valence`, `acousticness`, `instrumentalness`, `liveness`, `speechiness`,
`loudness` and `tempo`. Substituting them would silently change what the cosine
similarity means.

## 3. Why the candidates cannot be integrated as-is

1. **Feature semantics differ.** Spotify's nine features are proprietary
   perceptual estimates with their own scales and distributions. Essentia models or
   raw DSP statistics (RMS energy, spectral centroid, librosa tempo) are *different
   measurements* that happen to share names.
2. **The engine compares one shared vector space.** Mixing two feature definitions
   in one matrix means a track's "energy" is sometimes Spotify's estimate and
   sometimes an RMS value. Scores would be meaningless, and the user's own rules
   forbid filling features with arbitrary or proxy values.
3. **The Arabic datasets that are legally clean contain audio, not features.**
   Ar-MGC is CC0 and could legally be used — but only to *derive* features, and the
   derivable ones are not the required nine.
4. **Licensing.** Ar-MGC (CC0) and the maqam dataset (Apache 2.0) are usable;
   the lyrics and clip datasets have **Unknown** licences and should not be used.
5. **Access.** Hugging Face — the natural home for feature-rich datasets — is
   unreachable from this machine, so we cannot even verify the obvious candidates.

**Conclusion: no dataset available in this environment provides the nine required
audio features for Arabic tracks. Per the project rules, Stage B must not start by
inventing or approximating features.**

## 4. Recommended options (need a decision)

| Option | What it would take | Honesty | Effort |
|---|---|---|---|
| **A. Metadata-only Arabic layer** *(recommended if Arabic coverage matters more than audio similarity)* | A licensed metadata source (MusicBrainz dump filtered to Arabic artists, or a CC0 metadata set). New module `arabic_catalogue.py`; Arabic tracks ranked by genre/artist metadata only, in a **separate, clearly labelled** "Arabic (metadata)" mode. Never mixed into the cosine scores. | Fully honest: no audio claims | M |
| **B. Obtain Spotify credentials** | A Spotify developer app (client ID/secret) owned by the student. A one-off script fetches audio features for a curated Arabic track list. Requires confirming that audio-features access is still granted to new apps and that storing the features is permitted. | Honest, exact feature compatibility | S–M + credentials |
| **C. Second Arabic engine on open features** | Use Ar-MGC (CC0) audio → compute an open feature set (tempo, RMS, spectral, MFCC) → a **separate** cosine engine for Arabic only, with its own documented feature definition. Arabic and international catalogues are separate search modes; scores are never compared across them. | Honest but two engines | L (8 GB download + DSP) |
| **D. Stop the expansion** | Keep the current international-only system. | Honest | 0 |

Option A is the only route that adds Arabic music without touching the protected
engine or fabricating features. Option B is the only route that makes Arabic tracks
*first-class* in the same audio space, and it needs credentials from the student.

## 5. Estimated impact if Option B were approved

Based on the existing pipeline (89,583 tracks, 2.0 s build, ~7 MB matrix data):

* Adding ~2,000 Arabic tracks would grow the matrix by ~2.2 % (≈1,800 extra
  rows), add ~0.2 s to the build and < 1 MB to memory;
* the dedup/duplicate rules would apply unchanged;
* Apple preview matching works the same way, but **Arabic titles are harder to
  match** (transliteration, `feat.` variants, dialect spelling) — expect a higher
  "Preview unavailable" rate than the ~80 % seen for international tracks.

## 6. What was *not* done

No files were modified for this stage. `recommender.py`, `preprocessing.py`,
`evaluation.py` and `previews.py` are byte-identical (SHA256 recorded in the
Phase 11 Stage A report). The dataset is untouched, nothing was downloaded into the
repository, and no Git operation was performed.
