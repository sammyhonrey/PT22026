# VoiceForge — Academic Report Outline and Fact Sheet

This file supports the written university report. It contains a structure, a
verified fact sheet and diagram sketches. It deliberately contains **no
invented measurements**: every number here was produced by commands in this
repository.

## 1. Report structure (target: 10+ pages)

### 1. Title page

- Title: *VoiceForge: A Local Text-to-Speech Web Application with a
  Pretrained Kokoro-82M Engine and an Experimental Tacotron-lite Model*
- Course: YN3012170034
- Student: Almaqtari Abdulrahman Mohammed Najib — 20243120042
- Institution: Yunnan University
- Date: November 2026

### 2. Abstract (~200 words)

Cover: problem (accessible local TTS), two engines (pretrained vs
student-trained), the web application (six pages), persistence (SQLite),
and the measured test results (200 Python tests, 100 JavaScript tests,
16 real-model tests).

### 3. Introduction

- Motivation: offline, private, local speech synthesis.
- Scope: English single-speaker TTS, document input, reusable library.
- Contributions: integration and evaluation of Kokoro; Tacotron-lite
  implementation; a complete local application with long-text jobs,
  batch processing, pronunciation rules, MP3 export and presets.

### 4. Project objectives and requirements

- Functional: generate speech from text, documents; long text; batch;
  library with search/rename/delete; MP3; pronunciation; presets.
- Non-functional: local-only, no external services at runtime, safe
  uploads, reproducible tests.

### 5. Related work

- Tacotron / Tacotron 2 (sequence-to-sequence TTS).
- Griffin–Lim vocoder (phase reconstruction).
- Kokoro-82M (pretrained open-weight TTS).
- Rule-based pronunciation correction (lexicon substitution).
- Modern local web applications using the standard library.

### 6. System architecture

```
Browser (HTML + CSS + ES modules)
   │  fetch JSON / audio
   ▼
Python standard-library HTTP server (voiceforge/web_app.py)
   ├── /api/engines, /api/generate, /api/jobs, /api/library,
   │   /api/documents, /api/pronunciation, /api/presets
   ├── JobManager (one worker, bounded queue)
   ├── Audio Library (SQLite + WAV files)
   └── VoiceForgeApp (voiceforge/app_core.py)
         ├── KokoroBackend  → KPipeline (Kokoro-82M, 24 kHz)
         └── VoiceForgeTTS  → Tacotron-lite + Griffin–Lim (22.05 kHz)
```

- Synthesis is centralised in `VoiceForgeApp.synthesize`.
- The server holds one CPU/GPU model instance; a lock serialises access.
- Jobs reuse the same synthesis path; the library records every result.

### 7. Technologies and tools

Python 3.11.9, PyTorch 2.14.1, torchaudio, NumPy, soundfile, kokoro 0.9.4,
pypdf, python-docx, lameenc, SQLite (stdlib), `http.server`, HTML/CSS/ES
modules, Node.js built-in test runner, pytest.

### 8. TTS engine and model selection

- Kokoro-82M: pretrained, 24 kHz, seven US-English voices, speed 0.5–2.0,
  licence Apache-2.0. Used as the main engine.
- Custom model: Tacotron-lite (attention-based encoder/decoder), 22.05 kHz,
  Griffin–Lim, 38-character vocabulary, trained on LJSpeech; **experimental**
  and audibly less natural. It is presented as a learning artefact, not as a
  production engine.

### 9. System design and implementation

- Text pipeline: normalisation → vocabulary (custom) or G2P (Kokoro).
- Long text: sentence-aware splitter, engine-sized chunks, WAV join with
  150 ms gaps.
- Jobs: `queued → running → completed/failed`, real item counts.
- Audio: WAV (`PCM_16`) always; MP3 on demand through `lameenc`.
- Persistence: SQLite schema v2 with `audio_records`,
  `pronunciation_entries`, `voice_presets`.

### 10. Main features (one subsection each)

Speech Studio · Long Text · Batch Studio · Audio Library · Document to
Speech · Pronunciation Dictionary · MP3 Export · Voice Presets.

### 11. Database design

ER sketch:

```
audio_records(id PK, name, original_text, filename, source_filename,
              engine, voice, speed, sample_rate, duration_seconds,
              file_size, source_type, job_id, job_item_id,
              created_at, updated_at)
pronunciation_entries(id PK, word UNIQUE NOCASE, pronunciation,
                      description, created_at, updated_at)
voice_presets(id PK, name UNIQUE NOCASE, engine, voice, speed,
              description, created_at, updated_at)
```

Indexes: `created_at`, `name COLLATE NOCASE`, `engine`, `source_type`,
unique `source_filename`, unique `(job_id, job_item_id)`.
Migrations use `PRAGMA user_version` (1 → 2, additive only).

### 12. API design

REST-style JSON endpoints, one error shape:

```json
{"ok": false, "error": "human message", "code": "machine_code"}
```

List the endpoints from the README table (engines, generate, jobs,
library, documents, pronunciation, presets, MP3).

### 13. Testing methodology

- Python: pytest, 200 passing tests, 16 environment-gated real-Kokoro
  tests (skipped without `VOICEFORGE_TEST_KOKORO=1`).
- JavaScript: Node built-in test runner, 100 passing tests.
- Real-engine tests: cached Kokoro weights, no re-download.
- Browser: headless Edge via CDP; 6 pages × 4 viewport sizes (1440×900,
  1366×768, 768×1024, 390×844) = 24 checks, zero horizontal overflow,
  zero uncaught JS exceptions.

### 14. Results and discussion

Report only measured facts:

- Test counts above; all suites green.
- Real Kokoro jobs verified end-to-end (studio, long text, batch,
  documents, MP3 export, pronunciation, presets).
- Pronunciation rule changes the phoneme string (e.g. `Abdulrahman` vs
  `Ab-dool-rah-man`) and produces different audio; this demonstrates the
  rule reaches the model, **not** that pronunciation is objectively more
  accurate.
- Library records survive a server restart.
- Custom engine: digits are rejected, output is buzzy; discuss honestly.

### 15. Limitations

Scanned PDFs (no OCR), respelling-based pronunciation, in-memory job
history, single-worker throughput, custom model quality, English only,
no authentication (local single user).

### 16. Conclusion and future work

Persistent job history, OCR, phoneme-level lexicon, multi-speaker custom
model, fine-tuning Kokoro, streaming synthesis.

### 17. References (real sources only)

1. Wang, Y. et al. (2017). *Tacotron: Towards End-to-End Speech Synthesis.*
   arXiv:1703.10135.
2. Griffin, D. and Lim, J. (1984). *Signal Estimation from Modified
   Short-Time Fourier Transform.* IEEE Transactions on ASSP.
3. hexgrad. *Kokoro-82M* model card, Hugging Face:
   https://huggingface.co/hexgrad/Kokoro-82M (Apache-2.0).
4. hexgrad. *kokoro* Python package / GitHub repository.
5. *misaki* G2P library (hexgrad), used by Kokoro for English.
6. SQLite documentation: https://www.sqlite.org/docs.html
7. pypdf documentation: https://pypdf.readthedocs.io/
8. python-docx documentation: https://python-docx.readthedocs.io/
9. LAME MP3 encoder project: https://lame.sourceforge.io/
10. Fielding, R. et al. (2022). *HTTP Semantics*, RFC 9110.

### 18. Appendices

- A: API examples (from README).
- B: Test command output.
- C: Screenshots (generated locally under `outputs/`, not committed).
- D: Database schema SQL.

## 2. Verified fact sheet

| Fact | Value | How verified |
|---|---|---|
| Python version | 3.11.9 | `python --version` |
| Kokoro package | 0.9.4 | `pip list` |
| Kokoro sample rate | 24 000 Hz | `/api/engines`, WAV headers |
| Custom engine sample rate | 22 050 Hz | `/api/engines` |
| Kokoro voices | 7 (US English) | `/api/engines` |
| Speed range | 0.5–2.0 (Kokoro only) | `/api/engines`, validation tests |
| Short-text limit | 400 characters | `/api/engines` limits |
| Long-text limit | 20 000 characters per item | text splitter + job validation |
| Batch limits | 20 items; 4 active jobs | job manager |
| Upload limits | 1 MB/file, 20 files, 5 MB total (TXT); 10 MB (PDF/DOCX) | uploads/document modules |
| MP3 bitrates | 128 / 192 / 320 kbps | `mp3.py` |
| DB schema version | 2 | `PRAGMA user_version` |
| Python tests | 200 passed, 16 skipped (gated real-Kokoro) | pytest |
| JS tests | 100 passed | `node --test` |
| Real Kokoro tests | 16 passed | gated pytest |
| Browser checks | 24 (6 pages × 4 sizes), 0 overflow, 0 exceptions | CDP run |

## 3. Things not to claim

- Do not claim the custom model is natural or production quality.
- Do not claim OCR support; it does not exist.
- Do not claim pronunciation accuracy beyond "the rule reaches the model".
- Do not claim measured latency/RTF numbers that were not recorded in a
  controlled benchmark; the repository has none.
