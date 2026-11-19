# VoiceForge

**An English text-to-speech application with a pretrained natural-speech engine
and a student-built acoustic model.**

VoiceForge is a university project with **two speech engines**:

1. **Kokoro-82M** — a pretrained, open-weight TTS model by
   [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0). This is
   the default engine and is what makes the application produce natural
   speech. It was **not** trained by this project; the work here is the
   integration, interface, testing and evaluation. See
   [docs/kokoro.md](docs/kokoro.md).
2. **The VoiceForge custom model** — a Tacotron-lite acoustic model written
   and trained from scratch by the student (text processing, attention
   encoder-decoder, duration predictor, Mel spectrograms, Griffin-Lim). It is
   kept as an experimental backend and as evidence of the development work.

> **Current status:** both engines work in the application. Kokoro produces
> natural speech at ~30× faster than real time. The student's own model is
> **not intelligible** — that is measured and documented in
> [docs/evaluation.md](docs/evaluation.md), not hidden.

| | |
|---|---|
| Course | Professional Training (YN3012170034) |
| University | Yunnan University |
| Student | Almaqtari Abdulrahman Mohammed Najib (20243120042) |
| Branch | `alm042` |
| Main engine | Kokoro-82M (pretrained, Apache-2.0) |
| Custom model | Tacotron-lite, 2,997,410 parameters |
| Dataset (custom model) | LJSpeech 1.1 (13,100 clips, single speaker, 22,050 Hz) |
| Tests | 300 automated tests (200 Python + 100 JavaScript), plus 16 optional real-Kokoro tests |

---

## Table of contents

1. [What VoiceForge does](#1-what-voiceforge-does)
2. [Features](#2-features)
3. [How it works](#3-how-it-works)
4. [Repository structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Dataset setup](#6-dataset-setup)
7. [Model checkpoints](#7-model-checkpoints)
8. [Training](#8-training)
9. [Generating speech](#9-generating-speech)
10. [Running the application](#10-running-the-application)
11. [Evaluation](#11-evaluation)
12. [Testing](#12-testing)
13. [Known limitations](#13-known-limitations)
14. [Future improvements](#14-future-improvements)
15. [Documentation](#15-documentation)

---

## 1. What VoiceForge does

A text-to-speech (TTS) system turns written text into a spoken waveform. The
hard part is not producing sound — it is learning *which* part of the audio
belongs to *which* character, and doing so with a single speaker and a small
GPU.

VoiceForge solves this with its own attention-based acoustic model:

```
 English text
      │
      ▼
 normalise text ──► characters ──► character IDs (38-symbol vocabulary)
      │
      ▼
 acoustic model (Tacotron-lite) ──► Mel spectrogram (80 bands × time)
      │
      ▼
 Griffin-Lim phase reconstruction ──► waveform (22,050 Hz)
      │
      ▼
 WAV file / browser playback / Python API
```

## 1.1 Project objectives

1. Provide a **working local text-to-speech application** with no cloud
   service and no API key.
2. Integrate the pretrained **Kokoro-82M** engine as the main, usable voice.
3. Implement and honestly evaluate a small **custom model** (Tacotron-lite
   with a Griffin-Lim vocoder) as a learning artefact.
4. Support realistic workflows: short text, long text, batches, documents,
   reusable recordings, pronunciation rules, MP3 and presets.
5. Keep the system testable and safe: parameterised SQL, validated uploads,
   no tracebacks in responses, reproducible automated tests.

## 2. Features

- **Two speech engines** — pretrained Kokoro-82M for natural speech (default)
  and the student-built model, selectable in the interface.
- **Character-level text normalisation** — smart quotes, accents, brackets and
  whitespace are normalised before encoding (custom model).
- **Attention-based acoustic model** — convolutional encoder, bidirectional
  GRU, location-sensitive attention, GRU decoder, postnet and a duration
  predictor (student-built).
- **Two decoding modes** — length-controlled decoding (stable) and fully
  autoregressive decoding with a stop token.
- **Griffin-Lim vocoder** — reproducible, dependency-free waveform
  reconstruction for the custom model; Kokoro uses an iSTFTNet decoder.
- **Seven Kokoro voices** — American English female and male voices.
- **Speech speed control** — 0.5× to 2.0× for Kokoro, applied per request
  without reloading the model.
- **Persistent Audio Library** — every generated WAV is recorded in SQLite
  with its text, engine, voice, speed and duration, and survives restarts.
- **Document to Speech** — extract text from PDF and DOCX, edit it, then
  synthesise it through the long-text job workflow.
- **Pronunciation dictionary** — teach names and terms how they should sound;
  rules are applied before synthesis in every workflow.
- **MP3 export** — download library recordings and batch results as MP3 at
  128/192/320 kbps, alongside the original WAV.
- **Voice presets** — save engine + voice + speed combinations and apply
  them in Speech Studio, Batch Studio and Document to Speech.
- **Command line generation** — `python -m scripts.generate`.
- **Interactive web application** — engine and voice selectors, text input,
  playback, WAV download, validation messages and a loading indicator.
- **Python library** — `from voiceforge import VoiceForgeTTS` and
  `from voiceforge.kokoro_backend import KokoroBackend`.
- **Objective evaluation tools** — Mel error, duration error, spectral detail,
  vocoder round-trip and audio defect detection.

## 3. How it works

### Kokoro-82M (default engine)

```
 English text → misaki G2P → phonemes → text encoder + duration predictor
              → style-conditioned features → iSTFTNet decoder → 24 kHz waveform
```

### VoiceForge custom model (experimental)

The full model, training configuration and measured limitations are described
in [docs/architecture.md](docs/architecture.md) and
[docs/training-v2.md](docs/training-v2.md).

| Stage | What happens | Where |
|---|---|---|
| Normalisation | Lower-case, ASCII, standard punctuation, collapsed whitespace | `voiceforge/text.py` |
| Encoding | 38-character vocabulary → integer IDs | `voiceforge/text.py` |
| Mel analysis | 80-band log-Mel, `n_fft=1024`, `hop=256` | `voiceforge/audio.py` |
| Acoustic model | Encoder → attention → decoder → postnet | `voiceforge/model.py` |
| Reconstruction | Inverse Mel scale + Griffin-Lim | `voiceforge/audio.py` |
| Application | Validation, model caching, HTTP interface | `voiceforge/app_core.py`, `voiceforge/web_app.py` |

The full architecture, including layer shapes, parameter counts, loss terms
and decoding modes, is documented in
[docs/architecture.md](docs/architecture.md).

### Model at a glance

| Component | Output | Parameters |
|---|---|---|
| Character embedding | (batch, text_len, 128) | 4,864 |
| 3 × Conv1d + BatchNorm + ReLU | (batch, 256, text_len) | 821,504 |
| Bidirectional GRU | (batch, text_len, 256) | 296,448 |
| Location-sensitive attention | context (batch, 256) | 193,632 |
| Decoder GRU + Mel/stop heads | (batch, 80) per frame | 632,913 |
| Postnet (4 × Conv1d) | residual correction | 862,704 |
| Duration head | log frames | 33,025 |
| **Total** | | **2,997,410** |

## 4. Repository structure

```
alm042/
├── app.py                        # Launch the web application
├── app_gradio.py                 # Optional Gradio front-end (needs gradio)
├── requirements.txt
├── README.md
├── PROJECT_SUMMARY.md            # Technical summary for report writing
├── checkpoints/                  # Model weights (git-ignored)
├── data/                         # LJSpeech dataset (git-ignored)
├── docs/
│   ├── api.md                    # Python API reference
│   ├── architecture.md           # Model and training design
│   ├── demo-plan.md              # 3-5 minute demonstration script
│   ├── evaluation.md             # Metrics, results and interpretation
│   ├── kokoro.md                 # Kokoro integration notes
│   └── report-outline.md         # Academic report outline and fact sheet
├── frontend/                     # Static UI: HTML + CSS + ES modules
│   ├── index.html                # Application shell
│   ├── css/                      # style.css, layout.css, components.css
│   ├── js/
│   │   ├── app.js                # Bootstrap, navigation, routing
│   │   ├── api.js                # All HTTP calls
│   │   ├── state.js              # Store + generation gate
│   │   ├── validation.js         # Text rules
│   │   ├── keyboard.js           # Enter / Shift+Enter logic
│   │   ├── engine-ui.js          # Engine/voice UI rules
│   │   ├── components/           # notifications.js, audio-player.js
│   │   └── pages/                # studio.js, coming-soon.js
│   └── tests/                    # Node unit tests (node --test)
├── outputs/                      # Generated audio (git-ignored)
├── scripts/
│   ├── check_audio.py            # Detect NaN, silence, clipping, buzz
│   ├── check_text_vocabulary.py  # Verify dataset text is encodable
│   ├── evaluate.py               # Objective model evaluation
│   ├── generate.py               # Command-line speech generation
│   └── train.py                  # Training entry point
├── tests/                        # Python tests (200), shared conftest
└── voiceforge/
    ├── __init__.py               # Public API
    ├── app_core.py               # Application logic (shared by UIs)
    ├── architectures.py          # Architecture identifiers
    ├── audio.py                  # Audio I/O, Mel, Griffin-Lim
    ├── batching.py               # Padding, collation, length bucketing
    ├── checkpoint.py             # Save/load and architecture dispatch
    ├── dataset.py                # LJSpeech loader + Mel cache
    ├── kokoro_backend.py         # Kokoro-82M pretrained backend
    ├── model.py                  # Tacotron-lite model
    ├── model_legacy.py           # Original baseline model (preserved)
    ├── text.py                   # Normalisation and vocabulary
    ├── training.py               # Training loop and losses
    ├── tts.py                    # VoiceForgeTTS inference pipeline
    └── web_app.py                # HTTP server: API + static assets
```

## 5. Installation

Requires **Python 3.11+**.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

PyTorch is listed without a version so pip installs a build matching your
machine. For CUDA, install the wheel from the official index first, for
example:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
```

Verify the environment:

```bash
python -c "import torch, torchaudio, soundfile; print(torch.__version__, torch.cuda.is_available())"
```

### Kokoro weights (first run)

`pip install -r requirements.txt` also installs the `kokoro` package. The
pretrained weights (~330 MB) are downloaded from Hugging Face the first time
the Kokoro engine is used, then cached. If huggingface.co is unreachable, use
a mirror:

```bash
set HF_ENDPOINT=https://hf-mirror.com      # Windows
export HF_ENDPOINT=https://hf-mirror.com   # Linux / macOS
```

Set `HF_HOME` to control where the cache lives. Details:
[docs/kokoro.md](docs/kokoro.md).

The web interface (`app.py`) uses only the Python standard library, so no
extra packages are needed. See [section 10](#10-running-the-application) for
the optional Gradio front-end.

## 6. Dataset setup

VoiceForge uses the public **LJSpeech 1.1** dataset (13,100 short English
clips, one female speaker, about 24 hours, 22,050 Hz).

1. Download it from <https://keithito.com/LJ-Speech-Dataset/>.
2. Extract it so that the folder layout is exactly:

```
data/
└── LJSpeech-1.1/
    └── LJSpeech-1.1/
        ├── metadata.csv
        └── wavs/
            ├── LJ001-0001.wav
            └── ...
```

3. Verify that every transcript can be encoded by VoiceForge's 38-character
   vocabulary:

```bash
python -m scripts.check_text_vocabulary
```

Expected output ends with `SUCCESS: All dataset transcripts are supported.`

The dataset is **not** included in this repository (several gigabytes). It is
excluded by `.gitignore`.

## 7. Model checkpoints

Checkpoints are also excluded from Git (about 34 MB each). Two are used:

| File | Description |
|---|---|
| `checkpoints/voiceforge_attn.pt` | **Current model.** Tacotron-lite, epoch 14, validation loss 1.3123 |
| `checkpoints/voiceforge.pt` | Original baseline (GRU + linear interpolation), kept for comparison |

To obtain them you can either:

* copy the trained files into `checkpoints/`, or
* train a fresh model with the command in [section 8](#8-training).

Checkpoints store their architecture and hyper-parameters, so both models load
automatically:

```python
from voiceforge import load_model

model, config, metadata = load_model("checkpoints/voiceforge_attn.pt")
print(config["architecture"], metadata["epoch"])
```

Checkpoints created before the duration predictor existed are detected and
reported with a clear error instead of being loaded incorrectly.

## 8. Training

Training is CPU/GPU agnostic and controlled from the command line:

```bash
python -m scripts.train \
    --max-samples 3000 \
    --val-samples 64 \
    --epochs 24 \
    --batch-size 32 \
    --max-seconds 5 \
    --learning-rate 0.001 \
    --checkpoint checkpoints/voiceforge_attn.pt \
    --history outputs/training_history.json
```

Options worth knowing:

| Option | Meaning |
|---|---|
| `--max-samples` | Number of training rows (0 = all 13,036 usable rows) |
| `--max-seconds` | Keep only clips shorter than this (speed on small GPUs) |
| `--batch-size` | Batch size; 32 fits comfortably in 6 GB |
| `--val-samples` | Held-out rows used for validation and best-checkpoint selection |
| `--arch` | `tacotron_lite` (default) or the legacy baseline |
| `--amp` | Mixed precision (measured slower for this model; off by default) |

What the last full run actually did:

| Setting | Value |
|---|---|
| Data | 3,000 clips ≤ 5.0 s |
| Epochs run | 14 (stopped when validation plateaued) |
| Batch size | 32 |
| Optimiser | Adam, lr 1e-3, gradient clipping 1.0 |
| Training time | 50.8 minutes on an RTX 3060 Laptop (6 GB, ~1.6 GB used) |
| Best validation loss | 1.3123 |

The script refuses to select the evaluator's held-out rows, saves the best
epoch, writes a JSON history, and verifies the saved checkpoint can be
reloaded before reporting success.

## 9. Generating speech

### Command line

```bash
python -m scripts.generate "Hello, welcome to VoiceForge!" --output outputs/hello.wav
```

Several sentences at once:

```bash
python -m scripts.generate --output-dir outputs/samples \
    "First sentence." "Second sentence."
```

The script validates the checkpoint, writes the WAV file, reads it back and
prints a JSON summary (duration, sample count, peak level).

### Python

```python
from voiceforge import VoiceForgeTTS

tts = VoiceForgeTTS.from_pretrained("checkpoints/voiceforge_attn.pt")
waveform = tts.synthesize("Hello, welcome to VoiceForge!")   # torch.Tensor
tts.save("Hello, welcome to VoiceForge!", "output.wav")
```

One-shot helper:

```python
from voiceforge import generate_speech

generate_speech("Hello world.", "output.wav")
```

The complete API (arguments, attributes, errors, timings) is documented in
[docs/api.md](docs/api.md).

### Kokoro-82M (pretrained engine)

```python
from voiceforge.kokoro_backend import KokoroBackend

tts = KokoroBackend(voice="af_heart")      # or af_bella, am_michael, ...
tts.save("Hello, this is Kokoro speaking through VoiceForge.", "out.wav")
```

24 kHz output, ~0.03 real-time factor after warm-up. See
[docs/kokoro.md](docs/kokoro.md).

## 10. Running the application

```bash
python app.py
```

Then open <http://127.0.0.1:7860>.

Useful options:

```bash
python app.py --engine kokoro --voice am_michael   # pretrained engine (default)
python app.py --engine voiceforge                  # student-built model
python app.py --engine auto --open                 # pick automatically
python app.py --device cpu                         # force CPU
python app.py --port 7861 --open
```

The interface provides:

- a **speech engine selector** (Kokoro / VoiceForge custom),
- a **Kokoro voice selector** (seven American English voices),
- a **speed slider** (0.5×–2.0×, Kokoro only; hidden for the custom model),
- an English text box with a character counter and example sentences,
- a **Generate Speech** button with a loading indicator,
- in-browser playback and a **Download WAV** link,
- clear validation and error messages, with a notice explaining which engine
  is pretrained and which is the student's own experimental model.

Everything runs locally; no text or audio leaves the machine.

### HTTP API

| Method | Path | Body | Purpose |
|---|---|---|---|
| GET | `/` | — | The application shell |
| GET | `/health` | — | Legacy status + engine summary |
| GET | `/api/engines` | — | **Source of truth**: engines, voices, languages, sample rates, speed ranges, limits, capabilities |
| POST | `/generate` | `{text, engine, voice?, speed?}` | Legacy generation endpoint |
| POST | `/api/generate` | `{text, engine, voice?, speed?}` | Generation endpoint (same implementation) |
| GET | `/audio/<file>` | — | Playback and download |
| POST | `/api/jobs` | `{type, items, engine, voice?, speed?}` | Queue a long-text or batch job |
| GET | `/api/jobs/{id}` | — | Job and per-item status |
| GET | `/api/jobs/{id}/audio/{item}` | — | One completed item (WAV) |
| GET | `/api/jobs/{id}/download` | — | ZIP with all completed audio + manifest |
| POST | `/api/uploads` | `{files: [{name, data_base64}]}` | Validate and decode `.txt` uploads |
| GET | `/api/library` | — | List/search/sort/filter recordings |
| GET | `/api/library/{id}` | — | One recording |
| PATCH | `/api/library/{id}` | `{name}` | Rename a recording |
| DELETE | `/api/library/{id}` | — | Delete the record and its WAV |
| GET | `/api/library/{id}/audio` | `?download=1` optional | Playback or download |
| GET | `/api/library/{id}/mp3` | `?bitrate=128\|192\|320` | Convert the stored WAV to MP3 |
| POST | `/api/documents/extract` | `{name, data_base64}` | Extract text from a PDF/DOCX |
| GET | `/api/pronunciation` | `?q=` | List/search pronunciation rules |
| POST | `/api/pronunciation` | `{word, pronunciation, description?}` | Add a rule |
| PATCH | `/api/pronunciation/{id}` | same fields | Edit a rule |
| DELETE | `/api/pronunciation/{id}` | — | Delete a rule |
| GET | `/api/presets` | — | List voice presets |
| POST | `/api/presets` | `{name, engine, voice?, speed?, description?}` | Create a preset |
| PATCH | `/api/presets/{id}` | same fields | Edit a preset |
| DELETE | `/api/presets/{id}` | — | Delete a preset |

### Long text, batches and uploads

- **Long Text** (Speech Studio → *Long text*): text up to **20,000
  characters**, split at sentence boundaries and synthesised as a background
  job, then joined into one WAV with a 150 ms gap between chunks.
- **Batch Studio**: up to **20 entries** of 20,000 characters each, one job,
  per-item progress, individual WAV downloads and a ZIP with a manifest.
- **TXT upload**: up to **20 files**, **1 MB per file**, **5 MB total**,
  UTF-8 (BOM tolerated). Extensions and encoding are validated on the server;
  uploaded names are sanitised and never used as paths.
- **Jobs**: one worker, at most **4 active jobs**; extra submissions get a
  `429 queue_full`. Statuses are `queued`, `running`, `completed`, `failed`.
  Progress is real item counts (no percentages). Jobs live in memory only.

```json
POST /api/jobs
{
  "type": "batch",
  "engine": "kokoro",
  "voice": "af_heart",
  "speed": 1.0,
  "items": [
    {"name": "Welcome", "text": "Welcome to VoiceForge."},
    {"name": "Intro", "text": "This is a longer introduction."}
  ]
}

202
{"ok": true, "job": {"id": "a447111fee72", "status": "queued", "total": 2,
 "completed": 0, "failed": 0, "download_url": null, "items": [...]}}
```

```json
// POST /api/generate
{"text": "Hello world.", "engine": "kokoro", "voice": "af_heart", "speed": 1.25}

// 200
{
  "ok": true,
  "text": "Hello world.",
  "filename": "eeb6873e1412.wav",
  "audio_url": "/audio/eeb6873e1412.wav",
  "duration_seconds": 2.27,
  "sample_rate": 24000,
  "engine": "kokoro",
  "voice": "af_heart",
  "speed": 1.25,
  "padded": false,
  "truncated": false
}

// 400 (one consistent error shape)
{"ok": false, "error": "Speed must be between 0.5 and 2.0 (got 3).", "code": "invalid_speed"}
```

Error codes: `invalid_engine`, `invalid_voice`, `invalid_speed`, `speed_not_supported`,
`empty_text`, `text_too_long`, `unsupported_characters`, `bad_request`,
`audio_not_found`, `checkpoint_missing`, `internal_error`.

### Optional Gradio front-end

A second front-end is included for the originally planned framework:

```bash
pip install gradio
python app_gradio.py
```

> **Note:** `app_gradio.py` could not be executed in the development
> environment because Gradio was not installed and the machine had no network
> access to install it. It uses the same application core as `app.py`, so the
> model integration is identical, but treat it as untested until you run it.
> `app.py` is the verified interface.

## 11. Evaluation

Reproduce the numbers in [docs/evaluation.md](docs/evaluation.md):

```bash
# Acoustic model, duration and vocoder metrics on 64 held-out sentences
python -m scripts.evaluate --checkpoint checkpoints/voiceforge_attn.pt \
    --samples 64 --vocoder-samples 4 --output outputs/evaluation.json

# Audio defect checks (NaN, silence, clipping, DC offset, flat buzz)
python -m scripts.check_audio --checkpoint checkpoints/voiceforge_attn.pt \
    --output-dir outputs/audio_checks
```

Summary of the measured comparison (identical 64 held-out samples):

| Metric | Baseline | Current model | Verdict |
|---|---|---|---|
| Mel L1 (lower better) | 3.02 | 5.70 | worse (exposure bias) |
| Duration error | 25.9 % | **14.3 %** | better |
| Spectral detail (target 1.0) | 0.48 | **0.61** | better |
| Vocoder Mel correlation | 0.979 | 0.979 | unchanged |
| Audio files passing all checks | 1 / 4 | **4 / 4** | better |

## 12. Testing

```bash
# Python: API, validation, static serving, path-traversal protection
python -m pytest tests -q

# JavaScript: keyboard rules, validation, state gate, engine UI rules
node --test frontend/tests
```

```
200 passed, 16 skipped      # Python (the 16 skipped are optional real-Kokoro tests)
100 passed                  # JavaScript
```

The optional Kokoro API tests load the real 82M model, so they are gated:

```bash
set VOICEFORGE_TEST_KOKORO=1
python -m pytest tests/test_kokoro_api.py -q
```

An earlier, larger pytest suite (18 files, 198 tests) was removed from the
working tree at the author's request and remains recoverable from Git:

```bash
git checkout 84ba9f6 -- tests
```

Manual checks:

```bash
python app.py --engine kokoro        # generate and listen
python app.py --engine voiceforge    # custom model
python -m scripts.generate "Hello." --output outputs/hello.wav
python -m scripts.check_audio --checkpoint checkpoints/voiceforge_attn.pt
```


### Audio Library

Every successful generation — Speech Studio, Long Text, each Batch item — is
copied into `outputs/library/` and recorded in `outputs/library.db`
(SQLite). Both are local application data and are ignored by Git.

| Query parameter | Meaning |
|---|---|
| `q` | Search in display name and original text (max 200 chars) |
| `sort` | `newest` (default), `oldest`, `name`, `duration` |
| `engine` | `kokoro` or `voiceforge` |
| `source` | `studio`, `long_text`, `batch` |
| `page`, `limit` | Pagination; `limit` is capped at 100 |

```json
GET /api/library?q=welcome&sort=name&source=studio&page=1&limit=10
{
  "ok": true,
  "records": [{"id": "4497b644…", "name": "Welcome", "engine": "kokoro",
               "voice": "af_heart", "speed": 1.0, "sample_rate": 24000,
               "duration_seconds": 2.8, "file_size": 134444,
               "source_type": "studio", "created_at": "2026-11-18T14:49:00+00:00",
               "original_text": "Welcome to VoiceForge."}],
  "pagination": {"page": 1, "limit": 10, "total": 1, "pages": 1}
}
```

- **Rename** changes only the display name (`PATCH`); the WAV filename never
  changes. Names are trimmed, limited to 80 characters and must be non-empty.
- **Delete** removes the database row and the library WAV. It refuses names
  outside the library folder, tolerates an already-missing file, and rolls
  the transaction back if the file cannot be removed.
- **Missing audio**: a record whose file disappeared still appears in the
  list; its audio endpoint returns a clear `404 audio_not_found` instead of
  breaking the page.
- **Persistence**: records live in SQLite, so they remain after a restart
  even though background job status is still in memory.


### Document to Speech

Upload a PDF or DOCX (max 10 MB), review the extracted text in an editable
box, then generate. The job runs through the existing Long Text pipeline and
the result is saved in the Audio Library.

- PDF text is extracted page by page with `pypdf`; blank pages are skipped.
- DOCX paragraphs, headings and tables are read in document order with
  `python-docx`; table cells are joined with ` | `.
- **Scanned PDFs are not supported**: they contain images, and OCR is not
  implemented. The API returns `no_extractable_text` with that explanation.
- Encrypted PDFs return `encrypted_document`; corrupted files return
  `invalid_document`; unsupported extensions return `unsupported_document`.
- Extracted text longer than 20,000 characters is rejected with a clear
  message instead of being silently truncated.

### Pronunciation dictionary

Rules are stored in SQLite (`pronunciation_entries`). During synthesis the
matched word or phrase is replaced with your respelling **before** chunking,
so Kokoro's text-to-phoneme step produces the intended sounds.

- Matching is case-insensitive, whole-word / whole-phrase only, longest match
  wins, punctuation is preserved and replacements are never applied
  recursively.
- This is a respelling approach, not raw phoneme injection: write the
  pronunciation the way it should sound (`Ab-dool-rah-man`). Kokoro's G2P
  converts it to phonemes; the custom engine receives the same replaced text.
- The toggle appears in Speech Studio, Batch Studio and Document to Speech.
  Jobs snapshot the dictionary once, so every item in a batch uses the same
  rules.
- The custom VoiceForge model applies the same text substitution, but its
  phoneme behaviour differs from Kokoro's.

### MP3 export

`lameenc` (installed from `requirements.txt`) provides the LAME encoder, so
no external FFmpeg is required. If it is missing, the server reports
`mp3_unavailable` with installation instructions and can fall back to FFmpeg
when it is on `PATH`.

- Bitrates: 128, 192 (default) or 320 kbps.
- Library recordings: `GET /api/library/{id}/mp3?bitrate=192`.
- Batch results: `GET /api/jobs/{id}/download?format=mp3` (ZIP) or
  `GET /api/jobs/{id}/audio/{item}?format=mp3`.
- Conversions are cached under `outputs/mp3/`; the original WAV is never
  modified and remains downloadable.

### Voice presets

Presets (`voice_presets`) store a name, engine, voice and speed. They are
validated against the live `GET /api/engines` data, so invalid engine/voice
combinations and out-of-range speeds are rejected. The custom engine stores
no voice and no speed. Presets appear in Speech Studio, Batch Studio and
Document to Speech; applying one only changes the controls — it never starts
generation.

### Database migrations

The library database uses `PRAGMA user_version`:

| Version | Contents |
|---|---|
| 1 | `audio_records` and its indexes |
| 2 | adds `pronunciation_entries`, `voice_presets` and their indexes |

Opening an older database upgrades it in place; existing audio records are
never deleted or rewritten.


## 13. Known limitations

### Phase 5 features

1. **No OCR** — scanned PDFs cannot be read; only selectable text is
   extracted.
2. **Pronunciation is respelling, not IPA injection.** It changes the text
   fed to the G2P step; exact pronunciation depends on how the respelling is
   written. Kokoro's phonemizer may still normalise unusual spellings.
3. **The custom engine has no speed or voice selection**, and its prosody is
   much weaker than Kokoro's.
4. **MP3 needs an encoder**: `lameenc` is listed in `requirements.txt`; if it
   is removed, FFmpeg is required and the API returns `mp3_unavailable`.
5. **MP3 files are cached** under `outputs/mp3/` and never committed; they
   are not tracked as library records.
6. **No preset import/export** and no per-item presets in a batch job.

### Audio Library

1. **Single-user local database.** No accounts, no locking beyond SQLite's
   own; it is designed for one machine.
2. **No bulk delete or bulk export** in this phase.
3. **Jobs stay in memory** — their status and ZIP links disappear after a
   restart, although the library records and WAVs remain.
4. **Deleting a library record does not delete the job ZIP**, which still
   contains its own copy of the audio.
5. **Search is `LIKE`-based**, fine for thousands of rows but not a
   full-text index.

### Jobs and batch processing

1. **In-memory jobs.** Job state is not persisted; restarting the server
   clears completed jobs (finished WAVs stay on disk under `outputs/jobs/`).
2. **One worker, four jobs.** Jobs run one at a time; the queue rejects the
   fifth active job with `429 queue_full`.
3. **No cancellation.** A running job cannot be stopped from the UI.
4. **20,000 characters per item**, 20 items per job.
5. **ZIP is built in memory** — fine for the current limits, not for very
   large batches.
6. **Long text is split at sentences** and joined; prosody resets at each
   chunk, so a long passage may sound slightly segmented.

### Kokoro-82M (main engine)

1. **Pretrained, not ours.** The model was trained by hexgrad; this project
   integrates and evaluates it. No fine-tuning was performed.
2. **First-run download.** ~330 MB from Hugging Face unless the cache is
   pre-populated.
3. **English-first.** Only American English voices are exposed.
4. **Grapheme-to-phoneme errors.** Unusual proper nouns can be mispronounced
   by `misaki`/`espeak-ng`.
5. **No prosody control** beyond voice choice and a global speed factor.

### VoiceForge custom model (experimental)

1. **Not intelligible.** The student confirmed by listening that no words are
   understandable, despite strong alignment improvements in v2.
2. **Small training budget.** 6,000 clips and 5,000 steps is far below what a
   production Tacotron needs.
3. **Exposure bias.** Teacher-forced error 0.38, free-running 5.14 in v2.
4. **Griffin-Lim ceiling.** A phase-reconstruction baseline, not a neural
   vocoder.
5. **English, single speaker.** 38-character ASCII vocabulary and no digits.
6. **CPU inference is slow.** GPU generation takes ~0.4–0.7 s per sentence.
7. **Gradio front-end untested** (see section 10).

## 14. Future improvements

- Train much longer on the full dataset (10k+ steps) and use scheduled
  sampling to reduce exposure bias.
- Replace location-sensitive attention with forward attention or stepwise
  monotonic attention for faster, more reliable alignment.
- Replace Griffin-Lim with a small neural vocoder (HiFi-GAN).
- Move from characters to phonemes for better pronunciation.
- Add multi-speaker support and prosody controls.

## 15. Documentation

| Document | Contents |
|---|---|
| [docs/kokoro.md](docs/kokoro.md) | Kokoro-82M: why, architecture, licence, install, performance, comparison |
| [docs/architecture.md](docs/architecture.md) | Custom model design, losses, training configuration, limitations |
| [docs/training-v2.md](docs/training-v2.md) | The v2 training run: configuration, speed-ups, measured results |
| [docs/evaluation.md](docs/evaluation.md) | Metric definitions, before/after results, interpretation |
| [docs/api.md](docs/api.md) | Python API reference, errors, timings |

### Acknowledgements

- **Kokoro-82M** by hexgrad — <https://huggingface.co/hexgrad/Kokoro-82M> —
  Apache-2.0. Used as the pretrained main engine; **not trained by this
  project**.
- **LJSpeech 1.1** — Keith Ito and Linda Johnson, public domain dataset.
- **Tacotron** (Wang et al., 2017) and **Tacotron 2** (Shen et al., 2018) —
  architecture inspiration for the custom model.
- **Chorowski et al., 2015** — location-sensitive attention.
- **Griffin & Lim, 1984** — phase reconstruction algorithm.
- **PyTorch** and **torchaudio** for the implementation platform.

The technical report for the course is prepared separately and is not part of
this repository yet.

## 16. MP3 cache cleanup

MP3 files are generated on demand into `outputs/mp3/` and reused while their
source WAV is unchanged. At startup the server removes cached MP3 files that
are older than **7 days**, then trims the cache to **256 MB**, oldest first.

The policy is deliberately narrow: only `*.mp3` files directly inside
`outputs/mp3/` are eligible. Original WAV files, the SQLite database, the
Audio Library and job folders are never touched, so a referenced recording
can never be deleted by cleanup. The removed file count and freed bytes are
printed in the server log.

## 17. Clean installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

- Python **3.11+**; tested with 3.11.9 on Windows.
- PyTorch (CPU or CUDA) is the largest dependency; a clean install may
  download about 2 GB. A CUDA wheel is not required — CPU works.
- The first Kokoro start needs **internet access** to fetch the 82 MB model
  plus its voices (~330 MB total). After that the cached weights are reused;
  if the default Hugging Face cache is not writable, the project falls back
  to `outputs/hf/` automatically.
- `pip check` reports no broken requirements for the tested environment, and
  `pip install --dry-run -r requirements.txt` resolves cleanly.

## 18. University delivery

- **Technical summary for report writing:** [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- **Report outline and verified fact sheet:** [docs/report-outline.md](docs/report-outline.md)
- **3–5 minute demonstration script:** [docs/demo-plan.md](docs/demo-plan.md)
- **Kokoro integration notes:** [docs/kokoro.md](docs/kokoro.md)

The report outline contains only measured facts and lists the sources that
must be cited (Kokoro model card, Tacotron paper, Griffin–Lim, SQLite,
pypdf, python-docx, LAME, RFC 9110). It explicitly warns against claiming
OCR support, production-quality custom speech, or unmeasured performance
numbers.

## 19. Licence

- Application code in this repository: student coursework.
- **Kokoro-82M** weights: Apache-2.0, © hexgrad; see the model card at
  <https://huggingface.co/hexgrad/Kokoro-82M>. Attribution is retained here
  and in `docs/kokoro.md`.
- LJSpeech, if re-downloaded for training, is public domain.
- Python dependencies keep their own licences (PyTorch BSD, pypdf BSD,
  python-docx MIT, lameenc LGPL through LAME, etc.).

## 20. Known limitations (summary)

1. English only; Kokoro voices are US English.
2. Scanned PDFs are not supported (no OCR).
3. Pronunciation rules are respelling-based, not phoneme injection.
4. Job history is in memory; recordings persist in the library.
5. The custom Tacotron-lite model is experimental and audibly robotic.
6. One generation at a time (shared model lock); long jobs queue.
7. MP3 requires `lameenc` (listed) or FFmpeg.

