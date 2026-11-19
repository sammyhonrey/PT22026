# Kokoro-82M Backend

VoiceForge uses **Kokoro-82M** as its main speech engine. Kokoro is a
pretrained, open-weight text-to-speech model released by **hexgrad**. It was
**not** trained by this project. The student's contribution is the
integration, the application, the testing and the evaluation — plus the
separate custom VoiceForge acoustic model kept as an experimental backend.

| | |
|---|---|
| Model | [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| Code | [github.com/hexgrad/kokoro](https://github.com/hexgrad/kokoro) |
| Parameters | 82 million |
| Output | 24,000 Hz mono |
| Weights licence | Apache-2.0 (see the model card) |
| Our role | Integration, interface, tests, documentation, comparison |

## Why Kokoro was selected

1. **Size.** 82 M parameters run comfortably on a 6 GB laptop GPU and even on
   a CPU, so the application stays portable.
2. **Quality per parameter.** Kokoro is designed to deliver natural speech at
   a small size, which is exactly the gap the student's own model hit.
3. **Open weights and permissive licence.** Apache-2.0 allows academic use
   and redistribution of the integration.
4. **Simple API.** A single `KPipeline` call returns audio, which keeps the
   integration honest and small.
5. **No cloud dependency.** Everything runs locally after the one-time
   weight download.

## How it generates speech

Kokoro follows the StyleTTS 2 style of design:

1. **Text → phonemes.** English text is normalised and converted to phonemes
   by `misaki` (with `espeak-ng` available as a fallback for some words).
2. **Phonemes → acoustic features.** A transformer-based text encoder and a
   duration predictor turn the phoneme sequence into aligned features. A
   per-voice **style vector** conditions the prosody, which is why different
   `af_*` / `am_*` voices sound different with the same weights.
3. **Features → waveform.** An **iSTFTNet** decoder produces the waveform
   directly at 24 kHz, so no separate Griffin-Lim step is needed.

This is the key architectural difference from the student's model: Kokoro
predicts a waveform with a learned decoder, while VoiceForge predicts an 80-
band Mel spectrogram and reconstructs the phase with Griffin-Lim.

## Installation

```bash
pip install kokoro
```

`kokoro` pulls in `misaki` (grapheme-to-phoneme), `transformers`,
`huggingface_hub`, and `espeakng-loader`, which supplies the `espeak-ng`
library on Windows — no separate installer is required.

The first run downloads roughly **330 MB** of weights and voice files from
Hugging Face into the `huggingface_hub` cache. If `huggingface.co` is not
reachable, point the client at a mirror:

```bash
set HF_ENDPOINT=https://hf-mirror.com      # Windows
export HF_ENDPOINT=https://hf-mirror.com   # Linux / macOS
```

Set `HF_HOME` to choose where the weights are cached.

## Usage

### In the application

```bash
python app.py                 # Kokoro selected automatically when installed
python app.py --engine kokoro --voice am_michael
python app.py --engine voiceforge   # the student's custom model instead
```

The interface has an engine selector and a Kokoro voice selector.

### From Python

```python
from voiceforge.kokoro_backend import KokoroBackend

tts = KokoroBackend(voice="af_heart")
tts.save("Hello, this is Kokoro speaking through VoiceForge.", "out.wav")
```

### Voices exposed by the interface

| Voice | Description |
|---|---|
| `af_heart` | US female — warm and clear (default) |
| `af_bella` | US female — expressive |
| `af_nicole` | US female — soft |
| `af_sarah` | US female — neutral |
| `am_michael` | US male — calm |
| `am_adam` | US male — deeper |
| `am_onyx` | US male — low |

Kokoro ships many more voices; the list above is the curated American-English
subset exposed in the UI.

## Measured behaviour on this machine

RTX 3060 Laptop (6 GB), PyTorch 2.14.1+cu130, Python 3.11.9:

| Measurement | Value |
|---|---|
| First load (weights cached) | ~5 s |
| First-load download (uncaught) | ~48 s, 312 MB |
| Steady-state generation | **0.19–0.21 s for 5.5–7.8 s of audio** |
| Real-time factor (steady state) | **0.03** (~30× faster than real time) |
| Peak GPU memory | **0.73 GB** |
| Output format | 24,000 Hz, 16-bit PCM WAV |
| Clipping / silence / NaN | none observed |

The first one or two sentences of each process are slower (RTF 0.6–1.5)
because of kernel warm-up and itSTFT JIT compilation.

## Objective checks performed

Five sentences were generated with `af_heart`, `af_bella` and `am_michael`
(`outputs/kokoro_test/`):

| File | Audio length | Generation time | Peak | RMS |
|---|---|---|---|---|
| `01_hello_af_heart.wav` | 4.35 s | 6.39 s (first run) | 0.38 | 0.049 |
| `02_news_af_bella.wav` | 6.55 s | 4.23 s | 0.51 | 0.053 |
| `03_question_am_michael.wav` | 4.60 s | 3.37 s | 0.36 | 0.042 |
| `04_numbers_af_heart.wav` | 5.55 s | 0.19 s | 0.47 | 0.049 |
| `05_paragraph_af_bella.wav` | 7.78 s | 0.21 s | 0.41 | 0.057 |

All files passed the project's audio-defect checks: no non-finite samples, no
silence, no clipping, no DC offset.

**Caveat:** the author of this integration cannot listen to the audio. Validity
and timing are measured; naturalness and intelligibility must be judged by a
person listening to the files above. No claim of naturalness is made here
without that listening step.

## How it compares with the custom VoiceForge model

| | Kokoro-82M | VoiceForge custom |
|---|---|---|
| Trained by | hexgrad (pretrained) | **the student** |
| Parameters | 82 M | 3.0 M |
| Training data | large proprietary/curated corpus | 8.2 h LJSpeech subset |
| Output | 24 kHz waveform (iSTFTNet) | 80-band Mel → Griffin-Lim |
| Generation speed | ~0.03 RTF | ~0.15–0.5 RTF |
| Alignment quality | production-grade | improved but imperfect (v2: 44 % coverage) |
| Intelligibility | to be confirmed by listening | **not intelligible** (confirmed by the student) |
| Role in project | main engine | experimental comparison / evidence of development |

The custom model remains fully available: training, evaluation, checkpoints
and tests are untouched. It is what demonstrates the student's own work on
TTS internals; Kokoro is what makes the application useful.

## Limitations

- **English-first.** The integration exposes American English voices only.
- **One-time download.** About 330 MB from Hugging Face; offline machines
  need a pre-populated cache or a mirror.
- **G2P dependency.** `misaki`/`espeak-ng` decide pronunciation; unusual
  proper nouns can still be mispronounced.
- **Not fine-tuned.** Kokoro's prosody cannot be steered beyond the voice
  style and the speed factor (0.5×–2.0×, applied per request through
  `KPipeline.__call__`; the loaded model is reused).
- **Licence.** Weights are Apache-2.0; retain the model card attribution in
  any redistribution.

## Attribution

> Kokoro-82M by hexgrad — <https://huggingface.co/hexgrad/Kokoro-82M> —
> Apache-2.0. Used as a pretrained engine; not trained by this project.

## Troubleshooting

### `WinError 5` / `Access is denied` on the Hugging Face cache

On some Windows accounts (and in sandboxed environments) the default cache
folder `%USERPROFILE%\.cache\huggingface` cannot be written to, so loading
the weights fails with:

```
[WinError 5] Access is denied: 'C:\Users\<you>\.cache\huggingface\hub\...'
```

The application now checks this automatically before importing Kokoro. If the
default location is not writable it selects a project-local cache
(`outputs/hf`, git-ignored) and prints:

```
Hugging Face cache: ...\alm042\outputs\hf (the default cache location is not writable)
```

No action is needed. To force a specific location anyway, set `HF_HOME`
yourself; an explicit value is always respected:

```powershell
$env:HF_HOME = "D:\hf-cache"
python app.py
```

If the weights are not cached yet and `huggingface.co` is unreachable, also
set `HF_ENDPOINT=https://hf-mirror.com`.

### `HF_TOKEN` warning

"Unauthenticated requests to the HF Hub" is informational; downloading the
Kokoro weights does not require a token.
