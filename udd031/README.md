# Multimodal Mental State Analyser

**Emotion and stress prediction by fusing speech audio and transcribed text using Hugging Face Transformers**

> **Author** · UDDIN MEJBAH — `20243120031`
> **Programme** · Software Engineering
> **Course** · Professional Training 2(`YN3012170034`)
> **Project type** · individual, Python, Gradio web application

---

## 1. What this is

A single short recording goes in. Three opinions come out.

The application transcribes the speech, reads the **voice** with an acoustic
transformer, reads the **words** with a text transformer, and reconciles the two into
one emotion estimate plus a stress band — while reporting how much the two channels
**disagree** and how much the system genuinely **does not know**.

The interesting part is not that two models are combined. It is that the combination
is *auditable*: every number on screen can be traced to a formula, a configuration
value, and a stated assumption. Nothing is hidden behind a black box, and nothing is
claimed that was not measured.

| | |
|---|---|
| **Emotion** | 6 classes — happy, sad, angry, fear, disgust, neutral |
| **Stress** | Low / Medium / High, with a full contributor decomposition |
| **Explainability** | per-branch scores, per-class fusion weights, conflict index, ignorance |
| **Interface** | Gradio, six information zones, microphone or file upload |
| **Weights** | downloaded from the Hugging Face Hub at runtime — none committed |

### ⚠️ Ethical position, stated up front

This is an **experimental research prototype and screening aid**. It is **not** a
medical device, it does **not** diagnose any condition, and it **cannot** replace
assessment by a psychologist or psychiatrist. It must not be used for decisions about
employment, insurance, education, or care. Emotion recognition is culturally biased,
easily fooled, and frequently wrong; a system that displays a confident number about
someone's mental state carries a duty to say so plainly. That is why the disclaimer is
always visible in the interface rather than hidden behind a click.

---

## 2. Quick start

```bash
cd udd031
python -m pip install -r requirements.txt      # torch, transformers, gradio, librosa...
python scripts/download_models.py              # optional: pre-fetch and verify weights
python scripts/train_acoustic_probe.py         # optional: refit the 6-class head (~4 min)
python app.py                                  # http://127.0.0.1:7860
```

Open the URL, record ~5 seconds of speech (or upload a clip), press **Analyse clip**.

The third line is optional: if `models/acoustic_probe.npz` is absent, the acoustic
branch falls back to the published four-class head automatically and the interface
reports the reduced coverage instead of pretending otherwise.

With `make`:

```bash
make setup     # install dependencies
make models    # pre-download + verify label vocabularies
make samples   # generate the synthetic probe signals
make run       # serve the app
make test      # run the suite
```

**First run takes 60–120 s** while three checkpoints download. `make models` moves
that cost to an explicit setup step and verifies that each checkpoint still exposes the
label vocabulary the ontology adapter expects — an upstream rename would otherwise
appear as a confusing runtime error.

### Environment options

`.env.example` documents every variable. The application reads **real environment
variables only** — it does not load a `.env` file itself, so set the value in your
shell before starting rather than copying the file:

```bash
# PowerShell
$env:MMSA_DEVICE = "cpu"; python app.py
# bash / zsh
MMSA_DEVICE=cpu python app.py
```

Everything is optional; all values have working defaults.

| Variable | Default | Meaning |
|---|---|---|
| `MMSA_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `MMSA_CACHE_DIR` | HF default | where weights are cached (keep outside the repo) |
| `MMSA_LOG_LEVEL` | `INFO` | `DEBUG` … `ERROR` |
| `MMSA_DIMENSIONAL` | `0` (disabled) | optional arousal/dominance/valence head (~1.2 GB). Off by default: that checkpoint's regression head does not load through `AutoModelForAudioClassification`, so its outputs would be random. See `config/stress.yaml`. |
| `MMSA_SEED` | `1337` | determinism seed |
| `MMSA_ACOUSTIC_PROBE` | `models/acoustic_probe.npz` | path to the fitted six-class head; set to an empty string to force the published four-class fallback |

---

## 3. Pipeline

```

**1 · The label-ontology adapter keeps its ignorance.**
The published acoustic head was trained on four classes (`ang, hap, neu, sad`); the
target space has six. The adapter projects native logits into canonical space with a
sparse matrix whose **rows may sum to less than one**, so the projection returns both a
vector *and* its **coverage**. The unmapped remainder is never renormalised away inside
a branch, because a model with no vocabulary for `fear` must not appear confident about
`anger`. Only the fused vector is normalised, exactly once.

In the shipped system the branch is served by a **fitted six-class head** over the same
frozen encoder (`models/acoustic_probe.npz`, `scripts/train_acoustic_probe.py`), so
coverage is 1.0 and `fear` / `disgust` are reachable. The adapter still matters: it is
what keeps the published head honest whenever the fitted artefact is missing and the
four-label fallback runs instead.

**2 · Fusion weights reliability per class, not per model.**
```
r_m(c) = ρ_m(c) · coverage_m^γ · confidence_m^δ      (ρ = per-class modality prior)
r_text(c) ← r_text(c) · max(0.02, evidence(transcript))
p(c)   ∝ Σ_m r_m(c) · p̃_m(c)
```
A branch must be *competent, expressive and certain* before it dominates — and it may
dominate for one emotion while deferring on another. On top of that, the text branch's
vote is scaled by the **lexical evidence** its transcript actually carries
(`src/mmsa/evidence.py`): a sentence with no affect word and no self-reference ("Kids are
talking by the door") keeps 2 % of its reliability and may not assert a conflict index
either. Without that gate the branch voted *anger 0.84* on every RAVDESS clip and
dragged fusion 19 points below the acoustic branch — measured, not assumed
(`scripts/evaluate_fusion_ablation.py`). The conflict index is the normalised
Jensen–Shannon divergence, withheld entirely when a branch is too blind, or too
unevidenced, for the comparison to be meaningful.

**Stress is a composite, not a classifier.** Seven terms (proxy arousal, proxy valence,
pitch instability, spectral arousal, lexical negativity, transcript uncertainty, conflict)
combine through a logistic link. It is reported **with its contributors** and **without
an accuracy claim**, because it was not trained against stress labels. Its two band
boundaries are the 33rd and 67th percentile of a measured distribution
(`scripts/calibrate_stress.py`), not hand-set guesses.

---

## 4. Models

All weights are loaded from the Hub at runtime. Nothing is vendored; the one learned
artefact in the repository is 49.4 KB.

| Role | Checkpoint | Native labels | Licence |
|---|---|---|---|
| ASR | [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en) | — | MIT |
| Acoustic | [`superb/wav2vec2-base-superb-er`](https://huggingface.co/superb/wav2vec2-base-superb-er) | `ang, hap, neu, sad` | Apache-2.0 |
| Acoustic head *(fitted locally)* | `models/acoustic_probe.npz` — 6 × 1,536 linear map over the frozen encoder | six canonical classes | project code |
| Text | [`bhadresh-savani/distilbert-base-uncased-emotion`](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) | `sadness, joy, love, anger, fear, surprise` | Apache-2.0 |
| Dimensional *(optional)* | [`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) | `arousal, dominance, valence` | CC BY-NC-SA 4.0 |

Measured on speakers the fitted head never saw (RAVDESS actors 21–24, 208 clips):
**0.688 accuracy, 0.675 macro-F1**, against 0.267 / 0.156 for the published four-label
head and 0.167 chance. The number, the split and the per-class recalls are in
`results/acoustic_probe.json`; reproduce with `python scripts/train_acoustic_probe.py`.

> **Note on the brief.** The original specification named a checkpoint
> `wav2vec2-gem-speech-emotion` that does not exist on the Hub. Rather than ship a
> broken identifier, this build uses `superb/wav2vec2-base-superb-er`, which is the
> model that specification was evidently describing, and documents the substitution in
> the report's model-selection section. Every identifier above was resolved against the
> live Hub and its label vocabulary verified.

---

## 5. Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song.
Licence **CC BY-NC-SA 4.0**, DOI [`10.5281/zenodo.1188976`](https://doi.org/10.5281/zenodo.1188976).

- Record page — <https://zenodo.org/records/1188976>
- Speech archive (≈200 MB) — <https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1>
- Citation — Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLOS ONE, 13(5), e0196391.

**No dataset archive is committed to this repository.** `demo_samples/` holds six
short RAVDESS excerpts — one per canonical emotion, each from a different **held-out**
speaker (actors 21–24), mono 16 kHz, the whole utterance capped at four seconds, well
under 1 MB in total — so the interface can be demonstrated on real speech with a known
ground-truth label *without* scoring itself on a voice the acoustic head was trained on.
They are rebuilt by `python scripts/build_demo_clips.py` and redistributed with
attribution under CC BY-NC-SA 4.0. The ~200 MB archive itself is opt-in:

```bash
python scripts/fetch_ravdess.py --print-links
python scripts/fetch_ravdess.py --download          # ~200 MB
python scripts/fetch_ravdess.py --from-zip data/raw/Audio_Speech_Actors_01-24.zip
```

---

## 6. Layout

```
udd031/
├── app.py                     entry point: python app.py
├── config/
│   ├── ontology.yaml          native → canonical projection matrices
│   ├── fusion.yaml            γ, δ, conflict bands, modality priors, evidence gate
│   ├── stress.yaml            composite weights, measured band thresholds
│   ├── models.yaml            which acoustic head artefact to load
│   └── lexicon.yaml           valence lexicon for the lexical term + evidence gate
├── models/
│   └── acoustic_probe.npz     fitted 6-class head, 49.4 KB (git-ignored; rebuild below)
├── src/mmsa/
│   ├── schemas.py             frozen result types — the single source of truth
│   ├── errors.py              twelve error codes, each with a user-facing message
│   ├── config.py              YAML → frozen dataclasses, env overrides
│   ├── audio_io.py            layered decoding, validation, SNR, silence trim
│   ├── asr.py                 Whisper backends (transformers / openai-whisper)
│   ├── acoustic_branch.py     fitted six-class head over the frozen wav2vec2 encoder
│   ├── text_branch.py         DistilBERT emotion head
│   ├── evidence.py            lexical evidence of a transcript — the fusion gate
│   ├── ontology.py            coverage-preserving projection
│   ├── fusion.py              per-class reliability pooling, conflict index
│   ├── prosody.py             f0, energy, pauses, ZCR, syllable rate
│   ├── stress.py              transparent composite + contributor decomposition
│   ├── narrative.py           deterministic natural-language explanation
│   ├── pipeline.py            orchestration; dependency-injectable
│   ├── charts.py              Plotly figures
│   └── ui/                    theme, CSS, fragments, page assembly
├── scripts/                   download, training, calibration, evaluation, report
├── tests/                     102 tests, no models or network required
├── demo_samples/              six held-out RAVDESS excerpts + provenance
```

---

## 7. Testing

```bash
python -m pytest -q        # 102 passed
```

The suite is built so that **nothing needs a model, a network connection, or an audio
device**. Pure-maths modules (ontology, fusion, stress, narrative, prosody, audio I/O)
are tested directly; the pipeline is driven end-to-end with injected fakes that assert
stage ordering, error translation and idempotence. That is why CI can be offline and why
a marker can verify the logic in seconds.

---

## 8. Evaluation — what is measured and what is not

Two scripts form a genuine measurement chain:

```bash
python scripts/train_acoustic_probe.py      # fit + measure the six-class acoustic head
python scripts/evaluate_zeroshot.py         # published heads only, per-branch accuracy / macro-F1
python scripts/evaluate_fusion.py           # recombine measured outputs across strategies
python scripts/evaluate_fusion_ablation.py  # evidence gate on/off, 208 held-out clips
python scripts/calibrate_stress.py          # where the stress score actually falls -> thresholds
python scripts/benchmark_latency.py         # per-stage timing
python scripts/make_figures.py              # render report figures
```

`evaluate_zeroshot.py` writes per-sample branch distributions to
`results/branch_probs.json`; `evaluate_fusion.py` consumes them and compares unimodal,
uniform, confidence-weighted, credal and oracle strategies, plus a γ/δ sensitivity sweep.
`train_acoustic_probe.py` writes `models/acoustic_probe.npz` and
`results/acoustic_probe.json`; `evaluate_fusion_ablation.py` writes
`results/fusion_ablation.json`; `calibrate_stress.py` writes
`results/stress_calibration.json` and, with `--write`, the thresholds in
`config/stress.yaml`.

**These require labelled RAVDESS audio and a working model stack.** If the data or
weights are absent, the scripts say exactly what is missing and exit non-zero — they do
**not** emit placeholder metrics. Any result cell in the report marked `TODO(measure)`
is a deliberate admission that a number was not obtained on the machine that produced
this submission, and is discussed under Limitations. Reporting an unmeasured figure as
if it were measured would invalidate the entire document.

---

## 9. Limitations

Stated plainly, because a project that admits its edges is more credible than one that
does not:

- The **acoustic head is fitted here**, not published: 0.688 accuracy / 0.675 macro-F1 on
  speakers it never saw, which is honest but modest. Disgust is the weakest class
  (0.52 recall; 8 of 32 held-out clips read as anger).
- The **text branch cannot be validated on acted corpora**: every RAVDESS clip carries
  one of two affect-free carrier sentences, so the branch is muted by design there
  (§3.4.2) and its real-world behaviour on spontaneous speech is unmeasured.
- The **stress index has no ground truth**. It is a transparent composite with face
  validity and an explicit decomposition, not a validated predictor; even its band
  boundaries are only percentile-calibrated on 24 utterances.
- **Culture, language, age and recording conditions** all shift prosody; `whisper-small.en`
  is English-only.
- Utterances are capped at 30 s; the system analyses a fragment of behaviour, not a person.
- Only six excerpts ship, so a broader claim needs the full corpus rather than the
  bundled examples. On those six the fitted head matches three labels exactly — the
  demonstration shows that error rate rather than hiding it.

---

**UDDIN MEJBAH** · 20243120031 · Software Engineering · Professional Training (YN3012170034)
