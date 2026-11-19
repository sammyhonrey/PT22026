# VoiceForge Evaluation

This document explains how VoiceForge is evaluated, what each number means,
and what the measurements showed for **both** models. The numbers come from
real runs of `scripts/evaluate.py`; nothing here is estimated.

Evaluation is split into three parts. A text-to-speech system can fail in
three different places, and mixing the measurements together hides which
part is broken.

| Part | Question it answers |
|---|---|
| Acoustic model | Is the predicted Mel spectrogram close to real speech, and is the predicted duration right? |
| Vocoder | How much detail does Griffin-Lim lose when it reconstructs *real* speech? |
| Generated audio | Does audio produced from text have speech-like dynamics? |

## How to run

```bash
python -m scripts.evaluate \
    --checkpoint checkpoints/voiceforge_attn.pt \
    --samples 64 \
    --vocoder-samples 4 \
    --n-iter 32 \
    --output outputs/evaluation.json
```

The script writes a JSON file and prints the same content.

### Held-out samples

The evaluator uses the **last 64 rows of `metadata.csv`** (indices 13036 to
13099). Training uses a prefix of the dataset and `scripts/train.py`
explicitly refuses to select rows inside this window, so these samples are
never seen during training.

## Metric definitions

### `mel_l1` (lower is better)

Mean absolute error between the predicted and real log-Mel spectrogram, over
valid (non-padded) frames. Units are natural-log power. A perfect model
scores 0.

### `duration_mae_frames` and `duration_relative_error` (lower is better)

Difference between the predicted number of Mel frames and the true number.
One frame is 256 samples (11.6 ms) at 22,050 Hz.

### `spectral_detail_ratio` (closer to 1.0 is better)

Standard deviation of the predicted Mel divided by that of the real Mel. A
value well below 1 means an over-smoothed, blurry prediction.

### `mel_rmse` and `mel_correlation` (vocoder only)

A **real** Mel spectrogram is converted to audio with Griffin-Lim and back.
The input was correct, so any error measures the vocoder alone.

### `generated_energy_variation`

Coefficient of variation (std ÷ mean) of short-term energy over 1024-sample
windows. Real speech is well above zero because syllables alternate loud and
quiet. A flat buzz is near zero.

## Model A - baseline (`legacy_interpolate`)

Original GRU model with linear time interpolation. 2 epochs on 128 samples.
Checkpoint: `checkpoints/voiceforge.pt`.

| Measurement | Value |
|---|---|
| `mel_l1` | 3.0158 |
| `duration_mae_frames` | 151.56 |
| `duration_relative_error` | 25.86 % |
| `predicted_mel_std` / `target_mel_std` | 1.8968 / 3.9511 |
| `spectral_detail_ratio` | 0.4801 |
| Vocoder `mel_rmse` / `mel_correlation` | 0.8584 / 0.9789 |
| `generated_energy_variation` | 0.0999 |
| `real_energy_variation` | 0.7967 |
| Average generated length | 5.43 s |

## Model B - attention model (`tacotron_lite`)

New Tacotron-lite model, 14 epochs on 3,000 LJSpeech clips of at most 5 s.
Best validation loss 1.3123. Checkpoint: `checkpoints/voiceforge_attn.pt`.
Training configuration: see `docs/architecture.md`.

Identical command and identical 64 held-out samples.

| Measurement | Value |
|---|---|
| `mel_l1` | **5.7049** |
| `duration_mae_frames` | **64.94** |
| `duration_relative_error` | **14.31 %** |
| `predicted_mel_std` / `target_mel_std` | 2.4109 / 3.9511 |
| `spectral_detail_ratio` | **0.6102** |
| Vocoder `mel_rmse` / `mel_correlation` | 0.8610 / 0.9788 |
| `generated_energy_variation` | **2.0025** |
| `real_energy_variation` | 0.7967 |
| Average generated length | 7.61 s |

## Before / after comparison

| Metric | Baseline | Attention | Change | Verdict |
|---|---|---|---|---|
| `mel_l1` (lower better) | 3.0158 | 5.7049 | +89 % | **worse** |
| `duration_relative_error` (lower better) | 25.86 % | 14.31 % | −45 % | **better** |
| `duration_mae_frames` (lower better) | 151.56 | 64.94 | −57 % | **better** |
| `spectral_detail_ratio` (target 1.0) | 0.4801 | 0.6102 | +27 % | **better, still low** |
| Vocoder `mel_correlation` | 0.9789 | 0.9788 | unchanged | unchanged |
| `generated_energy_variation` (real ≈ 0.80) | 0.0999 | 2.0025 | — | **see below** |

### Honest interpretation

**Two clear improvements.** The attention model predicts sentence length
much more accurately (14 % vs 26 % error) and its spectrograms contain 27 %
more spectral detail. It has started to learn the text-to-audio mapping
rather than emitting one average spectrum.

**One clear regression.** `mel_l1` got worse. This is not a trick of the
numbers; it has a concrete cause:

* The training loss (teacher forced, where the decoder sees the true
  previous frame) is only **0.87** at the last epoch.
* Free-running generation, which is what the evaluator measures, scores
  **5.70**.

The gap is *exposure bias*: during generation the decoder consumes its own
imperfect predictions, small errors accumulate, and the spectrogram drifts
away from the target. L1 error also punishes any timing shift heavily, so a
model with more detail in slightly the wrong place can score worse than a
blurred average. The baseline's over-smoothed output happens to minimise L1
against an uncertain target - it is "safely wrong".

**The energy result needs care.** The baseline output was a nearly constant
buzz: energy variation 0.10 against 0.80 for real speech, and 3 of 4 test
sentences failed the `flat_buzz` check. The attention model's output varies
strongly (2.00-2.87), so it is **no longer a flat buzz**. However 2.0-2.9 is
*above* real speech (0.80), which means the dynamics are erratic - sharp
bursts and near-silent gaps - not natural-sounding prosody. This is typical
of an under-trained attention model and should not be described as natural
speech.

**Attention is still diffuse.** A diagnostic on a held-out sentence (154
characters, 811 frames) shows the average maximum attention weight is only
**0.022** (a uniform distribution over 154 characters is 0.0065). The
attention has not locked onto individual characters yet, which is exactly
why the output is not intelligible. With more data and training this is the
part that must improve.

### Conclusion

The new architecture is a genuine step forward on **timing** and **spectral
detail**, and it removes the flat buzz. It is **not** intelligible speech,
and it is worse than the baseline on raw `mel_l1`. Both facts are reported
here rather than hiding the regression.

## Generated-audio defect checks

`scripts/check_audio.py` generates four sentences and checks for NaN/Inf
samples, silence, clipping, DC offset and flat-buzz dynamics.

```bash
python -m scripts.check_audio --checkpoint <checkpoint> \
    --output-dir outputs/audio_checks --json outputs/audio_checks.json
```

| Check | Baseline | Attention model |
|---|---|---|
| Files passing all checks | 1 / 4 | **4 / 4** |
| Non-finite (NaN/Inf) samples | 0 | 0 |
| Silent files | 0 | 0 |
| Clipped files | 0 | 0 |
| DC offset | within limit | within limit |
| `flat_buzz` failures | **3 / 4** | 0 / 4 |
| Energy variation range | 0.09 - 0.16 | 2.63 - 2.87 |
| RMS (peak normalised to 0.95) | 0.18 - 0.20 | 0.036 - 0.043 |

The attention model's lower RMS at the same peak means a high crest factor:
short loud peaks separated by quiet parts. Again, this is structured but
erratic, not natural speech.

## Limitations of these metrics

These measurements are automatic and objective, but they **do not measure
intelligibility**. A spectrogram can have low error and still be
unintelligible, and vice versa. A proper evaluation would need a listening
test (for example MOS scoring with several listeners), which was not
performed. No subjective quality claim is made anywhere in this repository.

---

## Kokoro-82M: measured reference

The application's main engine is the pretrained Kokoro-82M model. It is
evaluated differently from the custom model because it produces waveforms,
not Mel spectrograms, so the Mel-based metrics above do not apply.

| Measurement | Value |
|---|---|
| Output | 24,000 Hz, 16-bit PCM |
| Steady-state real-time factor | **0.03** (~30x faster than real time) |
| Peak GPU memory | 0.73 GB |
| First-load download | ~330 MB (cached afterwards) |
| Audio defects (5 samples) | none (no NaN, silence, clipping or DC offset) |

The custom model remains the object of the Mel/duration/alignment evaluation
in this document. Full Kokoro details, including the voice list and the
objective checks, are in `docs/kokoro.md`.
