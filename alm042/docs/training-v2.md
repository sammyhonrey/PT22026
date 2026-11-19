# VoiceForge v2 — Training Run

This document records the training configuration and measured behaviour of the
second acoustic model (`checkpoints/voiceforge_v2.pt`). It exists so the run
can be reproduced exactly and so the numbers in the report can be traced back
to a real execution.

## Why a second training run

The first model (`voiceforge_attn.pt`) was trained for only 14 epochs on
3,000 clips of at most 5 seconds. Diagnostics showed:

- attention stayed diffuse (mean maximum weight 0.081 over 154 characters),
- teacher-forced Mel error 0.515 versus free-running 6.03 (11.7× gap),
- generated audio was low-passed (spectral centroid ~550 Hz versus ~1,500 Hz
  for real speech) and had three times too much amplitude modulation.

The diagnosis pointed at under-training and weak alignment pressure rather
than at the architecture or the vocoder (the Griffin-Lim round-trip on real
Mel reached 0.98 correlation).

## What changed relative to v1

| Item | v1 | v2 |
|---|---|---|
| Training clips | 3,000 (≤ 5 s) | **6,000 (≤ 7 s)** |
| Audio used | ~3.9 h | **8.2 h** |
| Optimiser steps | ~1,300 | **5,000** |
| Batch size | 32 | **48** |
| Learning rate | 1e-3 constant | **1e-3, 300-step warm-up, cosine to 1e-4** |
| Guided-attention weight | 0.5 | **1.0** |
| Guided-attention sigma | 0.4 | **0.3** |
| Mel cache | no | **yes** |
| Batch construction | random | **length-bucketed** |
| Architecture | Tacotron-lite | **identical** |

The architecture was deliberately left untouched so that any change in
quality can be attributed to training rather than to a new model.

## Configuration actually used

```
python -m scripts.train \
    --max-samples 6000 --val-samples 64 --epochs 40 \
    --batch-size 48 --max-seconds 7 \
    --learning-rate 0.001 --warmup-steps 300 --min-lr 0.0001 \
    --guided-weight 1.0 --guided-sigma 0.3 \
    --cache-dir outputs/mel_cache \
    --checkpoint checkpoints/voiceforge_v2.pt \
    --history outputs/training_history_v2.json \
    --log-every 25 --seed 42 --keep-awake
```

| Setting | Value |
|---|---|
| Training rows | 6,000 (metadata indices 1 … 11,129) |
| Validation rows | 64 (indices 11,132 … 11,269) |
| Test rows | 64 (indices 13,036 … 13,099) — unchanged held-out window |
| Mean / median clip | 4.88 s / 5.12 s |
| Mean Mel frames | 421 |
| Batches per epoch | 125 |
| Total optimiser steps | 5,000 |
| Device | RTX 3060 Laptop, 6 GB |
| Peak VRAM observed | ~4.3 GB |
| Time per epoch | ~3.0–3.5 min |

## Speed-ups and what they changed

**Mel cache.** Log-Mel tensors are computed once and stored under
`outputs/mel_cache/` (0.77 GB). Later epochs load them instead of decoding a
WAV and recomputing an STFT, cutting roughly 18 ms of CPU work per sample to
about 1.5 ms. The cached data is identical to the uncached data (verified by
a test).

**Length-bucketed batches.** Batches are padded to their longest sample, so
sorting by length before batching reduces padding waste. Batches are formed
from shuffled groups of eight neighbouring batches, which keeps composition
varied between epochs.

**Batch 48 instead of 32.** The decoder runs one sequential step per Mel
frame, so per-step time is dominated by launch overhead and is nearly flat in
batch size (measured: batch 8, 16 and 32 all took ~1.3 s per step). A larger
batch therefore processes more samples per second until memory or compute
saturates; batch 48 kept peak VRAM near 4.3 GB of 6 GB. Batch 64 was tried
first and reached 5.6 GB, which was too close to the limit with long-clip
batches, so it was reduced.

**Keep-awake.** The training process asks Windows to keep the session awake
(`SetThreadExecutionState`), because a sleeping laptop would kill a
multi-hour run.

## Training outcome

| Item | Value |
|---|---|
| Epochs completed | **40 / 40** |
| Total time | **125.2 minutes** |
| Peak GPU memory | **3.67 GB** of 6 GB |
| Final training loss | 1.2902 |
| Best validation loss | **1.0978** (epoch 40) |
| Optimiser steps | 5,000 |
| Final learning rate | 1e-4 (cosine floor) |
| Final loss terms | mel 0.796, postnet 0.488, stop 0.0040, duration 0.0054, attention 0.0013 |

Validation fell from 3.12 (epoch 1) to 1.10 (epoch 40) and was still improving
slowly at the end (~0.003 per epoch), so more training would likely help a
little, but with diminishing returns.

## Measured comparison against v1 (same 64 held-out sentences)

| Metric | v1 | v2 | Verdict |
|---|---|---|---|
| Mel L1 | 5.7049 | **4.9920** | −12.5 %, better |
| Duration relative error | 14.31 % | **12.11 %** | better |
| Duration MAE (frames) | 64.94 | 80.20 | worse |
| Spectral detail ratio | 0.6102 | 0.6166 | unchanged |
| Vocoder correlation | 0.9788 | 0.9788 | unchanged |
| Generated energy variation | 2.0025 | 2.1224 | unchanged |

### Alignment (the main target) improved substantially

Same seven sentences as the diagnostic:

| Metric | v1 | v2 | Change |
|---|---|---|---|
| Attention max weight | 0.081 | **0.236** | 2.9× sharper |
| Attention entropy | 0.86 | **0.525** | more focused |
| Diagonal error | 0.234 | **0.062** | 3.8× closer to the diagonal |
| Character coverage | 17.9 % | **43.8 %** | 2.4× |
| Monotonic fraction | 0.91 | **0.964** | better |
| Teacher-forced L1 | 0.515 | **0.384** | better |
| Free-running L1 | 6.029 | **5.137** | −15 % |

Attention now follows a proper diagonal through the sentence instead of
hovering over a few characters. This is the structural improvement the first
model was missing.

### Audio character

Three generated sentences were analysed with the same spectral measurements
used in the diagnostic:

| Sample | Centroid | Energy above 4 kHz | Energy variation |
|---|---|---|---|
| v2 "Hello, welcome to VoiceForge." | **1334 Hz** | **0.121** | 0.50 |
| v2 "This is a test of the speech synthesis system." | **1814 Hz** | **0.208** | 0.34 |
| v2 "Where are you going this evening?" | 675 Hz | 0.002 | 2.80 |
| Real LJSpeech (reference) | 1372–1662 Hz | 0.12–0.14 | 0.71–0.85 |
| v1 (reference) | ~550 Hz | 0.004 | 2.8 |

Two of the three sentences now have a spectral balance close to real speech
instead of the dull 550 Hz buzz of v1. The third sentence still collapses
into the old failure mode, so the improvement is **not uniform**.

### Where v2 is still worse

- **Duration is now systematically short**: predicted/true frame ratio is
  0.77–0.87 across the seven diagnostic sentences (v1 averaged 1.09). The
  model speaks too fast.
- **Mel correlation against the human recordings dropped** (0.245 vs
  0.33–0.41 for v1). Correlation is dominated by the smooth spectral
  envelope, which v1 reproduced well *because* it was over-smoothed; v2 adds
  sharper detail in slightly the wrong places, and its 15 % short duration
  misaligns it. This metric therefore does not support a quality claim on its
  own.
- Free-running L1 (5.14) is still far above teacher-forced L1 (0.38), so
  exposure bias remains the dominant remaining defect.

## Conclusion for this run

Training longer on more data with stronger alignment pressure **worked for
alignment**: attention is now diagonal and focused, teacher-forced error
improved, and two of three analysed samples have speech-like spectra.

It did **not** solve exposure bias, and it introduced a duration bias toward
short outputs. Whether the audio is now intelligible can only be decided by
listening to `outputs/quality_review_v2/`; the objective evidence is
promising but not conclusive, and no intelligibility claim is made here.

