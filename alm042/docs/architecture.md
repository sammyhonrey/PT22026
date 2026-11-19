# VoiceForge Model Architecture

This document explains the acoustic model used by VoiceForge: what it is
made of, why each part exists, and how it is trained. It covers both models
in the repository so the two can be compared.

## 1. The problem the model solves

A text-to-speech system has to turn a sentence (a sequence of characters)
into a Mel spectrogram (a grid of 80 frequency bands × hundreds of time
frames). The two sequences have very different lengths and there is no
simple rule that says which frame belongs to which character — that mapping
is called **alignment**, and learning it is the hard part of TTS.

## 2. Baseline: why it was replaced

The original model (`voiceforge/model_legacy.py`) is:

1. character embedding,
2. bidirectional GRU over the characters,
3. a linear layer to 80 Mel bands,
4. `F.interpolate` to stretch the character sequence to the target number
   of frames.

Step 4 is the problem. Stretching is uniform: every character gets the same
share of time and no character ever "owns" a specific part of the audio.
The network can therefore learn an average spectral shape but never
pronunciation or timing. Measured on 64 held-out sentences it produced a
spectrogram with only **48 %** of the spectral detail of real speech
(`spectral_detail_ratio` in `docs/evaluation.md`).

The baseline is kept unchanged so the old checkpoint still loads and so the
comparison is reproducible.

## 3. Current model: Tacotron-lite

`voiceforge/model.py` implements a small, from-scratch version of the
Tacotron idea (Wang et al., 2017; Shen et al., 2018). "Lite" means the
sizes were reduced to train on a 6 GB laptop GPU — not that a component is
missing.

```
 characters (batch, text_len)
        │
        ▼
 Embedding(38 → 128)
        │
        ▼
 3 × [Conv1d(5) + BatchNorm + ReLU + Dropout]
        │
        ▼
 BiGRU(128 → 2×128)  ──► Linear ──► encoder memory (batch, text_len, 256)
        │                                   │
        │                                   ├──► duration head ──► log(frames)
        │                                   │
        │            location-sensitive attention ◄── previous attention
        ▼                                   │
   previous Mel frame ──► Prenet ──┐        │ context (256)
                                   ▼        ▼
                            concat (512) ──► decoder GRU (256)
                                                  │
                              concat(decoder, context) = 512
                                    │                     │
                                    ▼                     ▼
                             Linear → 80 Mel        Linear → stop
                                    │
                                    ▼
                     Postnet (4 × Conv1d) → residual added
```

### 3.1 Encoder

| Layer | Output shape | Parameters |
|---|---|---|
| Embedding (38 → 128, padding_idx=0) | (B, L, 128) | 4,864 |
| 3 × Conv1d(k=5, 256) + BN + ReLU + Dropout | (B, 256, L) | 821,504 |
| BiGRU (128 hidden per direction) | (B, L, 256) | 296,448 |
| Linear projection | (B, L, 256) | 65,792 |

The convolutions look at five neighbouring characters at once, which helps
with letter combinations. The bidirectional GRU then gives every character
information about the whole sentence, so a character can be pronounced
differently depending on context.

### 3.2 Attention

`LocationSensitiveAttention` decides, for each output frame, which
characters are being spoken. It computes a score for every character:

```
score(t, j) = v · tanh( W_q · decoder_state_t
                      + W_m · memory_j
                      + W_l * conv(previous_attention) )

attention_t = softmax(score(t, :))
context_t   = Σ_j attention_t[j] · memory_j
```

Two details matter:

* **Location features.** The previous attention weights are convolved and
  added to the score. This discourages attention from jumping backwards or
  forwards randomly and encourages it to move through the sentence in order.
  This is the mechanism from Chorowski et al. (2015) that Tacotron 2 uses.
* **Masking.** Padding positions get a score of −10⁴ before the softmax, so
  they receive exactly zero attention. A test checks this.

Attention parameters: 193,632.

### 3.3 Decoder

The decoder runs one step per output frame:

1. the previous Mel frame goes through a two-layer **Prenet** (dropout 0.1),
2. attention produces a context vector,
3. a single-layer GRU updates its hidden state from `[prenet, context]`,
4. two linear heads produce the Mel frame and the stop probability.

| Layer | Parameters |
|---|---|
| Prenet | 86,528 |
| Decoder GRU | 591,360 |
| Mel projection | 41,040 |
| Stop projection | 513 |

The stop head lets the model decide when the sentence is finished. It is
trained with binary cross-entropy, with the target set to 1 on the last real
frame.

### 3.4 Postnet

Four 1-D convolutions predict a correction that is added to the decoder
output. This sharpens the blurry frame-by-frame prediction.

Parameters: 862,704.

**Total: 2,997,410 parameters.**

### 3.5 Duration predictor

A small head mean-pools the encoder memory and predicts `log(number of Mel
frames)`. It is trained with MSE against the true log length.

At inference it is used for **length-controlled decoding**: instead of
relying only on the stop token (which an under-trained model handles badly),
the pipeline asks the duration predictor how long the sentence should be and
runs the decoder for exactly that many frames. Fully autoregressive decoding
with the stop token is still available as
`model.generate(..., use_stop_token=True)`.

## 4. Training objective

For each batch the loss is:

```
L = L1(mel_before, target)
  + L1(mel_postnet, target)
  + 1.0 × BCE(stop_logits, stop_targets)
  + 0.1 × MSE(predicted_log_length, log(true_length))
  + 0.5 × L_guided_attention
```

All Mel terms ignore padded frames, so short sentences are not penalised for
batching zeroes.

**Guided attention** is the term that makes small-scale training work. It
penalises attention that is far from the diagonal:

```
L_guided = mean( attention(t, j) · (1 − exp(−(t/T − j/L)² / (2σ²))) ), σ = 0.4
```

Without it, an attention model trained on a few thousand clips often never
becomes monotonic. With it, alignment becomes roughly diagonal within the
first epochs, and attention can then refine itself.

## 5. Training configuration actually used

| Setting | Value |
|---|---|
| Dataset | LJSpeech 1.1, clips **≤ 5.0 s** |
| Training samples | 3,000 |
| Validation samples | 64 (held out from training) |
| Batch size | 32 (drop_last) |
| Epochs | 24 |
| Optimizer | Adam, lr 1e-3 |
| Gradient clipping | 1.0 |
| Precision | float32 (mixed precision was measurably slower here) |
| Seed | 42 |
| GPU | RTX 3060 Laptop, 6 GB |
| GPU memory observed | ~1.6 GB |
| Time per epoch | ~200 s |

Why only 3,000 clips of ≤5 s? Two reasons, both practical:

1. **Speed.** The decoder is a Python loop over frames. Longer clips mean
   more loop iterations per batch, so short clips give more gradient updates
   per minute — the main constraint on a 6 GB GPU.
2. **Memory and stability.** Short clips keep the attention matrix small and
   make early training less likely to diverge.

This is a deliberate limitation, not a claim that 3,000 clips is enough for
high-quality speech. It is documented in `docs/evaluation.md` and in the
report.

The held-out evaluation rows (the last 64 rows of `metadata.csv`, indices
13036–13099) are never selected for training; `scripts/train.py` explicitly
refuses to select them.

## 6. Inference pipeline

```
text → normalize → character IDs → encoder → duration predictor
     → decoder (fixed length, autoregressive on its own Mel)
     → postnet → log-Mel → Griffin-Lim → waveform → WAV
```

Griffin-Lim is a classical iterative phase-reconstruction algorithm, **not**
a neural vocoder. It estimates the phase that is missing from a magnitude
spectrogram. It is the standard baseline, and its output is noticeably
buzzer than a trained vocoder such as WaveNet or HiFi-GAN. This is a
deliberate scope decision: the project is about building the acoustic model,
and a neural vocoder would be a separate large project.

## 7. Compatibility with the old pipeline

The public interface is unchanged:

```python
mel = model(text, text_lengths, mel_lengths)   # fixed length
mel = model(text, text_lengths)                # duration-predicted length
lengths = model.predict_mel_lengths(text, text_lengths)
```

`VoiceForgeTTS`, `scripts/generate.py`, `scripts/evaluate.py` and the tests
work with both architectures. Checkpoints record `architecture` and
`model_config`, so `load_model()` rebuilds the right class automatically;
old checkpoints without a config are treated as the legacy architecture.

`scripts/train.py` now builds the model through the same `build_model()`
factory that checkpoint loading uses, and after training it loads the saved
checkpoint back and checks the architecture and parameter count. Training
and inference therefore cannot silently disagree about the model.

Checkpoints created before the duration predictor existed (the early
`smoke_test.pt` artifact) are not compatible with the current model. The
loader detects the missing weights and reports an actionable error instead
of failing with a raw state-dict dump. Those files are historical
smoke-test artefacts, not usable models.

## 8. Known limitations

* **Small training set.** 3,000 short clips and ~80 minutes of training are
  far below what Tacotron needs (typically tens of hours, ~10k+ steps).
  The model learns rough alignment and speech-like spectral structure, not
  clear pronunciation.
* **Griffin-Lim ceiling.** Even with a perfect Mel spectrogram, Griffin-Lim
  output sounds buzzy.
* **English only, single speaker.** The vocabulary is 38 ASCII characters
  and the model was trained only on LJSpeech's single female speaker.
* **No prosody control.** Duration is predicted, not controllable.

---

## 9. Speech engines in the application

The application ships two engines behind one interface
(`voiceforge/app_core.py`):

| Engine | Implementation | Output |
|---|---|---|
| `kokoro` (default) | `voiceforge/kokoro_backend.py`, pretrained Kokoro-82M | 24 kHz waveform (iSTFTNet) |
| `voiceforge` | `voiceforge/tts.py`, the student-built Tacotron-lite | 80-band Mel → Griffin-Lim, 22.05 kHz |

`VoiceForgeApp` validates text per engine (the custom model's 38-character
vocabulary rejects digits; Kokoro accepts them), loads models lazily, and
serialises CUDA work with a lock. The web layer passes `engine` and `voice`
through `/generate`; `/health` reports both engines. Selection falls back to
the custom model when Kokoro is unavailable, so the application always has a
working engine. See `docs/kokoro.md` for the pretrained model's details.
