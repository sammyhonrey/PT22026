# VoiceForge Python API

VoiceForge can be used as a normal Python library. This page documents the
public interface; everything else in the package is internal.

## Quick start

```python
from voiceforge import VoiceForgeTTS

tts = VoiceForgeTTS.from_pretrained("checkpoints/voiceforge_attn.pt")
tts.save("Hello, welcome to VoiceForge!", "output.wav")
```

`save` writes a 22,050 Hz 16-bit PCM WAV file and returns the waveform
tensor.

## One-shot helper

For a single sentence in a script:

```python
from voiceforge import generate_speech

path = generate_speech("Hello world.", "output.wav")
print(path)   # output.wav
```

This loads the checkpoint on every call. If you generate more than one
sentence, create a `VoiceForgeTTS` once and reuse it — loading the model
takes a few seconds, generation takes about half a second.

## `VoiceForgeTTS`

### `VoiceForgeTTS.from_pretrained(checkpoint_path, device=None, **options)`

| Argument | Meaning |
|---|---|
| `checkpoint_path` | Path to a `.pt` checkpoint produced by `scripts/train.py` |
| `device` | `"cpu"`, `"cuda"`, or `None` to choose automatically |
| `griffin_lim_iterations` | Reconstruction quality (default 32) |
| `max_mel_frames` | Safety cap on generated length (default 2200 ≈ 25 s) |

The checkpoint stores its own architecture and hyper-parameters, so both the
attention model and the legacy baseline can be loaded without extra flags.

### `synthesize(text) -> torch.Tensor`

Returns a one-dimensional waveform at `tts.sample_rate`.

```python
waveform = tts.synthesize("Hello world.")
print(waveform.shape)   # torch.Size([61440])
```

### `save(text, output_path) -> torch.Tensor`

Generates speech and writes it to `output_path` (parent folders are
created automatically).

### Attributes

| Attribute | Meaning |
|---|---|
| `sample_rate` | 22050 |
| `device` | `torch.device` used for inference |
| `was_truncated` | `True` if the last output hit the frame cap |
| `was_padded` | `True` if the last output was too short and was padded |
| `checkpoint_path` | Path passed to `from_pretrained` |
| `model_config` | Architecture dictionary read from the checkpoint |

## `load_model(checkpoint_path)`

Lower-level loader. Returns `(model, config, metadata)` and is what
`from_pretrained` uses internally.

```python
from voiceforge import load_model

model, config, metadata = load_model("checkpoints/voiceforge_attn.pt")
print(config["architecture"], metadata["epoch"])
```

## Text utilities

```python
from voiceforge import normalize_text, text_to_sequence

normalize_text("  HELLO   World!  ")   # 'hello world!'
text_to_sequence("hello")              # [8, 5, 12, 12, 15]
```

The vocabulary is 38 characters: lowercase letters, space, and
`. , ! ? - ' " ; : ( )`. Digits and other symbols are rejected.

## Errors

| Situation | Exception |
|---|---|
| Checkpoint file does not exist | `FileNotFoundError` |
| Checkpoint does not match the architecture | `ValueError` |
| `text` is not a string | `TypeError` |
| `text` is empty or whitespace | `ValueError` |
| `text` contains unsupported characters | `ValueError` |

All messages are written to be shown directly to a user.

## Performance

Measured on this project's machine (RTX 3060 Laptop, 6 GB):

| Operation | Time |
|---|---|
| `from_pretrained` (model load) | ~2-3 s |
| `synthesize` (one sentence, GPU) | ~0.4-0.7 s |
| `synthesize` (one sentence, CPU) | several seconds |

## Quality

The model is small and trained on a limited subset of LJSpeech. The audio is
recognisably synthesised speech but is **not expected to be fully
intelligible**. See `docs/evaluation.md` for the measurements behind that
statement.
