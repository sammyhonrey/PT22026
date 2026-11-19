#!/usr/bin/env python
"""Fit the six-class acoustic head from frozen wav2vec2 embeddings.

RAVDESS speech is embedded with the SUPERB encoder (mean+std pooling), the eight
acted emotions are mapped onto the six canonical labels, and multinomial logistic
regression is fitted with C chosen on actors 19-20 and scored on 21-24. Each clip
is embedded twice, at natural gain and at -3 dBFS, because the application
normalises uploads and the archive does not.

    python scripts/train_acoustic_probe.py [--refresh] [--no-augment]

Writes ``models/acoustic_probe.npz`` and ``results/acoustic_probe.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "data" / "raw" / "Audio_Speech_Actors_01-24.zip"
CACHE = ROOT / "data" / "embeddings_wav2vec2.npz"
OUT_MODEL = ROOT / "models" / "acoustic_probe.npz"
OUT_JSON = ROOT / "results" / "acoustic_probe.json"

CANONICAL = ("happy", "sad", "angry", "fear", "disgust", "neutral")

# RAVDESS emotion code -> canonical label. 08 (surprised) is excluded: the
# ontology has no surprise class and inventing one would be a label error.
EMOTIONS = {
    "01": "neutral",
    "02": "neutral",  # calm
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
}

TRAIN_ACTORS = tuple(f"{i:02d}" for i in range(1, 19))   # 01-18
VAL_ACTORS = ("19", "20")
TEST_ACTORS = ("21", "22", "23", "24")

SAMPLE_RATE = 16_000
MAX_SECONDS = 4.0
TARGET_PEAK = 0.7079
"""Every clip is scaled to this peak (-3 dBFS) before the encoder runs.

This is not cosmetic: ``mmsa.audio_io.load_audio`` already normalises every uploaded
clip to -3 dBFS, so a head trained on un-normalised waveforms sees a different input
level at inference and changes its answer because of gain alone. Measured during
development: a clip at half amplitude moved the head from ``angry 0.94`` to
``disgust 0.49``. The same constant is applied in ``mmsa.acoustic_branch``.
"""
MODEL_ID = "superb/wav2vec2-base-superb-er"
SEED = 1337


def normalise_level(wave: np.ndarray, target_peak: float = TARGET_PEAK) -> np.ndarray:
    """Scale a waveform to a fixed peak, exactly as ``audio_io.load_audio`` does.

    Idempotent: a clip that is already at -3 dBFS comes out unchanged, which is why
    the branch can apply it on top of ``load_audio`` without double-normalising.
    """
    x = np.asarray(wave, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if not np.isfinite(peak) or peak < 1e-6:
        return x
    return np.clip(x * (target_peak / peak), -1.0, 1.0).astype(np.float32)


# Data


def _read_clips(zip_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (actor ids, canonical labels, raw waveforms) for every kept clip."""
    import soundfile as sf

    items: list[tuple[str, str, bytes]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".wav"):
                continue
            parts = Path(name).stem.split("-")
            if len(parts) != 7 or parts[0] != "03":
                continue
            emotion = EMOTIONS.get(parts[2])
            if emotion is None:
                continue
            items.append((parts[6], emotion, zf.read(name)))

    actors, labels, waves = [], [], []
    for actor, emotion, blob in items:
        data, sr = sf.read(io.BytesIO(blob), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        actors.append(actor)
        labels.append(emotion)
        waves.append(np.asarray(data[: int(SAMPLE_RATE * MAX_SECONDS)], dtype=np.float32))
    return actors, labels, waves


# Embeddings


def _embed(waves: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
    """Mean+std pooled last hidden state of the frozen SUPERB-ER encoder."""
    import torch
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID, local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID, local_files_only=True).eval()

    out: list[np.ndarray] = []
    started = time.perf_counter()
    for start in range(0, len(waves), batch_size):
        batch = waves[start : start + batch_size]
        inputs = extractor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
        frame_mask = model._get_feature_vector_attention_mask(hidden.shape[1], inputs["attention_mask"])
        mask = frame_mask.unsqueeze(-1).float()
        count = mask.sum(1).clamp(min=1)
        mean = (hidden * mask).sum(1) / count
        var = (((hidden - mean.unsqueeze(1)) ** 2) * mask).sum(1) / count
        out.append(torch.cat([mean, var.clamp(min=0).sqrt()], dim=1).numpy())
        done = start + len(batch)
        if done % 160 == 0 or done == len(waves):
            rate = done / max(time.perf_counter() - started, 1e-9)
            print(f"  embedded {done}/{len(waves)}  ({rate:.1f} clips/s)", flush=True)
    return np.concatenate(out).astype(np.float32)


def build_cache(
    zip_path: Path, cache_path: Path, quick: bool, refresh: bool, augment_gain: bool = True
) -> dict:
    if cache_path.exists() and not refresh and not quick:
        data = np.load(cache_path, allow_pickle=True)
        print(f"cache   : {cache_path} ({data['x'].shape[0]} clips)")
        return {"actors": data["actors"], "labels": data["labels"], "x": data["x"]}

    print(f"reading : {zip_path}")
    actors, labels, waves = _read_clips(zip_path)
    if quick:
        keep = np.isin(actors, list(TRAIN_ACTORS[:4]) + list(VAL_ACTORS) + list(TEST_ACTORS[:1]))
        actors = [a for a, k in zip(actors, keep) if k]
        labels = [l for l, k in zip(labels, keep) if k]
        waves = [w for w, k in zip(waves, keep) if k]
    print(f"kept    : {len(waves)} clips  {dict(Counter(labels))}")

    # The app normalises uploads to -3 dBFS and the archive is not at that level, so
    # each clip is embedded at both gains with the same label. Two identical labels
    # stop the head relying on recording level.
    if augment_gain:
        waves = waves + [normalise_level(w) for w in waves]
        actors = list(actors) + list(actors)
        labels = list(labels) + list(labels)
        print(f"augment : {len(waves)} clips after gain augmentation (natural + -3 dBFS)")

    x = _embed(waves)
    print(f"features: {x.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, x=x, actors=np.array(actors), labels=np.array(labels))
    print(f"cached  : {cache_path}")
    return {"actors": np.array(actors), "labels": np.array(labels), "x": x}


# Fit


def fit(actors: np.ndarray, labels: np.ndarray, x: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler

    train = np.isin(actors, TRAIN_ACTORS)
    val = np.isin(actors, VAL_ACTORS)
    test = np.isin(actors, TEST_ACTORS)
    if not (train.any() and val.any() and test.any()):
        raise SystemExit("split is empty -- is the cache from --quick? re-run without it")

    scaler = StandardScaler().fit(x[train])
    xtr, xval, xte = scaler.transform(x[train]), scaler.transform(x[val]), scaler.transform(x[test])

    grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    scores = []
    for c in grid:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        clf.fit(xtr, labels[train])
        macro = f1_score(labels[val], clf.predict(xval), average="macro")
        scores.append((macro, c, clf))
        print(f"  C={c:<6} val macro-F1 {macro:.3f}")

    best_macro, best_c, clf = max(scores, key=lambda t: t[0])

    # Refit on train+val at the selected strength: model selection is finished,
    # and withholding those speakers from the fit would waste 10% of the corpus.
    refit_mask = train | val
    scaler = StandardScaler().fit(x[refit_mask])
    xr, xte = scaler.transform(x[refit_mask]), scaler.transform(x[test])
    final = LogisticRegression(C=best_c, max_iter=3000, class_weight="balanced").fit(xr, labels[refit_mask])

    pred = final.predict(xte)
    acc = float((pred == labels[test]).mean())
    macro_f1 = f1_score(labels[test], pred, average="macro")
    per_class = {
        c: {
            "support": int((labels[test] == c).sum()),
            "recall": float((pred[labels[test] == c] == c).mean()) if (labels[test] == c).any() else None,
        }
        for c in CANONICAL
    }
    labels_order = list(CANONICAL)
    cm = confusion_matrix(labels[test], pred, labels=labels_order).tolist()

    print(f"\nchosen  : C={best_c} (val macro-F1 {best_macro:.3f})")
    print(f"val     : actors {', '.join(VAL_ACTORS)}  ({val.sum()} clips)")
    print(f"test    : actors {', '.join(TEST_ACTORS)}  ({test.sum()} clips)")
    print(f"TEST    : accuracy {acc:.3f}   macro-F1 {macro_f1:.3f}   (chance 0.167)")
    for c in CANONICAL:
        r = per_class[c]["recall"]
        print(f"  {c:<8} n={per_class[c]['support']:>3}  recall {r:.2f}" if r is not None else f"  {c:<8} n=0")

    return {
        "model": final,
        "scaler": scaler,
        "c": best_c,
        "accuracy": acc,
        "macro_f1": float(macro_f1),
        "per_class": per_class,
        "confusion": cm,
        "labels_order": labels_order,
        "n_train": int(refit_mask.sum()),
        "n_test": int(test.sum()),
        "val_macro_f1": float(best_macro),
    }


def save(result: dict, out_model: Path, out_json: Path, actors: np.ndarray, labels: np.ndarray) -> None:
    classes = list(result["model"].classes_)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_model,
        W=result["model"].coef_.astype(np.float32),
        b=result["model"].intercept_.astype(np.float32),
        mean=result["scaler"].mean_.astype(np.float32),
        scale=result["scaler"].scale_.astype(np.float32),
        classes=np.array(classes),
    )
    print(f"\nmodel   : {out_model}  ({out_model.stat().st_size / 1024:.1f} KB)")

    payload = {
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "encoder": MODEL_ID,
        "probe": "multinomial logistic regression on mean+std pooled hidden states",
        "split": {
            "train": list(TRAIN_ACTORS),
            "val": list(VAL_ACTORS),
            "test": list(TEST_ACTORS),
            "rule": "speaker-disjoint: no actor appears in more than one split",
            "refit": "final head fitted on train+val at the selected C",
        },
        "classes": classes,
        "selected_c": result["c"],
        "val_macro_f1": result["val_macro_f1"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "per_class": result["per_class"],
        "confusion": result["confusion"],
        "confusion_labels": result["labels_order"],
        "corpus_counts": dict(Counter(labels.tolist())),
        "note": (
            "Speaker-disjoint RAVDESS transfer. The head is fitted on actors 01-20 and "
            "measured once on 21-24; nothing was selected on the test speakers beyond the "
            "regularisation strength chosen on 19-20."
        ),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"metrics : {out_json}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--out", type=Path, default=OUT_MODEL)
    parser.add_argument("--json", type=Path, default=OUT_JSON)
    parser.add_argument("--refresh", action="store_true", help="ignore the cached embeddings")
    parser.add_argument("--quick", action="store_true", help="six actors only, for a smoke run")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="train on natural gain only, for the ablation table (loses gain invariance)",
    )
    args = parser.parse_args(argv)

    if not args.zip.exists():
        print(f"RAVDESS archive not found: {args.zip}")
        print("  python scripts/fetch_ravdess.py --download")
        return 1

    started = time.perf_counter()
    bundle = build_cache(
        args.zip, args.cache, quick=args.quick, refresh=args.refresh, augment_gain=not args.no_augment
    )
    result = fit(bundle["actors"], bundle["labels"], bundle["x"])
    save(result, args.out, args.json, bundle["actors"], bundle["labels"])
    print(f"elapsed : {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
