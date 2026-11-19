"""Evaluate trained checkpoints: clean accuracy, robustness to weather, calibration and rejection.

Example:
  python evaluation/evaluate.py --data-root /content/gtsrb \
      --ckpt outputs/resnet18_clean/model.pt outputs/resnet18_weather/model.pt

Writes evaluation/results/<name>.json and evaluation/results/robustness.csv
Also fits a temperature on the VALIDATION set and stores it inside each checkpoint (used by the app).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import (expected_calibration_error, fit_temperature, nll,  # noqa: E402
                             reliability_bins, selective_accuracy, softmax_np)
from src.config import CONDITIONS  # noqa: E402
from src.data import SignDataset, find_root, group_split, load_arrays, load_index  # noqa: E402
from src.trainer import get_device, load_checkpoint, predict_logits  # noqa: E402


def metrics(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict:
    p_raw, p_t = softmax_np(logits), softmax_np(logits, temperature)
    return {"accuracy": float((logits.argmax(1) == labels).mean()),
            "ece_raw": expected_calibration_error(p_raw, labels),
            "ece_temp": expected_calibration_error(p_t, labels),
            "mean_confidence_temp": float(p_t.max(1).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--severities", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test images (smoke tests)")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--out-dir", default="evaluation/results")
    args = ap.parse_args()

    device = get_device()
    root = find_root(args.data_root)
    _, val_df = group_split(load_index(root, "train"))
    test_df = load_index(root, "test")
    if args.limit:
        test_df = test_df.iloc[: args.limit]
    tag = "" if args.limit else "_full"
    val_x, val_y = load_arrays(val_df, cache=f"{args.cache_dir}/val{tag}.npz")
    test_x, test_y = load_arrays(test_df, cache=f"{args.cache_dir}/test{tag}.npz")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in args.ckpt:
        name = Path(path).parent.name
        model, ckpt = load_checkpoint(path, device)
        print(f"\n=== {name} ===")

        # 1) Temperature scaling, fitted on clean VALIDATION data only (never on the test set).
        val_logits = predict_logits(model, SignDataset(val_x, val_y), device)
        temperature = fit_temperature(val_logits, val_y)
        ckpt["temperature"] = temperature
        torch.save(ckpt, path)

        # 2) Clean test set.
        clean_logits = predict_logits(model, SignDataset(test_x, test_y), device)
        result = {"name": name, "model": ckpt["model_name"], "weather_p": ckpt["weather_p"],
                  "temperature": temperature, "val_nll_raw": nll(val_logits, val_y),
                  "val_nll_temp": nll(val_logits, val_y, temperature),
                  "clean": metrics(clean_logits, test_y, temperature),
                  "reliability_clean_raw": reliability_bins(softmax_np(clean_logits), test_y),
                  "reliability_clean_temp": reliability_bins(softmax_np(clean_logits, temperature), test_y),
                  "selective_clean": selective_accuracy(softmax_np(clean_logits, temperature), test_y),
                  "conditions": []}
        print(f"temperature={temperature:.3f}  clean test accuracy={result['clean']['accuracy']:.4f}")
        rows.append({"run": name, "condition": "clean", "severity": 0, **result["clean"]})

        # 3) Every weather condition at the chosen severities.
        pooled_probs, pooled_labels = [], []
        for cond in CONDITIONS:
            for sev in args.severities:
                logits = predict_logits(model, SignDataset(test_x, test_y, fixed_condition=(cond, sev)), device)
                m = metrics(logits, test_y, temperature)
                result["conditions"].append({"condition": cond, "severity": sev, **m})
                rows.append({"run": name, "condition": cond, "severity": sev, **m})
                print(f"  {cond:6s} sev {sev}: acc={m['accuracy']:.4f}  ece_raw={m['ece_raw']:.3f}  ece_temp={m['ece_temp']:.3f}")
                if sev == 3:
                    pooled_probs.append(softmax_np(logits, temperature))
                    pooled_labels.append(test_y)
        if pooled_probs:
            result["selective_sev3_pooled"] = selective_accuracy(np.concatenate(pooled_probs),
                                                                 np.concatenate(pooled_labels))
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))

    csv_path = out_dir / "robustness.csv"
    new = pd.DataFrame(rows)
    if csv_path.exists():                                   # keep earlier runs, replace re-evaluated ones
        old = pd.read_csv(csv_path)
        new = pd.concat([old[~old["run"].isin(new["run"])], new])
    new.to_csv(csv_path, index=False)
    print(f"\nSaved results to {out_dir}/")


if __name__ == "__main__":
    main()
