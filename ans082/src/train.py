"""Generic training loop for SmartSort AI.

Used for the ResNet-50 baseline (Phase 5) and the ViT experiments (Phase 6).
Only the train and validation splits are used here. The test split and the real photos
are never touched during training or tuning.

Run from the ans082 folder, for example:
    python src/train.py --config configs/resnet50.yaml
"""
import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from data import CLASSES, SmartSortDataset, build_loader, build_transforms
from model_loader import build_model, count_parameters
from utils import load_config, plot_learning_curves, set_seed

logger = logging.getLogger(__name__)

LOG_COLUMNS = [
    "epoch", "train_loss", "train_acc", "train_macro_f1",
    "val_loss", "val_acc", "val_macro_f1", "lr", "epoch_time_s",
]


def build_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Linear warm-up followed by cosine decay, stepped once per batch."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None,
              scheduler=None, scaler=None, desc=""):
    """Run one pass over a loader. Trains if an optimizer is given, otherwise evaluates.

    Returns (loss, accuracy, macro-F1).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, seen = 0.0, 0
    all_preds, all_labels = [], []
    with torch.enable_grad() if training else torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item() * images.size(0)
            seen += images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_labels).numpy()
    accuracy = float((preds == targets).mean())
    macro_f1 = float(f1_score(targets, preds, labels=list(range(len(CLASSES))),
                              average="macro", zero_division=0))
    return total_loss / seen, accuracy, macro_f1


def _atomic_save(obj, path: Path) -> None:
    """Write to a temporary file first, so a disconnect cannot leave a half-written checkpoint."""
    tmp_path = Path(str(path) + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _append_log(log_csv: Path, row: dict) -> None:
    """Append one epoch row to the training log, writing the header for a new file."""
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(
        log_csv, mode="a", header=not log_csv.exists(), index=False)


def train(cfg: dict, resume: bool = False, max_train=None, max_val=None) -> dict:
    """Train a model as described by cfg and return a summary dictionary."""
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg = cfg["train"]
    use_amp = bool(tcfg.get("fp16", True)) and device.type == "cuda"
    name = cfg["experiment"]

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    results_dir = Path(cfg["output"]["results_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    log_csv = results_dir / f"{name}_log.csv"

    model, mean, std = build_model(cfg["model"])
    total_params, trainable_params = count_parameters(model)
    model.to(device)
    logger.info("Device: %s | fp16: %s | parameters: %d total, %d trainable",
                device, use_amp, total_params, trainable_params)

    dcfg = cfg["data"]
    train_tf, eval_tf = build_transforms(mean, std, int(dcfg["image_size"]),
                                         dcfg.get("augmentation", "standard"))
    train_ds = SmartSortDataset(dcfg["split_csv"], dcfg["data_root"], "train",
                                train_tf, max_train, seed)
    val_ds = SmartSortDataset(dcfg["split_csv"], dcfg["data_root"], "val",
                              eval_tf, max_val, seed)
    batch_size, workers = int(dcfg["batch_size"]), int(dcfg.get("num_workers", 2))
    train_loader = build_loader(train_ds, batch_size, True, workers, seed)
    val_loader = build_loader(val_ds, batch_size, False, workers, seed)
    logger.info("Images: %d train, %d validation", len(train_ds), len(val_ds))

    weight = None
    if tcfg.get("class_weights", False):
        counts = np.bincount(train_ds.labels, minlength=len(CLASSES)).astype(float)
        weight = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)),
                              dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight,
                                    label_smoothing=float(tcfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.01)))
    epochs = int(tcfg["epochs"])
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * float(tcfg.get("warmup_ratio", 0.0)))
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    start_epoch, best_f1 = 1, -1.0
    if resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_f1 = state["epoch"] + 1, float(state["best_f1"])
        if log_csv.exists():  # drop log rows that were never checkpointed
            old_log = pd.read_csv(log_csv)
            old_log[old_log["epoch"] <= state["epoch"]].to_csv(log_csv, index=False)
        logger.info("Resumed after epoch %d (best validation macro-F1 so far: %.4f)",
                    state["epoch"], best_f1)
    elif log_csv.exists():
        log_csv.unlink()  # a fresh run starts a fresh log

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, device, use_amp, optimizer, scheduler,
            scaler, desc=f"train {epoch}/{epochs}")
        va_loss, va_acc, va_f1 = run_epoch(
            model, val_loader, criterion, device, use_amp, desc=f"val {epoch}/{epochs}")
        row = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "train_macro_f1": round(tr_f1, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
            "val_macro_f1": round(va_f1, 4),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(time.time() - started, 1),
        }
        _append_log(log_csv, row)

        is_best = va_f1 > best_f1
        if is_best:
            best_f1 = va_f1
        _atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_f1": float(best_f1),
        }, last_path)
        if is_best:
            _atomic_save({
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": CLASSES, "mean": mean, "std": std,
                "model_cfg": cfg["model"], "epoch": epoch,
                "val_macro_f1": float(va_f1), "val_acc": float(va_acc),
            }, best_path)
        logger.info(
            "epoch %d/%d | train loss %.4f acc %.4f f1 %.4f | val loss %.4f acc %.4f f1 %.4f%s | %.0fs",
            epoch, epochs, tr_loss, tr_acc, tr_f1, va_loss, va_acc, va_f1,
            "  (best)" if is_best else "", row["epoch_time_s"])

    log = pd.read_csv(log_csv)
    best_row = log.loc[log["val_macro_f1"].idxmax()]
    summary = {
        "experiment": name,
        "model": cfg["model"]["name"],
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "epochs_logged": int(len(log)),
        "best_epoch": int(best_row["epoch"]),
        "best_val_macro_f1": float(best_row["val_macro_f1"]),
        "best_val_acc": float(best_row["val_acc"]),
        "total_train_time_s": float(log["epoch_time_s"].sum()),
        "mean_epoch_time_s": float(log["epoch_time_s"].mean()),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "fp16": use_amp,
        "seed": seed,
        "torch_version": torch.__version__,
    }
    with open(results_dir / f"{name}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    plot_learning_curves(log_csv, results_dir / f"{name}_curves.png", title=name)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML experiment config")
    parser.add_argument("--data-root", help="override data.data_root")
    parser.add_argument("--checkpoint-dir", help="override output.checkpoint_dir")
    parser.add_argument("--results-dir", help="override output.results_dir")
    parser.add_argument("--epochs", type=int, help="override train.epochs")
    parser.add_argument("--max-train", type=int, help="use only N training images (smoke test)")
    parser.add_argument("--max-val", type=int, help="use only N validation images (smoke test)")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt if present")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.checkpoint_dir:
        cfg["output"]["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir:
        cfg["output"]["results_dir"] = args.results_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    summary = train(cfg, resume=args.resume, max_train=args.max_train, max_val=args.max_val)
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
