"""Fine-tune ResNet-50 or ViT-base on the chest X-ray dataset."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from model import build_model
from utils import make_dataloaders


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    """Run one epoch. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, preds_all, labels_all = 0.0, [], []
    with torch.set_grad_enabled(training):
        for i, (images, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds_all.extend(logits.argmax(1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    n = len(labels_all)
    acc = sum(p == t for p, t in zip(preds_all, labels_all)) / n
    f1 = f1_score(labels_all, preds_all, average="macro")
    return total_loss / n, acc, f1


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model on chest X-rays")
    parser.add_argument("--model", choices=["resnet50", "vit"], default="resnet50")
    parser.add_argument("--data-dir", default="data/chest_xray")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per epoch (quick smoke test only)")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, _, classes, class_weights = make_dataloaders(
        args.data_dir, args.model, args.batch_size, seed=args.seed,
        num_workers=args.num_workers)
    print(f"Classes: {classes} | class weights: {class_weights.tolist()}")

    model = build_model(args.model, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir, results_dir = Path(args.out_dir), Path(args.results_dir)
    out_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    history, best_f1 = [], -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer, args.max_batches)
        va_loss, va_acc, va_f1 = run_epoch(
            model, val_loader, criterion, device, None, args.max_batches)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "train_f1": tr_f1, "val_loss": va_loss, "val_acc": va_acc,
                        "val_f1": va_f1})
        print(f"Epoch {epoch}/{args.epochs} ({time.time() - start:.0f}s) | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}")

        if va_f1 > best_f1:  # keep the best checkpoint by validation macro-F1
            best_f1 = va_f1
            torch.save(model.state_dict(), out_dir / f"{args.model}_best.pt")

    with open(results_dir / f"{args.model}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val macro-F1: {best_f1:.3f}")


if __name__ == "__main__":
    main()