"""Training loop and inference helpers (kept separate from the CLI so they can be unit-tested)."""
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.classes import CLASS_NAMES
from src.data import SignDataset
from src.models import build_model

DEFAULT_LR = {"smallcnn": 2e-3, "resnet18": 1e-3, "convnext_tiny": 5e-4}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_logits(model: nn.Module, dataset: SignDataset, device: torch.device,
                   batch_size: int = 256, num_workers: int = 2) -> np.ndarray:
    """Run the model over a whole dataset and return raw logits as an (N, C) numpy array."""
    model.eval().to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    for x, _ in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out.append(model(x.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def train_model(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                model_name: str, out_dir: str, epochs: int = 8, batch_size: int = 128,
                lr: Optional[float] = None, weather_p: float = 0.0, pretrained: bool = True,
                num_workers: int = 2, device: Optional[torch.device] = None) -> Dict:
    """Train one model and save `<out_dir>/model.pt` (best validation accuracy) and `history.json`.

    weather_p = 0   -> "clean" training (the baseline)
    weather_p > 0   -> weather-augmented training (the robust variant)
    """
    device = device or get_device()
    lr = lr or DEFAULT_LR[model_name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = SignDataset(train_x, train_y, train=True, weather_p=weather_p)
    val_ds = SignDataset(val_x, val_y, train=False)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                        drop_last=True, pin_memory=device.type == "cuda")

    model = build_model(model_name, pretrained=pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()   # no label smoothing: we study calibration separately

    history, best_acc, start = [], -1.0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        val_acc = float((predict_logits(model, val_ds, device, num_workers=num_workers).argmax(1) == val_y).mean())
        row = {"epoch": epoch, "train_loss": total_loss / seen, "train_acc": correct / seen,
               "val_acc": val_acc, "seconds": round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_name": model_name, "state_dict": model.state_dict(), "weather_p": weather_p,
                        "class_names": CLASS_NAMES, "temperature": 1.0, "val_acc": val_acc}, out / "model.pt")

    summary = {"model": model_name, "params": n_params, "weather_p": weather_p, "epochs": epochs,
               "lr": lr, "best_val_acc": best_acc, "history": history}
    (out / "history.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_checkpoint(path: str, device: Optional[torch.device] = None):
    """Load a saved checkpoint. Returns (model in eval mode, checkpoint dict)."""
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
