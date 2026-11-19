"""Small shared helpers: random seeds, config loading and learning-curve plots."""
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch so runs can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path) -> dict:
    """Read a YAML experiment config into a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def plot_learning_curves(log_csv, out_png, title: str = "") -> None:
    """Plot loss, accuracy and macro-F1 per epoch (train vs validation) from a training log."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = pd.read_csv(log_csv)
    panels = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_acc", "val_acc"),
        ("Macro-F1", "train_macro_f1", "val_macro_f1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, train_col, val_col) in zip(axes, panels):
        ax.plot(log["epoch"], log[train_col], marker="o", label="train")
        ax.plot(log["epoch"], log[val_col], marker="o", label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("Saved learning curves to %s", out_png)
