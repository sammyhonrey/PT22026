"""Dataset loading for the Kaggle GTSRB layout, leakage-free splitting and PyTorch datasets.

Expected folder (as in the Kaggle 'GTSRB - German Traffic Sign Recognition Benchmark' download):
    <root>/Train/<class_id>/*.png      (and optionally Train.csv with columns Path, ClassId)
    <root>/Test/*.png                  with Test.csv (columns Path, ClassId)
"""
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import IMG_SIZE, MEAN, SEED, STD
from src.corruptions import corrupt, random_weather

_IMG_EXT = {".png", ".ppm", ".jpg", ".jpeg"}
_TRACK_RE = re.compile(r"^(\d+)_(\d+)_(\d+)")  # class_track_frame, e.g. 00000_00012_00003.png


def find_root(path: str) -> Path:
    """Return the folder that actually contains Train/ and Test/ (handles one extra nesting level)."""
    root = Path(path)
    for cand in [root, *[p for p in root.glob("*") if p.is_dir()]]:
        if (cand / "Train").is_dir():
            return cand
    raise FileNotFoundError(f"Could not find a 'Train' folder under {path}. "
                            "Check the unzip location (see README).")


def load_index(root: Path, split: str) -> pd.DataFrame:
    """Return a DataFrame with columns: path, label, group for 'train' or 'test'."""
    split = split.lower()
    csv = root / f"{split.capitalize()}.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df = pd.DataFrame({"path": [str(root / p) for p in df["Path"]], "label": df["ClassId"].astype(int)})
    elif split == "train":
        rows = [(str(p), int(p.parent.name)) for p in sorted((root / "Train").rglob("*"))
                if p.suffix.lower() in _IMG_EXT and p.parent.name.isdigit()]
        df = pd.DataFrame(rows, columns=["path", "label"])
    else:
        raise FileNotFoundError(f"{csv} not found; Test.csv is required for the test split.")
    if df.empty:
        raise ValueError(f"No images found for the {split} split under {root}")

    def group_of(p: str, label: int) -> str:
        m = _TRACK_RE.match(Path(p).stem)
        return f"{m.group(1)}_{m.group(2)}" if m else f"{label}_{Path(p).stem}"

    df["group"] = [group_of(p, y) for p, y in zip(df["path"], df["label"])]
    return df


def group_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by physical sign (track), not by image, so near-identical frames never straddle the split.

    GTSRB has ~30 consecutive frames of each real sign. A random image-level split would put sibling
    frames in both train and validation and inflate validation accuracy.
    """
    rng = np.random.default_rng(seed)
    val_groups = set()
    for _, sub in df.groupby("label"):
        groups = sorted(sub["group"].unique())
        rng.shuffle(groups)
        n_val = max(1, math.ceil(len(groups) * val_frac)) if len(groups) > 1 else 0
        val_groups.update(groups[:n_val])
    is_val = df["group"].isin(val_groups)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def load_arrays(df: pd.DataFrame, size: int = IMG_SIZE, cache: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load every image resized to size x size into one uint8 array (fast training). Optional .npz cache."""
    if cache and Path(cache).exists():
        z = np.load(cache)
        if len(z["y"]) == len(df):
            return z["x"], z["y"]
    xs = np.empty((len(df), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(df["path"]):
        with Image.open(p) as im:
            xs[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
    ys = df["label"].to_numpy(dtype=np.int64)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, x=xs, y=ys)
    return xs, ys


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HWC image -> normalised float CHW tensor."""
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0   # np.array copies (read-only safe)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (x - mean) / std


def _geometric_jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small rotation and shift. NO horizontal flip: 'turn left' and 'turn right' would swap meaning."""
    im = Image.fromarray(img)
    im = im.rotate(float(rng.uniform(-10, 10)), resample=Image.BILINEAR)
    dx, dy = float(rng.uniform(-4, 4)), float(rng.uniform(-4, 4))
    im = im.transform(im.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), resample=Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


class SignDataset(Dataset):
    """In-memory sign dataset.

    train=True            -> geometric jitter every sample
    weather_p > 0         -> weather augmentation (conditions in config.AUG_CONDITIONS, severity <= 4)
    fixed_condition=(c,s) -> deterministically apply condition c at severity s (for evaluation)
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False,
                 weather_p: float = 0.0, fixed_condition: Optional[Tuple[str, int]] = None, seed: int = SEED):
        self.images, self.labels = images, labels
        self.train, self.weather_p, self.fixed, self.seed = train, weather_p, fixed_condition, seed

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = self.images[i]
        # Per-sample generator: reproducible for evaluation, and different per index for training.
        rng = np.random.default_rng([self.seed, i, 0 if not self.train else int(torch.randint(0, 2**31 - 1, (1,)))])
        if self.train:
            img = _geometric_jitter(img, rng)
        if self.weather_p > 0:
            img = random_weather(img, rng, self.weather_p)
        if self.fixed is not None:
            img = corrupt(img, self.fixed[0], self.fixed[1], rng)
        return to_tensor(img), int(self.labels[i])
