"""Reading and writing landmark tables.

Each row is one pose: label, subject, split, source, 33 x (x, y, z) world
coordinates and 33 visibility scores.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import landmarks as lm

META = ["label", "subject", "split", "source"]
COORD_COLUMNS = [f"{name}_{axis}" for name in lm.NAMES for axis in "xyz"]
VIS_COLUMNS = [f"{name}_v" for name in lm.NAMES]
COLUMNS = META + COORD_COLUMNS + VIS_COLUMNS


@dataclass
class PoseTable:
    points: np.ndarray
    visibility: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    splits: np.ndarray
    sources: np.ndarray

    def __len__(self):
        return len(self.labels)

    def select(self, mask):
        return PoseTable(*(getattr(self, f)[mask] for f in self.__dataclass_fields__))

    @property
    def classes(self):
        return sorted(set(self.labels.tolist()))


def empty():
    return PoseTable(
        np.zeros((0, lm.COUNT, 3), np.float32), np.zeros((0, lm.COUNT), np.float32),
        *(np.array([], dtype=str) for _ in range(4)),
    )


def concat(tables):
    tables = [t for t in tables if len(t)]
    if not tables:
        return empty()
    return PoseTable(*(np.concatenate([getattr(t, f) for t in tables]) for f in PoseTable.__dataclass_fields__))


def make_row(label, subject, split, source, world, visibility):
    coords = [f"{v:.5f}" for v in np.asarray(world).reshape(-1)]
    vis = [f"{v:.3f}" for v in np.asarray(visibility).reshape(-1)]
    return [label, subject, split, source] + coords + vis


def write_rows(path, rows, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not (append and path.exists() and path.stat().st_size > 0)
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(COLUMNS)
        writer.writerows(rows)


def load(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != COLUMNS:
            raise ValueError(f"{path.name} does not have the expected landmark columns")
        rows = [r for r in reader if r]
    if not rows:
        return empty()
    meta = np.array([r[:4] for r in rows], dtype=str)
    values = np.array([r[4:] for r in rows], dtype=np.float32)
    n_coords = lm.COUNT * 3
    return PoseTable(
        points=values[:, :n_coords].reshape(-1, lm.COUNT, 3),
        visibility=values[:, n_coords:],
        labels=meta[:, 0], subjects=meta[:, 1], splits=meta[:, 2], sources=meta[:, 3],
    )
