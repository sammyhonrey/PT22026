"""Shared fixtures: a tiny synthetic dataset in the same folder layout as the Kaggle GTSRB download."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_CLASSES, TRACKS, FRAMES, N_TEST = 43, 3, 5, 86


def _sign_image(c: int, rng: np.random.Generator, size: int = 40) -> Image.Image:
    """A coloured disc or square whose colours depend on the class, plus a little noise."""
    bg = ((c * 37) % 200 + 30, (c * 91) % 200 + 30, (c * 53) % 200 + 30)
    fg = (255 - bg[0], 255 - bg[1], 255 - bg[2])
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    pad = int(rng.integers(6, 10))
    box = [pad, pad, size - pad, size - pad]
    d.ellipse(box, fill=fg) if c % 2 == 0 else d.rectangle(box, fill=fg)
    arr = np.asarray(im, dtype=np.float32) + rng.normal(0, 6, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("gtsrb") / "archive"       # one extra nesting level, like a real unzip
    rng = np.random.default_rng(0)
    rows = []
    for c in range(N_CLASSES):
        (root / "Train" / str(c)).mkdir(parents=True)
        for t in range(TRACKS):
            for f in range(FRAMES):
                rel = f"Train/{c}/{c:05d}_{t:05d}_{f:05d}.png"
                _sign_image(c, rng).save(root / rel)
                rows.append({"Path": rel, "ClassId": c})
    (root / "Test").mkdir()
    test_rows = []
    for i in range(N_TEST):
        c = i % N_CLASSES
        rel = f"Test/{i:05d}.png"
        _sign_image(c, rng).save(root / rel)
        test_rows.append({"Path": rel, "ClassId": c})
    pd.DataFrame(test_rows).to_csv(root / "Test.csv", index=False)
    return root.parent                                          # pass the PARENT to test find_root()
