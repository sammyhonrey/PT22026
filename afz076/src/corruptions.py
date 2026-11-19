"""Weather and image-quality simulations (fog, night, rain, blur, noise, glare).

All functions take and return a uint8 RGB numpy array of shape (H, W, 3). Severity runs from
1 (mild) to 5 (extreme). Passing a numpy Generator makes every result reproducible.
"""
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import AUG_CONDITIONS, AUG_MAX_SEVERITY, CONDITIONS, SEVERITIES

_FOG_ALPHA = (0.20, 0.35, 0.50, 0.65, 0.80)
_NIGHT_BRIGHTNESS = (0.70, 0.55, 0.40, 0.28, 0.18)
_NIGHT_NOISE = (4, 8, 12, 16, 20)
_RAIN_STREAKS = (10, 20, 35, 50, 70)
_BLUR_RADIUS = (0.5, 0.8, 1.2, 1.6, 2.2)
_NOISE_SIGMA = (8, 16, 25, 35, 50)
_GLARE_RADIUS = (0.25, 0.35, 0.45, 0.55, 0.70)
_GLARE_STRENGTH = (0.40, 0.55, 0.70, 0.85, 1.00)


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _check(severity: int) -> int:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}, got {severity}")
    return severity - 1


def fog(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Blend the image with a smooth, bright haze layer."""
    i = _check(severity)
    h, w, _ = img.shape
    low = rng.random((4, 4)).astype(np.float32)                       # coarse random haze
    haze = np.asarray(Image.fromarray((low * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                      dtype=np.float32) / 255.0
    haze = 200 + 40 * haze[..., None]                                 # light grey, 200-240
    a = _FOG_ALPHA[i]
    return _clip((1 - a) * img.astype(np.float32) + a * haze)


def night(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Darken the image, flatten contrast and add sensor noise."""
    i = _check(severity)
    x = img.astype(np.float32) / 255.0
    x = (x ** 1.4) * _NIGHT_BRIGHTNESS[i]
    x = x * 255 + rng.normal(0, _NIGHT_NOISE[i], x.shape)
    return _clip(x)


def rain(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Draw semi-transparent diagonal streaks and slightly dim the scene."""
    i = _check(severity)
    h, w, _ = img.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(_RAIN_STREAKS[i]):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        length = int(rng.integers(max(3, h // 8), max(5, h // 4)))
        draw.line([(x0, y0), (x0 - length // 3, y0 + length)], fill=(220, 220, 230, 140), width=1)
    base = Image.fromarray(_clip(img.astype(np.float32) * (1 - 0.04 * (i + 1)))).convert("RGBA")
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"), dtype=np.uint8)


def blur(img: np.ndarray, severity: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Gaussian blur (out-of-focus or motion-smeared camera)."""
    i = _check(severity)
    out = Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=_BLUR_RADIUS[i]))
    return np.asarray(out, dtype=np.uint8)


def noise(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian noise (low-quality sensor)."""
    i = _check(severity)
    return _clip(img.astype(np.float32) + rng.normal(0, _NOISE_SIGMA[i], img.shape))


def glare(img: np.ndarray, severity: int, rng: np.random.Generator) -> np.ndarray:
    """Add a bright radial light blob (sun or headlight glare) at a random position."""
    i = _check(severity)
    h, w, _ = img.shape
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sigma = _GLARE_RADIUS[i] * min(h, w)
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))[..., None]
    return _clip(img.astype(np.float32) + 255 * _GLARE_STRENGTH[i] * blob)


_FUNCS = {"fog": fog, "night": night, "rain": rain, "blur": blur, "noise": noise, "glare": glare}


def corrupt(img: np.ndarray, condition: str, severity: int,
            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply one named condition. `rng` defaults to a fixed seed so results are reproducible."""
    if condition not in _FUNCS:
        raise ValueError(f"unknown condition '{condition}', choose from {CONDITIONS}")
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img must be a uint8 array of shape (H, W, 3)")
    rng = rng if rng is not None else np.random.default_rng(0)
    return _FUNCS[condition](img, severity, rng)


def random_weather(img: np.ndarray, rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Training augmentation: with probability p apply a random TRAINING condition (never glare)
    at a random severity from 1 to AUG_MAX_SEVERITY."""
    if rng.random() >= p:
        return img
    cond = AUG_CONDITIONS[int(rng.integers(len(AUG_CONDITIONS)))]
    sev = int(rng.integers(1, AUG_MAX_SEVERITY + 1))
    return corrupt(img, cond, sev, rng)
