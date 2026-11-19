"""Confidence calibration, temperature scaling and the reject-option ("I'm not sure") analysis."""
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average |accuracy - confidence| over equal-width confidence bins, weighted by bin size."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> List[Dict]:
    """Per-bin confidence, accuracy and count, for plotting a reliability diagram."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()),
                    "confidence": float(conf[m].mean()) if m.any() else None,
                    "accuracy": float(correct[m].mean()) if m.any() else None})
    return out


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T > 0 minimising the negative log-likelihood of softmax(logits / T) on held-out data."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)          # optimise log T so that T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    p = softmax_np(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def selective_accuracy(probs: np.ndarray, labels: np.ndarray, thresholds=(0.5, 0.7, 0.9, 0.95)) -> List[Dict]:
    """For each confidence threshold: coverage (share of inputs answered) and accuracy on those answered.

    Rejected inputs ("not sure") are not counted as errors, so a good rejector raises accuracy while
    lowering coverage. This is the trade-off a safety-critical system has to choose.
    """
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append({"threshold": t, "coverage": float(keep.mean()),
                     "accuracy_on_answered": float(correct[keep].mean()) if keep.any() else None})
    return rows
