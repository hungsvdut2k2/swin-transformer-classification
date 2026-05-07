"""Expected Calibration Error + reliability bin data for a diagram."""
from __future__ import annotations
import numpy as np


def reliability_bins(softmax: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> dict:
    """Bin examples by their max-softmax confidence; report avg conf and acc per bin."""
    confs = softmax.max(axis=1)
    preds = softmax.argmax(axis=1)
    correct = (preds == targets).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        bin_count[b] = int(mask.sum())
        if bin_count[b] > 0:
            bin_acc[b] = float(correct[mask].mean())
            bin_conf[b] = float(confs[mask].mean())
    return {
        "bin_lower": edges[:-1].tolist(),
        "bin_upper": edges[1:].tolist(),
        "bin_acc": bin_acc.tolist(),
        "bin_conf": bin_conf.tolist(),
        "bin_count": bin_count.tolist(),
    }


def expected_calibration_error(softmax: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|."""
    bins = reliability_bins(softmax, targets, n_bins=n_bins)
    counts = np.array(bins["bin_count"], dtype=np.float64)
    accs = np.array(bins["bin_acc"])
    confs = np.array(bins["bin_conf"])
    n = counts.sum()
    if n == 0:
        return 0.0
    return float((counts / n * np.abs(accs - confs)).sum())
