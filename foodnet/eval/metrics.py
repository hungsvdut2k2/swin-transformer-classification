"""Top-k accuracy, macro-F1, per-class accuracy from softmax + targets."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score


def topk_accuracy(softmax: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Fraction of rows whose true class is within the top-k predictions."""
    if k > softmax.shape[1]:
        k = softmax.shape[1]
    topk = np.argpartition(-softmax, kth=k - 1, axis=1)[:, :k]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(hits.mean())


def macro_f1(softmax: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    preds = softmax.argmax(axis=1)
    return float(f1_score(targets, preds, labels=list(range(num_classes)), average="macro", zero_division=0))


def per_class_accuracy(softmax: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    preds = softmax.argmax(axis=1)
    accs = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = float((preds[mask] == c).mean())
    return accs
