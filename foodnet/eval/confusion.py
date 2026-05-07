"""Confusion matrix + most-confused (true, pred) pairs."""
from __future__ import annotations
import numpy as np


def confusion_matrix(softmax: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    preds = softmax.argmax(axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def most_confused_pairs(
    cm: np.ndarray,
    k: int,
    class_names: list[str] | None = None,
) -> list[dict]:
    """Return the top-k off-diagonal cells sorted by count desc."""
    n = cm.shape[0]
    cells = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                cells.append({"true": class_names[i] if class_names else i, "pred": class_names[j] if class_names else j, "count": int(cm[i, j])})
    cells.sort(key=lambda d: d["count"], reverse=True)
    return cells[:k]
