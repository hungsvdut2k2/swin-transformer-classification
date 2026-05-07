"""Worst-K classes by accuracy; hardest correct / hardest incorrect examples by loss."""
from __future__ import annotations
import numpy as np


def worst_k_classes(per_class_acc: np.ndarray, k: int, class_names: list[str]) -> list[dict]:
    order = np.argsort(per_class_acc)
    out = []
    for i in order[:k]:
        out.append({"class": class_names[int(i)], "accuracy": float(per_class_acc[int(i)]), "label": int(i)})
    return out


def hardest_examples(
    losses: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    k: int,
    kind: str,
) -> list[dict]:
    if kind not in ("correct", "incorrect"):
        raise ValueError(f"kind must be 'correct' or 'incorrect', got {kind!r}")
    correct_mask = preds == targets
    mask = correct_mask if kind == "correct" else ~correct_mask
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    sub = losses[idx]
    order = np.argsort(-sub)[:k]
    out = []
    for j in order:
        i = int(idx[int(j)])
        out.append({"index": i, "loss": float(losses[i]), "pred": int(preds[i]), "target": int(targets[i])})
    return out
