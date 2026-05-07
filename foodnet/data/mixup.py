"""Mixup/CutMix builder. Returns None when both alphas are 0 (disabled)."""
from __future__ import annotations
from timm.data import Mixup


def build_mixup_fn(
    num_classes: int,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    label_smoothing: float = 0.1,
    prob: float = 1.0,
    switch_prob: float = 0.5,
):
    """Return a timm Mixup callable, or None if both alphas are 0."""
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=prob,
        switch_prob=switch_prob,
        mode="batch",
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
