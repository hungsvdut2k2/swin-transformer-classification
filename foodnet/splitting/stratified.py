"""Per-class shuffle-and-slice stratified split."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test) by per-class shuffle-and-slice.

    Each class is shuffled independently with a class-derived seed, then sliced
    by the given ratios. Counts use floor for train and val; test gets the
    remainder so all rows are accounted for.

    Args:
        df: Frame with columns including ``label``.
        ratios: (train, val, test). Must sum to 1.0 within 1e-6.
        seed: Master seed; per-class seed is ``seed + label``.

    Returns:
        (train_df, val_df, test_df), reset_index applied.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)} from {ratios}")
    r_train, r_val, _ = ratios
    train_parts, val_parts, test_parts = [], [], []
    for label, group in df.groupby("label", sort=True):
        items = group.sample(frac=1.0, random_state=seed + int(label)).reset_index(drop=True)
        n = len(items)
        n_train = int(math.floor(n * r_train))
        n_val = int(math.floor(n * r_val))
        train_parts.append(items.iloc[:n_train])
        val_parts.append(items.iloc[n_train : n_train + n_val])
        test_parts.append(items.iloc[n_train + n_val :])
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, val, test
