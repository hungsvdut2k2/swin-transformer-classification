"""Early stopping on a scalar monitor (val/top1 by default)."""
from __future__ import annotations
import math


class EarlyStopper:
    """Stop after `patience` epochs with no improvement greater than min_delta.

    mode='max' for metrics like accuracy; mode='min' for losses.
    `step(value)` returns True when training should stop.
    """

    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 1e-3) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self._best = -math.inf if mode == "max" else math.inf
        self._stale = 0
        self.should_stop = False

    def _better(self, current: float) -> bool:
        if self.mode == "max":
            return current > self._best + self.min_delta
        return current < self._best - self.min_delta

    def step(self, current: float) -> bool:
        if self._better(current):
            self._best = current
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                self.should_stop = True

        return self.should_stop

    @property
    def best(self) -> float:
        return self._best
