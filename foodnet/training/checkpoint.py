"""Save/load training checkpoints (model + optimizer + scaler + metadata)."""
from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    best_metric: float,
    extras: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "extras": extras or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scaler=None,
    map_location: str = "cpu",
) -> dict:
    state = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state
