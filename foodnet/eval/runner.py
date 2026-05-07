"""Single-pass inference that writes per-example softmax + per-example loss to parquet."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    num_classes: int,
    device: str,
    filepaths: list[str],
    out_path: Path,
    amp: bool = True,
) -> None:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="none")
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
        all_losses.append(loss.float().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    losses = np.concatenate(all_losses, axis=0)

    if len(filepaths) != len(targets):
        raise RuntimeError(f"filepaths ({len(filepaths)}) does not match inference rows ({len(targets)})")

    df = pd.DataFrame({"filepath": filepaths, "label": targets, "loss": losses})
    for c in range(num_classes):
        df[f"p_{c}"] = probs[:, c]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
