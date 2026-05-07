"""train_one_epoch + validate. AMP-aware, grad-clip-aware, mixup-aware."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.no_grad()
def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.size(1))
    _, pred = logits.topk(k, dim=1)
    return pred.eq(targets.unsqueeze(1)).any(dim=1).sum().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    grad_clip: float = 1.0,
    mixup_fn=None,
    amp: bool = True,
    progress: bool = False,
    desc: str | None = None,
) -> dict:
    model.train()
    loss_sum, n = 0.0, 0
    steps = 0
    iterator = loader
    pbar = None
    if progress:
        pbar = tqdm(loader, desc=desc or "train", leave=False, dynamic_ncols=True)
        iterator = pbar
    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
        steps += 1
        if pbar is not None:
            pbar.set_postfix(loss=f"{loss_sum / max(n, 1):.4f}", refresh=False)
    if pbar is not None:
        pbar.close()
    return {"loss": loss_sum / max(n, 1), "steps": steps}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    amp: bool = True,
    progress: bool = False,
    desc: str | None = None,
) -> dict:
    model.eval()
    loss_sum, n = 0.0, 0
    top1, top5 = 0, 0
    iterator = loader
    pbar = None
    if progress:
        pbar = tqdm(loader, desc=desc or "val", leave=False, dynamic_ncols=True)
        iterator = pbar
    for x, y in iterator:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
        top1 += _topk_correct(logits, y, 1)
        top5 += _topk_correct(logits, y, 5)
        if pbar is not None:
            pbar.set_postfix(top1=f"{top1 / max(n, 1):.4f}", refresh=False)
    if pbar is not None:
        pbar.close()
    return {
        "loss": loss_sum / max(n, 1),
        "top1": top1 / max(n, 1),
        "top5": top5 / max(n, 1),
        "n": n,
    }
