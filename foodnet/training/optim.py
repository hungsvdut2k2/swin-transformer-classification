"""AdamW + timm CosineLRScheduler + AMP GradScaler."""
from __future__ import annotations
import torch
from timm.scheduler import CosineLRScheduler


def build_optimizer(
    param_groups: list[dict],
    kind: str = "adamw",
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    if kind != "adamw":
        raise ValueError(f"Only adamw is supported; got {kind}")
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 2,
    min_lr: float = 1e-6,
    warmup_lr_init: float = 1e-7,
) -> CosineLRScheduler:
    return CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=min_lr,
        warmup_t=warmup_epochs,
        warmup_lr_init=warmup_lr_init,
        cycle_limit=1,
        t_in_epochs=True,
    )


def build_scaler(enabled: bool = True) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=enabled)
