"""Tiny W&B wrapper that no-ops when disabled (so tests + offline runs work)."""
from __future__ import annotations
from typing import Any


class WandBLogger:
    def __init__(self, enabled: bool, project: str | None = None, run_name: str | None = None, config: dict | None = None) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        import wandb  # imported lazily so tests don't require wandb auth
        self._run = wandb.init(project=project, name=run_name, config=config or {}, reinit=True)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.log(payload, step=step)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.finish()
