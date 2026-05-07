"""Layer-wise LR decay param groups for Swin (5 layer buckets)."""
from __future__ import annotations
from typing import Iterable
import torch.nn as nn

NO_DECAY_KEYWORDS = ("bias", "norm", "relative_position_bias_table", "absolute_pos_embed")


def _layer_id(name: str, num_layers: int = 4) -> int:
    """Map parameter name to a layer bucket id in [0, num_layers+1].

    0 = patch_embed (earliest)
    1..num_layers = layers.0 .. layers.{num_layers-1}
    num_layers+1 = head (latest)
    """
    if name.startswith("patch_embed") or name.startswith("absolute_pos_embed"):
        return 0
    if name.startswith("layers."):
        block_id = int(name.split(".")[1])
        return block_id + 1
    if name.startswith("norm.") or name.startswith("head"):
        return num_layers + 1
    return num_layers + 1  # default to head bucket


def _should_no_decay(name: str, param_shape: tuple[int, ...]) -> bool:
    if param_shape == () or len(param_shape) == 1:
        return True
    return any(k in name for k in NO_DECAY_KEYWORDS)


def llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float = 0.75,
    num_layers: int = 4,
) -> list[dict]:
    """Group parameters by (layer-id, decay/no-decay) and assign per-bucket lr.

    lr_scale[layer_id] = layer_decay ** ((num_layers + 1) - layer_id)
    so the head (layer_id = num_layers+1) has scale 1.0 and patch_embed the smallest.
    """
    layer_names = ["patch_embed"] + [f"layers.{i}" for i in range(num_layers)] + ["head"]
    n_buckets = num_layers + 2
    groups: dict[tuple[int, str], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _layer_id(name, num_layers=num_layers)
        nd = _should_no_decay(name, tuple(p.shape))
        key = (lid, "no_decay" if nd else "decay")
        if key not in groups:
            scale = layer_decay ** (n_buckets - 1 - lid)
            groups[key] = {
                "name": f"{layer_names[lid]}_{'no_decay' if nd else 'decay'}",
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if nd else weight_decay,
                "lr_scale": scale,
            }
        groups[key]["params"].append(p)
    return [g for g in groups.values() if g["params"]]
