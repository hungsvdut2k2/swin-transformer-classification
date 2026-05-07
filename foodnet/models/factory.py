"""Swin-Tiny via timm with project defaults."""
from __future__ import annotations
import timm
import torch.nn as nn


def build_model(
    arch: str = "swin_tiny_patch4_window7_224",
    num_classes: int = 101,
    pretrained: bool = True,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """Build a timm model with a fresh classifier head sized to num_classes."""
    model = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model
