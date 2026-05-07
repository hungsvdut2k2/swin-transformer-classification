"""ImageNet-stat transforms via timm.data.create_transform."""
from __future__ import annotations
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_train_transform(
    img_size: int = 224,
    auto_augment: str = "rand-m9-mstd0.5-inc1",
    re_prob: float = 0.25,
    color_jitter: float = 0.4,
):
    """Train transform: RandomResizedCrop + HFlip + RandAugment + RandomErasing + ImageNet norm."""
    return create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment=auto_augment,
        re_prob=re_prob,
        re_mode="pixel",
        re_count=1,
        color_jitter=color_jitter,
        interpolation="bicubic",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


def build_eval_transform(img_size: int = 224, crop_pct: float = 224 / 256):
    """Eval transform: Resize(round(img_size/crop_pct)) -> CenterCrop(img_size) -> Normalize."""
    return create_transform(
        input_size=img_size,
        is_training=False,
        interpolation="bicubic",
        crop_pct=crop_pct,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
