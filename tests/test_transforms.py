import torch
from PIL import Image
import numpy as np
from foodnet.data.transforms import build_train_transform, build_eval_transform


def _img() -> Image.Image:
    return Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"), "RGB")


def test_train_transform_outputs_3x224x224():
    t = build_train_transform(img_size=224)
    out = t(_img())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_eval_transform_outputs_3x224x224():
    t = build_eval_transform(img_size=224)
    out = t(_img())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
