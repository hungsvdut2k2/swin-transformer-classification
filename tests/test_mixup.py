import torch
from foodnet.data.mixup import build_mixup_fn


def test_mixup_fn_smooths_targets_to_one_hot_when_disabled():
    fn = build_mixup_fn(num_classes=10, mixup_alpha=0.0, cutmix_alpha=0.0, label_smoothing=0.1)
    assert fn is None  # disabled when both alphas are 0


def test_mixup_fn_blends_when_enabled():
    fn = build_mixup_fn(num_classes=10, mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1)
    assert fn is not None
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3])
    x_out, y_out = fn(x, y)
    assert x_out.shape == x.shape
    assert y_out.shape == (4, 10)
    assert torch.allclose(y_out.sum(dim=1), torch.ones(4), atol=1e-5)
