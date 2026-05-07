import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from foodnet.training.loop import train_one_epoch, validate


def _toy_setup():
    torch.manual_seed(0)
    x = torch.randn(32, 3, 16, 16)
    y = torch.randint(0, 4, (32,))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=8)
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 4),
    )
    return dl, model


def test_train_one_epoch_returns_loss_and_steps():
    dl, model = _toy_setup()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    metrics = train_one_epoch(model, dl, opt, crit, scaler, device="cpu", grad_clip=1.0, mixup_fn=None, amp=False)
    assert "loss" in metrics
    assert metrics["steps"] == 4
    assert metrics["loss"] > 0


def test_validate_returns_top1_top5_loss():
    dl, model = _toy_setup()
    crit = nn.CrossEntropyLoss()
    metrics = validate(model, dl, crit, device="cpu", num_classes=4)
    for key in ("loss", "top1", "top5"):
        assert key in metrics
    assert 0.0 <= metrics["top1"] <= 1.0
