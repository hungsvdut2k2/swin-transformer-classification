import torch
from foodnet.models.factory import build_model
from foodnet.models.llrd import llrd_param_groups
from foodnet.training.optim import build_optimizer, build_scheduler, build_scaler


def test_build_optimizer_adamw():
    model = build_model("swin_tiny_patch4_window7_224", num_classes=10, pretrained=False)
    groups = llrd_param_groups(model, base_lr=1e-4, weight_decay=0.05)
    opt = build_optimizer(groups, kind="adamw", betas=(0.9, 0.999))
    assert isinstance(opt, torch.optim.AdamW)


def test_build_scheduler_cosine_with_warmup():
    model = build_model("swin_tiny_patch4_window7_224", num_classes=10, pretrained=False)
    groups = llrd_param_groups(model, base_lr=1e-4, weight_decay=0.05)
    opt = build_optimizer(groups, kind="adamw")
    sched = build_scheduler(opt, num_epochs=10, warmup_epochs=2, min_lr=1e-6)
    # timm CosineLRScheduler exposes step(epoch).
    assert hasattr(sched, "step")


def test_build_scaler_returns_gradscaler():
    s = build_scaler(enabled=True)
    assert isinstance(s, torch.cuda.amp.GradScaler)
    if torch.cuda.is_available():
        assert s.is_enabled() is True

    s2 = build_scaler(enabled=False)
    assert s2.is_enabled() is False
