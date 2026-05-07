import torch
from foodnet.models.factory import build_model


def test_build_model_swin_tiny_no_pretrained_3_classes():
    model = build_model("swin_tiny_patch4_window7_224", num_classes=3, pretrained=False, drop_path_rate=0.2)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 3)


def test_build_model_classifier_replaced():
    # The final head should match num_classes.
    model = build_model("swin_tiny_patch4_window7_224", num_classes=7, pretrained=False)
    n_out = sum(1 for p in model.get_classifier().parameters() if p.requires_grad and p.dim() == 2 and p.shape[0] == 7)
    assert n_out >= 1
