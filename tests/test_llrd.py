import torch
from foodnet.models.factory import build_model
from foodnet.models.llrd import llrd_param_groups


def test_llrd_param_groups_cover_all_params():
    model = build_model("swin_tiny_patch4_window7_224", num_classes=10, pretrained=False)
    groups = llrd_param_groups(model, base_lr=1e-4, weight_decay=0.05, layer_decay=0.75)
    n_in_groups = sum(len(g["params"]) for g in groups)
    n_in_model = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_in_groups == n_in_model


def test_llrd_lr_scales_decrease_for_earlier_layers():
    model = build_model("swin_tiny_patch4_window7_224", num_classes=10, pretrained=False)
    groups = llrd_param_groups(model, base_lr=1e-4, weight_decay=0.05, layer_decay=0.75)
    # head should have the highest lr; patch_embed the lowest.
    by_name = {g["name"]: g["lr"] for g in groups}
    assert by_name["head_decay"] > by_name["patch_embed_decay"]
    assert by_name["head_decay"] >= by_name["layers.3_decay"] >= by_name["layers.0_decay"] >= by_name["patch_embed_decay"]


def test_llrd_no_decay_on_norms_and_biases():
    model = build_model("swin_tiny_patch4_window7_224", num_classes=10, pretrained=False)
    groups = llrd_param_groups(model, base_lr=1e-4, weight_decay=0.05, layer_decay=0.75)
    no_decay = [g for g in groups if g["weight_decay"] == 0.0]
    assert len(no_decay) > 0
    has_decay = [g for g in groups if g["weight_decay"] > 0.0]
    assert len(has_decay) > 0
