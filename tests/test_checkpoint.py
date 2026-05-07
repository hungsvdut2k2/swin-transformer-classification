import torch
import torch.nn as nn
from foodnet.training.checkpoint import save_checkpoint, load_checkpoint


def test_save_load_checkpoint_roundtrip(tmp_path):
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(tmp_path / "ckpt.pt", model, opt, scaler=None, epoch=3, best_metric=0.42, extras={"args_sha": "abc"})
    model2 = nn.Linear(4, 2)
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    state = load_checkpoint(tmp_path / "ckpt.pt", model2, opt2, scaler=None, map_location="cpu")
    assert state["epoch"] == 3
    assert state["best_metric"] == 0.42
    assert state["extras"]["args_sha"] == "abc"
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
