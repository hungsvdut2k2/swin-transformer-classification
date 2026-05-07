import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from foodnet.eval.runner import run_inference


class _ToyModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


def test_run_inference_writes_parquet(tiny_food, tmp_path):
    from torchvision import transforms as T
    from torch.utils.data import DataLoader
    from foodnet.data.dataset import SplitsDataset

    ds = SplitsDataset(tiny_food["splits_dir"] / "test.csv", tiny_food["images_root"], transform=T.Compose([T.Resize((16, 16)), T.ToTensor()]))
    dl = DataLoader(ds, batch_size=2, num_workers=0)
    model = _ToyModel(num_classes=3)
    out_parquet = tmp_path / "preds.parquet"
    run_inference(model, dl, num_classes=3, device="cpu", filepaths=ds.filepaths, out_path=out_parquet, amp=False)

    df = pd.read_parquet(out_parquet)
    assert len(df) == len(ds)
    assert "filepath" in df.columns and "label" in df.columns and "loss" in df.columns
    softmax_cols = [c for c in df.columns if c.startswith("p_")]
    assert len(softmax_cols) == 3
    arr = df[softmax_cols].to_numpy()
    np.testing.assert_allclose(arr.sum(axis=1), 1.0, atol=1e-5)
