from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from foodnet.cli.evaluate import main as eval_main
from foodnet.training.checkpoint import save_checkpoint


def _stub_swin_checkpoint(out: Path, num_classes: int = 3) -> None:
    from foodnet.models.factory import build_model
    model = build_model("swin_tiny_patch4_window7_224", num_classes=num_classes, pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    save_checkpoint(out, model, opt, scaler=None, epoch=0, best_metric=0.0)


def test_evaluate_cli_writes_parquet(tiny_food, tmp_path):
    ckpt = tmp_path / "best.pt"
    _stub_swin_checkpoint(ckpt, num_classes=3)
    out_parquet = tmp_path / "test_preds.parquet"
    rc = eval_main([
        "--checkpoint", str(ckpt),
        "--splits-dir", str(tiny_food["splits_dir"]),
        "--images-root", str(tiny_food["images_root"]),
        "--split", "test",
        "--output-parquet", str(out_parquet),
        "--arch", "swin_tiny_patch4_window7_224",
        "--num-classes", "3",
        "--batch-size", "2",
        "--num-workers", "0",
        "--device", "cpu",
        "--no-amp",
        "--no-pretrained",
    ])
    assert rc == 0
    df = pd.read_parquet(out_parquet)
    assert len(df) > 0
    assert {"filepath", "label", "loss", "p_0", "p_1", "p_2"} <= set(df.columns)
