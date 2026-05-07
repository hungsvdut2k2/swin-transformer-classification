import json
from pathlib import Path
import pytest
from foodnet.cli.train import main as train_main


@pytest.mark.slow
def test_train_cli_smoke_one_epoch(tiny_food, tmp_path):
    """1 epoch on synthetic data, CPU, no pretrained, no AMP, no W&B.

    Verifies: args.json + last.pt + best.pt + train_log.csv all written.
    """
    out_dir = tmp_path / "runs" / "smoke"
    rc = train_main([
        "--splits-dir", str(tiny_food["splits_dir"]),
        "--images-root", str(tiny_food["images_root"]),
        "--output-dir", str(out_dir),
        "--arch", "swin_tiny_patch4_window7_224",
        "--num-classes", "3",
        "--epochs", "1",
        "--batch-size", "4",
        "--num-workers", "0",
        "--img-size", "224",
        "--lr", "1e-4",
        "--no-pretrained",
        "--no-amp",
        "--no-wandb",
        "--no-early-stop",
        "--mixup-alpha", "0.0",
        "--cutmix-alpha", "0.0",
        "--seed", "0",
        "--device", "cpu",
    ])
    assert rc == 0
    assert (out_dir / "args.json").exists()
    assert (out_dir / "last.pt").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "train_log.csv").exists()
    args = json.loads((out_dir / "args.json").read_text())
    assert args["epochs"] == 1


def test_train_cli_refuses_to_clobber(tiny_food, tmp_path):
    out_dir = tmp_path / "runs" / "guarded"
    out_dir.mkdir(parents=True)
    (out_dir / "last.pt").write_text("placeholder")
    with pytest.raises(SystemExit) as exc:
        train_main([
            "--splits-dir", str(tiny_food["splits_dir"]),
            "--images-root", str(tiny_food["images_root"]),
            "--output-dir", str(out_dir),
            "--num-classes", "3",
            "--epochs", "1",
            "--no-pretrained", "--no-amp", "--no-wandb", "--no-early-stop",
            "--device", "cpu",
        ])
    assert exc.value.code != 0
