import csv
import json
from pathlib import Path
import pandas as pd
from foodnet.cli.split import main as split_main


def _build_dataset(images_root: Path, n_classes: int = 3, per_class: int = 100) -> None:
    from PIL import Image
    import numpy as np
    images_root.mkdir(parents=True, exist_ok=True)
    for c in range(n_classes):
        cls = f"class_{c}"
        (images_root / cls).mkdir(exist_ok=True)
        for i in range(per_class):
            arr = (np.random.RandomState(c * 1000 + i).rand(8, 8, 3) * 255).astype("uint8")
            Image.fromarray(arr, "RGB").save(images_root / cls / f"{cls}_{i:04d}.jpg")


def test_split_cli_writes_csvs_and_meta(tmp_path: Path):
    images_root = tmp_path / "food-101" / "images"
    _build_dataset(images_root)
    out = tmp_path / "splits"
    rc = split_main([
        "--images-root", str(images_root),
        "--output-dir", str(out),
        "--seed", "42",
        "--ratios", "0.8,0.1,0.1",
    ])
    assert rc == 0
    for name, expected in [("train", 240), ("val", 30), ("test", 30)]:
        df = pd.read_csv(out / f"{name}.csv")
        assert len(df) == expected
        assert set(df.columns) == {"filepath", "label", "class_name"}
    assert (out / "classes.txt").read_text().strip().splitlines() == ["class_0", "class_1", "class_2"]
    meta = json.loads((out / "splits_meta.json").read_text())
    assert meta["seed"] == 42
    assert meta["ratios"] == [0.8, 0.1, 0.1]
    assert set(meta["sha256"].keys()) == {"train.csv", "val.csv", "test.csv"}
    assert meta["counts"] == {"train": 240, "val": 30, "test": 30}
