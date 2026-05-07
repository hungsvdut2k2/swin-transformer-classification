"""Pytest fixtures: synthetic Food-101-like dataset (3 classes × 10 imgs)."""
from __future__ import annotations
import csv
import random
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_food(tmp_path: Path) -> dict:
    """Build a 3-class × 10-image synthetic dataset under tmp_path.

    Returns a dict with keys: images_root, splits_dir, classes (list[str]),
    class_to_idx (dict), bad_file (relative path to one corrupt entry).
    """
    rng = random.Random(0)
    classes = ["apple_pie", "baby_back_ribs", "baklava"]
    images_root = tmp_path / "images"
    images_root.mkdir()
    rows = []
    bad_relpath = None
    for cls_idx, cls in enumerate(classes):
        cls_dir = images_root / cls
        cls_dir.mkdir()
        for i in range(10):
            arr = (np.random.RandomState(cls_idx * 100 + i).rand(32, 32, 3) * 255).astype("uint8")
            img = Image.fromarray(arr, mode="RGB")
            relpath = f"{cls}/{cls}_{i:03d}.jpg"
            img.save(images_root / relpath)
            rows.append((relpath, cls_idx, cls))
        if cls_idx == 0:
            # Make one file corrupt to exercise the PIL-error path
            bad_relpath = f"{cls}/{cls}_009.jpg"
            (images_root / bad_relpath).write_bytes(b"not-an-image")
    rng.shuffle(rows)

    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    # 8/1/1 of 10 = 8/1/1 per class
    train, val, test = [], [], []
    by_cls: dict[int, list] = {}
    for r in rows:
        by_cls.setdefault(r[1], []).append(r)
    for cls_idx, items in by_cls.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        rng2 = random.Random(42 + cls_idx)
        rng2.shuffle(items_sorted)
        train += items_sorted[:8]
        val += items_sorted[8:9]
        test += items_sorted[9:10]
    for name, split_rows in [("train", train), ("val", val), ("test", test)]:
        with (splits_dir / f"{name}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "label", "class_name"])
            for r in split_rows:
                w.writerow(r)
    with (splits_dir / "classes.txt").open("w") as f:
        for c in classes:
            f.write(c + "\n")
    return {
        "images_root": images_root,
        "splits_dir": splits_dir,
        "classes": classes,
        "class_to_idx": {c: i for i, c in enumerate(classes)},
        "bad_file": bad_relpath,
    }
