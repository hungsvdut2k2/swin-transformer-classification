# Phases 3 + 4 + 5 `foodnet` Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pip-installable `foodnet` package with four CLIs (split, train, evaluate, analyze) and one unified notebook that takes Phase 2 outputs and produces a Swin-Tiny baseline + full evaluation dashboard on Kaggle P100.

**Architecture:** Capability-layered package — `splitting/`, `data/`, `models/`, `training/`, `eval/`, `cli/`, `utils/`. Each subpackage has one job; CLIs in `foodnet/cli/` are thin shells around library functions. Two-stage evaluation: `evaluate` writes per-example softmax+loss parquet once, `analyze` re-renders dashboards from that parquet many times.

**Tech Stack:** PyTorch 2.2+, timm 1.0+ (Swin-Tiny + transforms + Mixup + cosine LR), torchvision, W&B, pandas+pyarrow (splits CSV + eval parquet), matplotlib (figures), pytest (synthetic-data fixtures), argparse (no YAML/Hydra).

**Reference docs:**
- Spec: [docs/superpowers/specs/2026-05-07-phases-3-4-5-package-design.md](../specs/2026-05-07-phases-3-4-5-package-design.md)
- Decision: [docs/superpowers/specs/2026-05-07-phases-3-4-5-package-decision.md](../specs/2026-05-07-phases-3-4-5-package-decision.md)
- Phase 2 outputs (consumed): `splits/{train,val,test}.csv`, `artifacts/phase2/{eda_stats.json,bad_files.json}`

---

## File Structure

**Created (package):**
- `pyproject.toml` — package metadata, console_scripts entry points
- `foodnet/__init__.py` — package version
- `foodnet/splitting/__init__.py`
- `foodnet/splitting/stratified.py` — per-class shuffle-and-slice 8:1:1 splitter
- `foodnet/data/__init__.py`
- `foodnet/data/splits.py` — load_split, load_class_names
- `foodnet/data/dataset.py` — `SplitsDataset` with PIL-error resample
- `foodnet/data/transforms.py` — train/eval transforms via `timm.data.create_transform`
- `foodnet/data/mixup.py` — `build_mixup_fn` wrapper around `timm.data.Mixup`
- `foodnet/models/__init__.py`
- `foodnet/models/factory.py` — `build_model` (Swin-Tiny via timm)
- `foodnet/models/llrd.py` — `llrd_param_groups` (5 layer buckets)
- `foodnet/training/__init__.py`
- `foodnet/training/optim.py` — `build_optimizer`, `build_scheduler`
- `foodnet/training/early_stop.py` — `EarlyStopper`
- `foodnet/training/loop.py` — `train_one_epoch`, `validate`
- `foodnet/training/checkpoint.py` — `save_checkpoint`, `load_checkpoint`
- `foodnet/eval/__init__.py`
- `foodnet/eval/metrics.py` — top-k, macro-F1, per-class accuracy
- `foodnet/eval/confusion.py` — confusion matrix + most-confused pairs
- `foodnet/eval/calibration.py` — ECE + reliability diagram
- `foodnet/eval/slices.py` — worst-K classes, hardest examples
- `foodnet/eval/runner.py` — inference pass that writes per-example parquet
- `foodnet/cli/__init__.py`
- `foodnet/cli/split.py` — `foodnet-split` CLI
- `foodnet/cli/train.py` — `foodnet-train` CLI
- `foodnet/cli/evaluate.py` — `foodnet-evaluate` CLI
- `foodnet/cli/analyze.py` — `foodnet-analyze` CLI
- `foodnet/utils/__init__.py`
- `foodnet/utils/seed.py` — `set_seed`
- `foodnet/utils/paths.py` — `detect_env`, `ensure_dir`
- `foodnet/utils/wandb_logger.py` — `WandBLogger` (no-op when disabled)
- `foodnet/utils/config_dump.py` — `dump_args` (writes args.json)

**Created (tests):**
- `tests/__init__.py`
- `tests/conftest.py` — synthetic 32×32 Food-101-like fixture
- `tests/test_stratified.py`
- `tests/test_splits_loader.py`
- `tests/test_dataset.py`
- `tests/test_transforms.py`
- `tests/test_mixup.py`
- `tests/test_factory.py`
- `tests/test_llrd.py`
- `tests/test_optim.py`
- `tests/test_early_stop.py`
- `tests/test_loop.py`
- `tests/test_checkpoint.py`
- `tests/test_metrics.py`
- `tests/test_confusion.py`
- `tests/test_calibration.py`
- `tests/test_slices.py`
- `tests/test_runner.py`
- `tests/test_cli_split.py`
- `tests/test_cli_train.py`
- `tests/test_cli_evaluate.py`
- `tests/test_cli_analyze.py`

**Created (notebook + dirs):**
- `notebooks/phases_3_4_5_pipeline.ipynb` — unified driver
- `runs/.gitkeep` — output dir

**Modified:**
- `requirements.txt` — add timm>=1.0, wandb>=0.16, torchmetrics>=1.3, pytest>=8.0, seaborn>=0.13
- `.gitignore` — add `runs/`, `*.egg-info/`, `wandb/`

---

## Task 1: Repo scaffolding + dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `foodnet/__init__.py`, `foodnet/{splitting,data,models,training,eval,cli,utils}/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `runs/.gitkeep`
- Modify: `requirements.txt`, `.gitignore`

- [ ] **Step 1: Create the package directory tree**

```bash
mkdir -p foodnet/{splitting,data,models,training,eval,cli,utils} tests runs
touch foodnet/__init__.py \
      foodnet/splitting/__init__.py \
      foodnet/data/__init__.py \
      foodnet/models/__init__.py \
      foodnet/training/__init__.py \
      foodnet/eval/__init__.py \
      foodnet/cli/__init__.py \
      foodnet/utils/__init__.py \
      tests/__init__.py \
      runs/.gitkeep
```

- [ ] **Step 2: Set the package version**

Write `foodnet/__init__.py`:

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write pyproject.toml**

Write `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "foodnet"
version = "0.1.0"
description = "Swin-Tiny on Food-101 — phases 3+4+5 package"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "timm>=1.0",
    "Pillow>=10.0",
    "numpy>=1.26",
    "pandas>=2.1",
    "pyarrow>=15.0",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "scikit-learn>=1.4",
    "tqdm>=4.66",
    "wandb>=0.16",
    "torchmetrics>=1.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
foodnet-split = "foodnet.cli.split:main"
foodnet-train = "foodnet.cli.train:main"
foodnet-evaluate = "foodnet.cli.evaluate:main"
foodnet-analyze = "foodnet.cli.analyze:main"

[tool.setuptools.packages.find]
include = ["foodnet*"]
```

- [ ] **Step 4: Update requirements.txt**

Replace `requirements.txt` (extends Phase 2 deps with Phase 3+ libs):

```
torch>=2.2
torchvision>=0.17
timm>=1.0
Pillow>=10.0
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
seaborn>=0.13
imagehash>=4.3
scikit-learn>=1.4
tqdm>=4.66
jupyter>=1.0
nbformat>=5.9
pyarrow>=15.0
wandb>=0.16
torchmetrics>=1.3
pytest>=8.0
```

- [ ] **Step 5: Update .gitignore**

Append to `.gitignore`:

```
# Phase 3+ artifacts
runs/
wandb/
*.egg-info/
```

- [ ] **Step 6: Write tests/conftest.py — synthetic Food-101 fixture**

Write `tests/conftest.py`:

```python
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
```

- [ ] **Step 7: Verify install works**

Run:

```bash
pip install -e . 2>&1 | tail -5
python -c "import foodnet; print(foodnet.__version__)"
```

Expected: prints `0.1.0`. (If `pip install -e .` is too slow or torch is already pinned, the developer can skip the install and run pytest with `PYTHONPATH=.`.)

- [ ] **Step 8: Verify pytest collects tests/conftest.py**

Run:

```bash
PYTHONPATH=. pytest tests/ --collect-only 2>&1 | tail -5
```

Expected: `no tests ran` or `0 tests collected` — fine. Importantly, no import errors from conftest.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml foodnet/ tests/ runs/.gitkeep requirements.txt .gitignore
git commit -m "feat(foodnet): scaffold package + synthetic test fixture"
```

---

## Task 2: Stratified 8:1:1 splitter

**Files:**
- Create: `foodnet/splitting/stratified.py`
- Test: `tests/test_stratified.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_stratified.py`:

```python
from pathlib import Path
import pandas as pd
import pytest
from foodnet.splitting.stratified import stratified_split


def _make_df(n_per_class: int = 1000, n_classes: int = 3) -> pd.DataFrame:
    rows = []
    for c in range(n_classes):
        for i in range(n_per_class):
            rows.append({"filepath": f"class_{c}/img_{i:04d}.jpg", "label": c, "class_name": f"class_{c}"})
    return pd.DataFrame(rows)


def test_stratified_split_sizes_per_class():
    df = _make_df(1000, 3)
    train, val, test = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    for cls in range(3):
        assert (train["label"] == cls).sum() == 800
        assert (val["label"] == cls).sum() == 100
        assert (test["label"] == cls).sum() == 100


def test_stratified_split_disjoint():
    df = _make_df(1000, 3)
    train, val, test = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    s_train = set(train["filepath"])
    s_val = set(val["filepath"])
    s_test = set(test["filepath"])
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    assert len(s_train) + len(s_val) + len(s_test) == 3000


def test_stratified_split_deterministic():
    df = _make_df(100, 3)
    a = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    b = stratified_split(df, ratios=(0.8, 0.1, 0.1), seed=42)
    for da, db in zip(a, b):
        pd.testing.assert_frame_equal(da.reset_index(drop=True), db.reset_index(drop=True))


def test_stratified_split_ratios_must_sum_to_one():
    df = _make_df(10, 2)
    with pytest.raises(ValueError):
        stratified_split(df, ratios=(0.7, 0.1, 0.1), seed=42)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_stratified.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'foodnet.splitting.stratified'`.

- [ ] **Step 3: Write the implementation**

Write `foodnet/splitting/stratified.py`:

```python
"""Per-class shuffle-and-slice stratified split."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test) by per-class shuffle-and-slice.

    Each class is shuffled independently with a class-derived seed, then sliced
    by the given ratios. Counts use floor for train and val; test gets the
    remainder so all rows are accounted for.

    Args:
        df: Frame with columns including ``label``.
        ratios: (train, val, test). Must sum to 1.0 within 1e-6.
        seed: Master seed; per-class seed is ``seed + label``.

    Returns:
        (train_df, val_df, test_df), reset_index applied.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)} from {ratios}")
    r_train, r_val, _ = ratios
    train_parts, val_parts, test_parts = [], [], []
    for label, group in df.groupby("label", sort=True):
        items = group.sample(frac=1.0, random_state=seed + int(label)).reset_index(drop=True)
        n = len(items)
        n_train = int(math.floor(n * r_train))
        n_val = int(math.floor(n * r_val))
        train_parts.append(items.iloc[:n_train])
        val_parts.append(items.iloc[n_train : n_train + n_val])
        test_parts.append(items.iloc[n_train + n_val :])
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, val, test
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_stratified.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/splitting/stratified.py tests/test_stratified.py
git commit -m "feat(foodnet/splitting): per-class stratified 8:1:1 splitter"
```

---

## Task 3: Splits CLI (`foodnet-split`)

**Files:**
- Create: `foodnet/cli/split.py`
- Test: `tests/test_cli_split.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_cli_split.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_cli_split.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'foodnet.cli.split'`.

- [ ] **Step 3: Write the implementation**

Write `foodnet/cli/split.py`:

```python
"""`foodnet-split` CLI: scan an images_root and write 8:1:1 split CSVs."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence
import pandas as pd
from foodnet.splitting.stratified import stratified_split


IMG_EXT = {".jpg", ".jpeg", ".png"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-split", description="Generate 8:1:1 stratified splits over an images_root.")
    p.add_argument("--images-root", type=Path, required=True, help="Directory whose immediate children are class folders.")
    p.add_argument("--output-dir", type=Path, required=True, help="Where to write {train,val,test}.csv and metadata.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", type=str, default="0.8,0.1,0.1", help="Comma-separated train,val,test ratios.")
    return p.parse_args(argv)


def _scan(images_root: Path) -> tuple[pd.DataFrame, list[str]]:
    classes = sorted(d.name for d in images_root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"No class subdirectories found under {images_root}.")
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    rows: list[dict] = []
    for c in classes:
        for f in sorted((images_root / c).iterdir()):
            if f.suffix.lower() in IMG_EXT and f.is_file():
                rows.append({"filepath": f"{c}/{f.name}", "label": cls_to_idx[c], "class_name": c})
    return pd.DataFrame(rows), classes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args: argparse.Namespace) -> int:
    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3:
        raise SystemExit(f"--ratios must be 3 comma-separated floats, got {args.ratios!r}")
    df, classes = _scan(args.images_root)
    train, val, test = stratified_split(df, ratios=ratios, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train", train), ("val", val), ("test", test)]:
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    (args.output_dir / "classes.txt").write_text("\n".join(classes) + "\n")
    meta = {
        "seed": args.seed,
        "ratios": list(ratios),
        "n_classes": len(classes),
        "counts": {"train": len(train), "val": len(val), "test": len(test)},
        "sha256": {f"{n}.csv": _sha256(args.output_dir / f"{n}.csv") for n in ("train", "val", "test")},
    }
    (args.output_dir / "splits_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[split] wrote {len(train)}/{len(val)}/{len(test)} rows to {args.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_cli_split.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/cli/split.py tests/test_cli_split.py
git commit -m "feat(foodnet/cli): foodnet-split CLI with splits_meta.json"
```

---

## Task 4: Splits loader

**Files:**
- Create: `foodnet/data/splits.py`
- Test: `tests/test_splits_loader.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_splits_loader.py`:

```python
from foodnet.data.splits import load_split, load_class_names


def test_load_split_returns_filepaths_and_labels(tiny_food):
    paths, labels = load_split(tiny_food["splits_dir"] / "train.csv")
    assert len(paths) == len(labels)
    assert all(isinstance(p, str) for p in paths)
    assert set(labels) <= {0, 1, 2}


def test_load_class_names(tiny_food):
    names = load_class_names(tiny_food["splits_dir"] / "classes.txt")
    assert names == tiny_food["classes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_splits_loader.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/data/splits.py`:

```python
"""Loaders for split CSVs and class-name list."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    """Return (filepaths, labels) lists from a Phase 2/3 split CSV."""
    df = pd.read_csv(csv_path)
    return df["filepath"].astype(str).tolist(), df["label"].astype(int).tolist()


def load_class_names(classes_txt: Path) -> list[str]:
    """One class per line, in label-id order."""
    return [line.strip() for line in Path(classes_txt).read_text().splitlines() if line.strip()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_splits_loader.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/data/splits.py tests/test_splits_loader.py
git commit -m "feat(foodnet/data): split CSV + class-names loader"
```

---

## Task 5: `SplitsDataset` with PIL-error resample

**Files:**
- Create: `foodnet/data/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_dataset.py`:

```python
import torch
from torchvision import transforms as T
from foodnet.data.dataset import SplitsDataset


def _to_tensor():
    return T.Compose([T.Resize((16, 16)), T.ToTensor()])


def test_dataset_len_and_getitem(tiny_food):
    ds = SplitsDataset(
        csv_path=tiny_food["splits_dir"] / "train.csv",
        images_root=tiny_food["images_root"],
        transform=_to_tensor(),
    )
    assert len(ds) == 24
    x, y = ds[0]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 16, 16)
    assert isinstance(y, int) and 0 <= y <= 2


def test_dataset_resamples_on_pil_error(tiny_food):
    ds = SplitsDataset(
        csv_path=tiny_food["splits_dir"] / "train.csv",
        images_root=tiny_food["images_root"],
        transform=_to_tensor(),
    )
    bad = tiny_food["bad_file"]
    relpaths = [r for r, _ in zip(ds.filepaths, ds.labels)]
    if bad in relpaths:
        i = relpaths.index(bad)
        x, y = ds[i]
        assert isinstance(x, torch.Tensor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_dataset.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/data/dataset.py`:

```python
"""SplitsDataset: csv-driven Food-101 dataset with PIL-error resample."""
from __future__ import annotations
import logging
import random
from pathlib import Path
from typing import Callable, ClassVar
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from foodnet.data.splits import load_split

log = logging.getLogger(__name__)


class SplitsDataset(Dataset):
    """Reads a Phase 2/3 split CSV. On a corrupt/missing file, logs once per
    path and resamples a random index from the same dataset."""

    _warned: ClassVar[set[str]] = set()

    def __init__(
        self,
        csv_path: Path,
        images_root: Path,
        transform: Callable | None = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.transform = transform
        self.filepaths, self.labels = load_split(Path(csv_path))
        self._rng = random.Random(0)

    def __len__(self) -> int:
        return len(self.filepaths)

    def _open(self, idx: int) -> tuple[Image.Image, int]:
        rel = self.filepaths[idx]
        full = self.images_root / rel
        img = Image.open(full).convert("RGB")
        return img, self.labels[idx]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        try:
            img, label = self._open(idx)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            rel = self.filepaths[idx]
            if rel not in self._warned:
                self._warned.add(rel)
                log.warning("SplitsDataset: skipping unreadable file %s (%s)", rel, type(e).__name__)
            return self.__getitem__(self._rng.randrange(len(self.filepaths)))
        if self.transform is not None:
            img = self.transform(img)
        return img, label
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_dataset.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/data/dataset.py tests/test_dataset.py
git commit -m "feat(foodnet/data): SplitsDataset with PIL-error resample + log-once"
```

---

## Task 6: Train/eval transforms

**Files:**
- Create: `foodnet/data/transforms.py`
- Test: `tests/test_transforms.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_transforms.py`:

```python
import torch
from PIL import Image
import numpy as np
from foodnet.data.transforms import build_train_transform, build_eval_transform


def _img() -> Image.Image:
    return Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"), "RGB")


def test_train_transform_outputs_3x224x224():
    t = build_train_transform(img_size=224)
    out = t(_img())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)


def test_eval_transform_outputs_3x224x224():
    t = build_eval_transform(img_size=224)
    out = t(_img())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_transforms.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/data/transforms.py`:

```python
"""ImageNet-stat transforms via timm.data.create_transform."""
from __future__ import annotations
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_train_transform(
    img_size: int = 224,
    auto_augment: str = "rand-m9-mstd0.5-inc1",
    re_prob: float = 0.25,
    color_jitter: float = 0.4,
):
    """Train transform: RandomResizedCrop + HFlip + RandAugment + RandomErasing + ImageNet norm."""
    return create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment=auto_augment,
        re_prob=re_prob,
        re_mode="pixel",
        re_count=1,
        color_jitter=color_jitter,
        interpolation="bicubic",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )


def build_eval_transform(img_size: int = 224, crop_pct: float = 224 / 256):
    """Eval transform: Resize(round(img_size/crop_pct)) -> CenterCrop(img_size) -> Normalize."""
    return create_transform(
        input_size=img_size,
        is_training=False,
        interpolation="bicubic",
        crop_pct=crop_pct,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_transforms.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/data/transforms.py tests/test_transforms.py
git commit -m "feat(foodnet/data): train/eval transforms via timm"
```

---

## Task 7: Mixup/CutMix collator

**Files:**
- Create: `foodnet/data/mixup.py`
- Test: `tests/test_mixup.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_mixup.py`:

```python
import torch
from foodnet.data.mixup import build_mixup_fn


def test_mixup_fn_smooths_targets_to_one_hot_when_disabled():
    fn = build_mixup_fn(num_classes=10, mixup_alpha=0.0, cutmix_alpha=0.0, label_smoothing=0.1)
    assert fn is None  # disabled when both alphas are 0


def test_mixup_fn_blends_when_enabled():
    fn = build_mixup_fn(num_classes=10, mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1)
    assert fn is not None
    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3])
    x_out, y_out = fn(x, y)
    assert x_out.shape == x.shape
    assert y_out.shape == (4, 10)
    assert torch.allclose(y_out.sum(dim=1), torch.ones(4), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_mixup.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/data/mixup.py`:

```python
"""Mixup/CutMix builder. Returns None when both alphas are 0 (disabled)."""
from __future__ import annotations
from timm.data import Mixup


def build_mixup_fn(
    num_classes: int,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    label_smoothing: float = 0.1,
    prob: float = 1.0,
    switch_prob: float = 0.5,
):
    """Return a timm Mixup callable, or None if both alphas are 0."""
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=prob,
        switch_prob=switch_prob,
        mode="batch",
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_mixup.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/data/mixup.py tests/test_mixup.py
git commit -m "feat(foodnet/data): Mixup/CutMix builder via timm"
```

---

## Task 8: Model factory

**Files:**
- Create: `foodnet/models/factory.py`
- Test: `tests/test_factory.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_factory.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_factory.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/models/factory.py`:

```python
"""Swin-Tiny via timm with project defaults."""
from __future__ import annotations
import timm
import torch.nn as nn


def build_model(
    arch: str = "swin_tiny_patch4_window7_224",
    num_classes: int = 101,
    pretrained: bool = True,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """Build a timm model with a fresh classifier head sized to num_classes."""
    model = timm.create_model(
        arch,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_factory.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/models/factory.py tests/test_factory.py
git commit -m "feat(foodnet/models): Swin-Tiny factory via timm"
```

---

## Task 9: LLRD param groups

**Files:**
- Create: `foodnet/models/llrd.py`
- Test: `tests/test_llrd.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_llrd.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_llrd.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/models/llrd.py`:

```python
"""Layer-wise LR decay param groups for Swin (5 layer buckets)."""
from __future__ import annotations
from typing import Iterable
import torch.nn as nn

NO_DECAY_KEYWORDS = ("bias", "norm", "relative_position_bias_table", "absolute_pos_embed")


def _layer_id(name: str, num_layers: int = 4) -> int:
    """Map parameter name to a layer bucket id in [0, num_layers+1].

    0 = patch_embed (earliest)
    1..num_layers = layers.0 .. layers.{num_layers-1}
    num_layers+1 = head (latest)
    """
    if name.startswith("patch_embed") or name.startswith("absolute_pos_embed"):
        return 0
    if name.startswith("layers."):
        block_id = int(name.split(".")[1])
        return block_id + 1
    if name.startswith("norm.") or name.startswith("head"):
        return num_layers + 1
    return num_layers + 1  # default to head bucket


def _should_no_decay(name: str, param_shape: tuple[int, ...]) -> bool:
    if param_shape == () or len(param_shape) == 1:
        return True
    return any(k in name for k in NO_DECAY_KEYWORDS)


def llrd_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    layer_decay: float = 0.75,
    num_layers: int = 4,
) -> list[dict]:
    """Group parameters by (layer-id, decay/no-decay) and assign per-bucket lr.

    lr_scale[layer_id] = layer_decay ** ((num_layers + 1) - layer_id)
    so the head (layer_id = num_layers+1) has scale 1.0 and patch_embed the smallest.
    """
    layer_names = ["patch_embed"] + [f"layers.{i}" for i in range(num_layers)] + ["head"]
    n_buckets = num_layers + 2
    groups: dict[tuple[int, str], dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lid = _layer_id(name, num_layers=num_layers)
        nd = _should_no_decay(name, tuple(p.shape))
        key = (lid, "no_decay" if nd else "decay")
        if key not in groups:
            scale = layer_decay ** (n_buckets - 1 - lid)
            groups[key] = {
                "name": f"{layer_names[lid]}_{'no_decay' if nd else 'decay'}",
                "params": [],
                "lr": base_lr * scale,
                "weight_decay": 0.0 if nd else weight_decay,
                "lr_scale": scale,
            }
        groups[key]["params"].append(p)
    return [g for g in groups.values() if g["params"]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_llrd.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/models/llrd.py tests/test_llrd.py
git commit -m "feat(foodnet/models): LLRD param groups (5 layer buckets, no-decay on norms/biases)"
```

---

## Task 10: Optimizer + scheduler + AMP scaler

**Files:**
- Create: `foodnet/training/optim.py`
- Test: `tests/test_optim.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_optim.py`:

```python
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
    assert s.is_enabled() is True

    s2 = build_scaler(enabled=False)
    assert s2.is_enabled() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_optim.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/training/optim.py`:

```python
"""AdamW + timm CosineLRScheduler + AMP GradScaler."""
from __future__ import annotations
import torch
from timm.scheduler import CosineLRScheduler


def build_optimizer(
    param_groups: list[dict],
    kind: str = "adamw",
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    if kind != "adamw":
        raise ValueError(f"Only adamw is supported; got {kind}")
    return torch.optim.AdamW(param_groups, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 2,
    min_lr: float = 1e-6,
    warmup_lr_init: float = 1e-7,
) -> CosineLRScheduler:
    return CosineLRScheduler(
        optimizer,
        t_initial=num_epochs,
        lr_min=min_lr,
        warmup_t=warmup_epochs,
        warmup_lr_init=warmup_lr_init,
        cycle_limit=1,
        t_in_epochs=True,
    )


def build_scaler(enabled: bool = True) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=enabled)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_optim.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/training/optim.py tests/test_optim.py
git commit -m "feat(foodnet/training): AdamW + cosine-warmup + AMP scaler builders"
```

---

## Task 11: Early stopping

**Files:**
- Create: `foodnet/training/early_stop.py`
- Test: `tests/test_early_stop.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_early_stop.py`:

```python
from foodnet.training.early_stop import EarlyStopper


def test_early_stop_max_mode_improves_resets_counter():
    es = EarlyStopper(patience=3, mode="max", min_delta=1e-3)
    assert not es.step(0.50)
    assert not es.step(0.55)
    assert not es.step(0.60)
    assert not es.step(0.55)  # 1
    assert not es.step(0.59)  # 2
    assert not es.step(0.6005)  # tiny improvement < min_delta -> 3 -> stop
    assert es.should_stop


def test_early_stop_returns_true_at_patience():
    es = EarlyStopper(patience=2, mode="max", min_delta=0.0)
    es.step(0.5)
    es.step(0.4)  # 1
    stop = es.step(0.3)  # 2 -> stop
    assert stop is True
    assert es.should_stop


def test_early_stop_min_mode():
    es = EarlyStopper(patience=2, mode="min", min_delta=1e-3)
    assert not es.step(1.0)
    assert not es.step(0.5)  # better
    assert not es.step(0.49)  # 1 (improvement < min_delta)
    assert es.step(0.50)  # 2 -> stop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_early_stop.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/training/early_stop.py`:

```python
"""Early stopping on a scalar monitor (val/top1 by default)."""
from __future__ import annotations
import math


class EarlyStopper:
    """Stop after `patience` epochs with no improvement greater than min_delta.

    mode='max' for metrics like accuracy; mode='min' for losses.
    `step(value)` returns True when training should stop.
    """

    def __init__(self, patience: int = 8, mode: str = "max", min_delta: float = 1e-3) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self._best = -math.inf if mode == "max" else math.inf
        self._stale = 0
        self.should_stop = False

    def _better(self, current: float) -> bool:
        if self.mode == "max":
            return current > self._best + self.min_delta
        return current < self._best - self.min_delta

    def step(self, current: float) -> bool:
        if self._better(current):
            self._best = current
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                self.should_stop = True
        return self.should_stop

    @property
    def best(self) -> float:
        return self._best
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_early_stop.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/training/early_stop.py tests/test_early_stop.py
git commit -m "feat(foodnet/training): EarlyStopper with min_delta (max/min modes)"
```

---

## Task 12: Training loop functions + checkpoint

**Files:**
- Create: `foodnet/training/loop.py`
- Create: `foodnet/training/checkpoint.py`
- Test: `tests/test_loop.py`
- Test: `tests/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_loop.py`:

```python
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
```

Write `tests/test_checkpoint.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_loop.py tests/test_checkpoint.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 3: Write `foodnet/training/loop.py`**

```python
"""train_one_epoch + validate. AMP-aware, grad-clip-aware, mixup-aware."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def _topk_correct(logits: torch.Tensor, targets: torch.Tensor, k: int) -> int:
    k = min(k, logits.size(1))
    _, pred = logits.topk(k, dim=1)
    return pred.eq(targets.unsqueeze(1)).any(dim=1).sum().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
    grad_clip: float = 1.0,
    mixup_fn=None,
    amp: bool = True,
) -> dict:
    model.train()
    loss_sum, n = 0.0, 0
    steps = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if mixup_fn is not None:
            x, y = mixup_fn(x, y)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
        steps += 1
    return {"loss": loss_sum / max(n, 1), "steps": steps}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    amp: bool = True,
) -> dict:
    model.eval()
    loss_sum, n = 0.0, 0
    top1, top5 = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
        top1 += _topk_correct(logits, y, 1)
        top5 += _topk_correct(logits, y, 5)
    return {
        "loss": loss_sum / max(n, 1),
        "top1": top1 / max(n, 1),
        "top5": top5 / max(n, 1),
        "n": n,
    }
```

- [ ] **Step 4: Write `foodnet/training/checkpoint.py`**

```python
"""Save/load training checkpoints (model + optimizer + scaler + metadata)."""
from __future__ import annotations
from pathlib import Path
import torch


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scaler,
    epoch: int,
    best_metric: float,
    extras: dict | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "extras": extras or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scaler=None,
    map_location: str = "cpu",
) -> dict:
    state = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=. pytest tests/test_loop.py tests/test_checkpoint.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add foodnet/training/loop.py foodnet/training/checkpoint.py tests/test_loop.py tests/test_checkpoint.py
git commit -m "feat(foodnet/training): train_one_epoch, validate, checkpoint save/load"
```

---

## Task 13: Utils — seed, paths, wandb logger, config dump

**Files:**
- Create: `foodnet/utils/seed.py`
- Create: `foodnet/utils/paths.py`
- Create: `foodnet/utils/wandb_logger.py`
- Create: `foodnet/utils/config_dump.py`

(No dedicated unit tests for these — they are exercised end-to-end in CLI tests.)

- [ ] **Step 1: Write `foodnet/utils/seed.py`**

```python
"""Deterministic seeding for python, numpy, torch."""
from __future__ import annotations
import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 2: Write `foodnet/utils/paths.py`**

```python
"""Environment detection + small path helpers."""
from __future__ import annotations
from pathlib import Path


def detect_env() -> str:
    """Return 'kaggle' if running on Kaggle, else 'local'."""
    if Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


def ensure_dir(path: Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
```

- [ ] **Step 3: Write `foodnet/utils/wandb_logger.py`**

```python
"""Tiny W&B wrapper that no-ops when disabled (so tests + offline runs work)."""
from __future__ import annotations
from typing import Any


class WandBLogger:
    def __init__(self, enabled: bool, project: str | None = None, run_name: str | None = None, config: dict | None = None) -> None:
        self.enabled = enabled
        self._run = None
        if not enabled:
            return
        import wandb  # imported lazily so tests don't require wandb auth
        self._run = wandb.init(project=project, name=run_name, config=config or {}, reinit=True)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.log(payload, step=step)

    def finish(self) -> None:
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.finish()
```

- [ ] **Step 4: Write `foodnet/utils/config_dump.py`**

```python
"""Dump argparse.Namespace as args.json next to a checkpoint dir."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def dump_args(args: argparse.Namespace, out_dir: Path, filename: str = "args.json") -> Path:
    out = Path(out_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out
```

- [ ] **Step 5: Smoke import**

Run:

```bash
PYTHONPATH=. python -c "from foodnet.utils.seed import set_seed; from foodnet.utils.paths import detect_env, ensure_dir; from foodnet.utils.wandb_logger import WandBLogger; from foodnet.utils.config_dump import dump_args; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add foodnet/utils/
git commit -m "feat(foodnet/utils): seed, env-detect, W&B wrapper, args.json dump"
```

---

## Task 14: Train CLI (`foodnet-train`)

**Files:**
- Create: `foodnet/cli/train.py`
- Test: `tests/test_cli_train.py`

- [ ] **Step 1: Write the failing test** (uses `tiny_food` fixture; runs 1 epoch on CPU, no AMP, no W&B)

Write `tests/test_cli_train.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_cli_train.py -v -m "not slow"`
Expected: 1 failure (clobber test) — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/cli/train.py`:

```python
"""`foodnet-train`: end-to-end training CLI for the Swin-Tiny baseline."""
from __future__ import annotations
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Sequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm.loss import SoftTargetCrossEntropy
from timm.utils import accuracy as _timm_acc  # noqa: F401  (kept for symmetry; we use our own loop)
from foodnet.data.dataset import SplitsDataset
from foodnet.data.transforms import build_train_transform, build_eval_transform
from foodnet.data.mixup import build_mixup_fn
from foodnet.models.factory import build_model
from foodnet.models.llrd import llrd_param_groups
from foodnet.training.optim import build_optimizer, build_scheduler, build_scaler
from foodnet.training.loop import train_one_epoch, validate
from foodnet.training.checkpoint import save_checkpoint, load_checkpoint
from foodnet.training.early_stop import EarlyStopper
from foodnet.utils.seed import set_seed
from foodnet.utils.paths import ensure_dir
from foodnet.utils.wandb_logger import WandBLogger
from foodnet.utils.config_dump import dump_args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-train")

    g = p.add_argument_group("data")
    g.add_argument("--splits-dir", type=Path, required=True)
    g.add_argument("--images-root", type=Path, required=True)
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument("--num-classes", type=int, default=101)
    g.add_argument("--num-workers", type=int, default=4)
    g.add_argument("--img-size", type=int, default=224)
    g.add_argument("--batch-size", type=int, default=64)

    g = p.add_argument_group("model")
    g.add_argument("--arch", type=str, default="swin_tiny_patch4_window7_224")
    g.add_argument("--drop-path", type=float, default=0.2)
    g.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    g.add_argument("--no-pretrained", dest="pretrained", action="store_false")

    g = p.add_argument_group("optim")
    g.add_argument("--lr", type=float, default=5e-4)
    g.add_argument("--weight-decay", type=float, default=0.05)
    g.add_argument("--layer-decay", type=float, default=0.75)
    g.add_argument("--epochs", type=int, default=30)
    g.add_argument("--warmup-epochs", type=int, default=2)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--label-smoothing", type=float, default=0.1)

    g = p.add_argument_group("aug")
    g.add_argument("--mixup-alpha", type=float, default=0.8)
    g.add_argument("--cutmix-alpha", type=float, default=1.0)
    g.add_argument("--re-prob", type=float, default=0.25)
    g.add_argument("--color-jitter", type=float, default=0.4)
    g.add_argument("--auto-augment", type=str, default="rand-m9-mstd0.5-inc1")

    g = p.add_argument_group("runtime")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--amp", dest="amp", action="store_true", default=True)
    g.add_argument("--no-amp", dest="amp", action="store_false")
    g.add_argument("--early-stop", dest="early_stop", action="store_true", default=True)
    g.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    g.add_argument("--early-stop-patience", type=int, default=8)
    g.add_argument("--early-stop-min-delta", type=float, default=1e-3)
    g.add_argument("--resume", action="store_true", default=False)

    g = p.add_argument_group("wandb")
    g.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    g.add_argument("--no-wandb", dest="wandb", action="store_false")
    g.add_argument("--wandb-project", type=str, default="foodnet")
    g.add_argument("--run-name", type=str, default=None)

    return p.parse_args(argv)


def _build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    t_train = build_train_transform(
        img_size=args.img_size,
        auto_augment=args.auto_augment,
        re_prob=args.re_prob,
        color_jitter=args.color_jitter,
    )
    t_eval = build_eval_transform(img_size=args.img_size)
    train_ds = SplitsDataset(args.splits_dir / "train.csv", args.images_root, transform=t_train)
    val_ds = SplitsDataset(args.splits_dir / "val.csv", args.images_root, transform=t_eval)
    pin = args.device.startswith("cuda")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    return train_dl, val_dl


def run(args: argparse.Namespace) -> int:
    out_dir = ensure_dir(args.output_dir)
    last_pt = out_dir / "last.pt"
    if last_pt.exists() and not args.resume:
        raise SystemExit(f"Refusing to clobber {last_pt}. Pass --resume to continue or pick a fresh --output-dir.")

    set_seed(args.seed)
    dump_args(args, out_dir)

    train_dl, val_dl = _build_loaders(args)
    device = args.device
    model = build_model(args.arch, num_classes=args.num_classes, pretrained=args.pretrained, drop_path_rate=args.drop_path).to(device)
    mixup_fn = build_mixup_fn(args.num_classes, args.mixup_alpha, args.cutmix_alpha, args.label_smoothing)

    if mixup_fn is not None:
        train_criterion: nn.Module = SoftTargetCrossEntropy()
    else:
        train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    groups = llrd_param_groups(model, base_lr=args.lr, weight_decay=args.weight_decay, layer_decay=args.layer_decay)
    optimizer = build_optimizer(groups, kind="adamw")
    scheduler = build_scheduler(optimizer, num_epochs=args.epochs, warmup_epochs=args.warmup_epochs, min_lr=args.min_lr)
    scaler = build_scaler(enabled=args.amp and device.startswith("cuda"))

    start_epoch, best_top1 = 0, 0.0
    if args.resume and last_pt.exists():
        state = load_checkpoint(last_pt, model, optimizer, scaler, map_location=device)
        start_epoch = state["epoch"] + 1
        best_top1 = state["best_metric"]
        print(f"[train] resumed from epoch {start_epoch} (best_top1={best_top1:.4f})")

    stopper = EarlyStopper(patience=args.early_stop_patience, mode="max", min_delta=args.early_stop_min_delta) if args.early_stop else None
    logger = WandBLogger(enabled=args.wandb, project=args.wandb_project, run_name=args.run_name, config=vars(args))
    log_csv = out_dir / "train_log.csv"
    new_log = not log_csv.exists()

    with log_csv.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["epoch", "train_loss", "val_loss", "val_top1", "val_top5", "elapsed_s"])
        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            tr = train_one_epoch(model, train_dl, optimizer, train_criterion, scaler, device=device, grad_clip=args.grad_clip, mixup_fn=mixup_fn, amp=args.amp)
            scheduler.step(epoch + 1)
            va = validate(model, val_dl, val_criterion, device=device, num_classes=args.num_classes, amp=args.amp)
            elapsed = time.time() - t0
            writer.writerow([epoch, f"{tr['loss']:.6f}", f"{va['loss']:.6f}", f"{va['top1']:.6f}", f"{va['top5']:.6f}", f"{elapsed:.2f}"])
            fh.flush()
            logger.log({"train/loss": tr["loss"], "val/loss": va["loss"], "val/top1": va["top1"], "val/top5": va["top5"], "epoch": epoch}, step=epoch)
            print(f"[train] epoch={epoch} train_loss={tr['loss']:.4f} val_top1={va['top1']:.4f} val_top5={va['top5']:.4f} ({elapsed:.1f}s)")

            save_checkpoint(last_pt, model, optimizer, scaler, epoch=epoch, best_metric=best_top1)
            if va["top1"] > best_top1:
                best_top1 = va["top1"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, epoch=epoch, best_metric=best_top1)
            if stopper is not None and stopper.step(va["top1"]):
                print(f"[train] early stop at epoch {epoch} (best={stopper.best:.4f})")
                break

    logger.finish()
    print(f"[train] done. best val/top1 = {best_top1:.4f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the fast test (clobber guard)**

Run: `PYTHONPATH=. pytest tests/test_cli_train.py::test_train_cli_refuses_to_clobber -v`
Expected: 1 passed.

- [ ] **Step 5: Run the slow smoke test (1-epoch CPU run)**

Run: `PYTHONPATH=. pytest tests/test_cli_train.py::test_train_cli_smoke_one_epoch -v`
Expected: 1 passed (will take 30-60s on CPU; the synthetic dataset is tiny).

- [ ] **Step 6: Commit**

```bash
git add foodnet/cli/train.py tests/test_cli_train.py
git commit -m "feat(foodnet/cli): foodnet-train CLI (LLRD+AdamW+cosine+AMP+mixup+early-stop+W&B)"
```

---

## Task 15: Eval metrics — top-k, macro-F1, per-class

**Files:**
- Create: `foodnet/eval/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_metrics.py`:

```python
import numpy as np
from foodnet.eval.metrics import topk_accuracy, macro_f1, per_class_accuracy


def test_topk_accuracy_perfect():
    softmax = np.eye(5)
    targets = np.arange(5)
    assert topk_accuracy(softmax, targets, k=1) == 1.0
    assert topk_accuracy(softmax, targets, k=5) == 1.0


def test_topk_accuracy_top5_includes_top1():
    softmax = np.array([
        [0.1, 0.2, 0.0, 0.6, 0.1],  # true=3 -> top1
        [0.0, 0.0, 0.4, 0.5, 0.1],  # true=1 -> not in top1, in top5 (rank 5)
    ])
    targets = np.array([3, 1])
    assert topk_accuracy(softmax, targets, k=1) == 0.5
    assert topk_accuracy(softmax, targets, k=5) == 1.0


def test_macro_f1_balanced_dataset():
    softmax = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
    ])
    targets = np.array([0, 1, 0, 1])
    f1 = macro_f1(softmax, targets, num_classes=2)
    assert f1 == 1.0


def test_per_class_accuracy_returns_one_per_class():
    softmax = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.9, 0.1, 0.0],  # wrong (true=2)
    ])
    targets = np.array([0, 1, 2, 2])
    pca = per_class_accuracy(softmax, targets, num_classes=3)
    assert len(pca) == 3
    assert pca[0] == 1.0
    assert pca[1] == 1.0
    assert pca[2] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_metrics.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/eval/metrics.py`:

```python
"""Top-k accuracy, macro-F1, per-class accuracy from softmax + targets."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score


def topk_accuracy(softmax: np.ndarray, targets: np.ndarray, k: int) -> float:
    """Fraction of rows whose true class is within the top-k predictions."""
    if k > softmax.shape[1]:
        k = softmax.shape[1]
    topk = np.argpartition(-softmax, kth=k - 1, axis=1)[:, :k]
    hits = (topk == targets[:, None]).any(axis=1)
    return float(hits.mean())


def macro_f1(softmax: np.ndarray, targets: np.ndarray, num_classes: int) -> float:
    preds = softmax.argmax(axis=1)
    return float(f1_score(targets, preds, labels=list(range(num_classes)), average="macro", zero_division=0))


def per_class_accuracy(softmax: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    preds = softmax.argmax(axis=1)
    accs = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = float((preds[mask] == c).mean())
    return accs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/eval/metrics.py tests/test_metrics.py
git commit -m "feat(foodnet/eval): top-k, macro-F1, per-class accuracy"
```

---

## Task 16: Confusion matrix + most-confused pairs

**Files:**
- Create: `foodnet/eval/confusion.py`
- Test: `tests/test_confusion.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_confusion.py`:

```python
import numpy as np
from foodnet.eval.confusion import confusion_matrix, most_confused_pairs


def test_confusion_matrix_shape_and_diagonal():
    softmax = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.1, 0.9],
        [0.9, 0.1, 0.0],
    ])
    targets = np.array([0, 1, 2, 2])
    cm = confusion_matrix(softmax, targets, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[1, 1] == 1
    assert cm[2, 2] == 1
    assert cm[2, 0] == 1


def test_most_confused_pairs_excludes_diagonal_and_returns_top_k():
    cm = np.array([
        [10, 1, 0],
        [2, 8, 0],
        [3, 0, 7],
    ])
    pairs = most_confused_pairs(cm, k=2, class_names=["a", "b", "c"])
    assert len(pairs) == 2
    assert pairs[0]["count"] == 3
    assert pairs[0]["true"] == "c"
    assert pairs[0]["pred"] == "a"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_confusion.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/eval/confusion.py`:

```python
"""Confusion matrix + most-confused (true, pred) pairs."""
from __future__ import annotations
import numpy as np


def confusion_matrix(softmax: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    preds = softmax.argmax(axis=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[int(t), int(p)] += 1
    return cm


def most_confused_pairs(
    cm: np.ndarray,
    k: int,
    class_names: list[str] | None = None,
) -> list[dict]:
    """Return the top-k off-diagonal cells sorted by count desc."""
    n = cm.shape[0]
    cells = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                cells.append({"true": class_names[i] if class_names else i, "pred": class_names[j] if class_names else j, "count": int(cm[i, j])})
    cells.sort(key=lambda d: d["count"], reverse=True)
    return cells[:k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_confusion.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/eval/confusion.py tests/test_confusion.py
git commit -m "feat(foodnet/eval): confusion matrix + most-confused pairs"
```

---

## Task 17: Calibration — ECE + reliability bins

**Files:**
- Create: `foodnet/eval/calibration.py`
- Test: `tests/test_calibration.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_calibration.py`:

```python
import numpy as np
from foodnet.eval.calibration import expected_calibration_error, reliability_bins


def test_ece_perfectly_calibrated_is_zero():
    # Confidence == accuracy in every bin → ECE = 0.
    n = 1000
    rng = np.random.default_rng(0)
    confs = rng.uniform(0.5, 1.0, size=n)
    correct = (rng.uniform(size=n) < confs).astype(np.int64)
    softmax = np.zeros((n, 2))
    softmax[np.arange(n), 0] = confs
    softmax[np.arange(n), 1] = 1 - confs
    targets = np.where(correct == 1, 0, 1)
    ece = expected_calibration_error(softmax, targets, n_bins=15)
    assert ece < 0.05


def test_reliability_bins_returns_15_bins():
    n = 100
    softmax = np.random.RandomState(0).rand(n, 5)
    softmax /= softmax.sum(axis=1, keepdims=True)
    targets = np.random.RandomState(1).randint(0, 5, size=n)
    bins = reliability_bins(softmax, targets, n_bins=15)
    assert len(bins["bin_lower"]) == 15
    assert len(bins["bin_upper"]) == 15
    assert len(bins["bin_acc"]) == 15
    assert len(bins["bin_conf"]) == 15
    assert len(bins["bin_count"]) == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_calibration.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/eval/calibration.py`:

```python
"""Expected Calibration Error + reliability bin data for a diagram."""
from __future__ import annotations
import numpy as np


def reliability_bins(softmax: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> dict:
    """Bin examples by their max-softmax confidence; report avg conf and acc per bin."""
    confs = softmax.max(axis=1)
    preds = softmax.argmax(axis=1)
    correct = (preds == targets).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        bin_count[b] = int(mask.sum())
        if bin_count[b] > 0:
            bin_acc[b] = float(correct[mask].mean())
            bin_conf[b] = float(confs[mask].mean())
    return {
        "bin_lower": edges[:-1].tolist(),
        "bin_upper": edges[1:].tolist(),
        "bin_acc": bin_acc.tolist(),
        "bin_conf": bin_conf.tolist(),
        "bin_count": bin_count.tolist(),
    }


def expected_calibration_error(softmax: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|."""
    bins = reliability_bins(softmax, targets, n_bins=n_bins)
    counts = np.array(bins["bin_count"], dtype=np.float64)
    accs = np.array(bins["bin_acc"])
    confs = np.array(bins["bin_conf"])
    n = counts.sum()
    if n == 0:
        return 0.0
    return float((counts / n * np.abs(accs - confs)).sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_calibration.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/eval/calibration.py tests/test_calibration.py
git commit -m "feat(foodnet/eval): ECE + 15-bin reliability data"
```

---

## Task 18: Eval slices — worst classes, hardest examples

**Files:**
- Create: `foodnet/eval/slices.py`
- Test: `tests/test_slices.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_slices.py`:

```python
import numpy as np
from foodnet.eval.slices import worst_k_classes, hardest_examples


def test_worst_k_classes_returns_lowest_acc():
    pca = np.array([0.9, 0.5, 0.7, 0.2])
    out = worst_k_classes(pca, k=2, class_names=["a", "b", "c", "d"])
    assert [r["class"] for r in out] == ["d", "b"]
    assert out[0]["accuracy"] == 0.2


def test_hardest_correct_high_loss_correct_only():
    losses = np.array([0.1, 2.0, 0.5, 3.0, 0.05])
    preds = np.array([0, 0, 0, 0, 0])
    targets = np.array([0, 1, 0, 1, 0])  # correct: 0,2,4
    out = hardest_examples(losses, preds, targets, k=2, kind="correct")
    assert len(out) == 2
    assert set(o["index"] for o in out) == {2, 0}


def test_hardest_incorrect_high_loss_incorrect_only():
    losses = np.array([0.1, 2.0, 0.5, 3.0, 0.05])
    preds = np.array([0, 0, 0, 0, 0])
    targets = np.array([0, 1, 0, 1, 0])  # incorrect: 1,3
    out = hardest_examples(losses, preds, targets, k=2, kind="incorrect")
    assert [o["index"] for o in out] == [3, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_slices.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/eval/slices.py`:

```python
"""Worst-K classes by accuracy; hardest correct / hardest incorrect examples by loss."""
from __future__ import annotations
import numpy as np


def worst_k_classes(per_class_acc: np.ndarray, k: int, class_names: list[str]) -> list[dict]:
    order = np.argsort(per_class_acc)
    out = []
    for i in order[:k]:
        out.append({"class": class_names[int(i)], "accuracy": float(per_class_acc[int(i)]), "label": int(i)})
    return out


def hardest_examples(
    losses: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    k: int,
    kind: str,
) -> list[dict]:
    if kind not in ("correct", "incorrect"):
        raise ValueError(f"kind must be 'correct' or 'incorrect', got {kind!r}")
    correct_mask = preds == targets
    mask = correct_mask if kind == "correct" else ~correct_mask
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    sub = losses[idx]
    order = np.argsort(-sub)[:k]
    out = []
    for j in order:
        i = int(idx[int(j)])
        out.append({"index": i, "loss": float(losses[i]), "pred": int(preds[i]), "target": int(targets[i])})
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_slices.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/eval/slices.py tests/test_slices.py
git commit -m "feat(foodnet/eval): worst-K classes + hardest-correct/incorrect examples"
```

---

## Task 19: Eval runner — inference once, write per-example parquet

**Files:**
- Create: `foodnet/eval/runner.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

Write `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. pytest tests/test_runner.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Write the implementation**

Write `foodnet/eval/runner.py`:

```python
"""Single-pass inference that writes per-example softmax + per-example loss to parquet."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader,
    num_classes: int,
    device: str,
    filepaths: list[str],
    out_path: Path,
    amp: bool = True,
) -> None:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="none")
        probs = F.softmax(logits.float(), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_targets.append(y.cpu().numpy())
        all_losses.append(loss.float().cpu().numpy())
    probs = np.concatenate(all_probs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    losses = np.concatenate(all_losses, axis=0)

    if len(filepaths) != len(targets):
        raise RuntimeError(f"filepaths ({len(filepaths)}) does not match inference rows ({len(targets)})")

    df = pd.DataFrame({"filepath": filepaths, "label": targets, "loss": losses})
    for c in range(num_classes):
        df[f"p_{c}"] = probs[:, c]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. pytest tests/test_runner.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add foodnet/eval/runner.py tests/test_runner.py
git commit -m "feat(foodnet/eval): inference runner writes per-example parquet"
```

---

## Task 20: `foodnet-evaluate` and `foodnet-analyze` CLIs

**Files:**
- Create: `foodnet/cli/evaluate.py`
- Create: `foodnet/cli/analyze.py`
- Test: `tests/test_cli_evaluate.py`
- Test: `tests/test_cli_analyze.py`

- [ ] **Step 1: Write `tests/test_cli_evaluate.py`**

```python
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
```

- [ ] **Step 2: Write `tests/test_cli_analyze.py`**

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
from foodnet.cli.analyze import main as analyze_main


def _make_preds(path: Path, n_per_class: int = 30, n_classes: int = 3) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_classes):
        for i in range(n_per_class):
            probs = rng.dirichlet(np.ones(n_classes))
            probs[c] += 0.6
            probs = probs / probs.sum()
            row = {"filepath": f"class_{c}/img_{i}.jpg", "label": c, "loss": float(-np.log(max(probs[c], 1e-12)))}
            for k in range(n_classes):
                row[f"p_{k}"] = float(probs[k])
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_analyze_cli_writes_dashboard(tmp_path: Path):
    parquet = tmp_path / "preds.parquet"
    _make_preds(parquet)
    classes_txt = tmp_path / "classes.txt"
    classes_txt.write_text("class_0\nclass_1\nclass_2\n")
    out = tmp_path / "dash"
    rc = analyze_main([
        "--predictions", str(parquet),
        "--classes", str(classes_txt),
        "--output-dir", str(out),
    ])
    assert rc == 0
    metrics = json.loads((out / "metrics.json").read_text())
    for key in ("top1", "top5", "macro_f1", "ece", "per_class_acc"):
        assert key in metrics
    assert (out / "confusion.png").exists()
    assert (out / "reliability.png").exists()
    assert (out / "dashboard.md").exists()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `PYTHONPATH=. pytest tests/test_cli_evaluate.py tests/test_cli_analyze.py -v`
Expected: FAIL — modules not found.

- [ ] **Step 4: Write `foodnet/cli/evaluate.py`**

```python
"""`foodnet-evaluate`: load checkpoint, run inference on one split, write parquet."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Sequence
from torch.utils.data import DataLoader
from foodnet.data.dataset import SplitsDataset
from foodnet.data.transforms import build_eval_transform
from foodnet.models.factory import build_model
from foodnet.training.checkpoint import load_checkpoint
from foodnet.eval.runner import run_inference


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--splits-dir", type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--output-parquet", type=Path, required=True)
    p.add_argument("--arch", type=str, default="swin_tiny_patch4_window7_224")
    p.add_argument("--num-classes", type=int, default=101)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", dest="amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--pretrained", dest="pretrained", action="store_true", default=False)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    transform = build_eval_transform(img_size=args.img_size)
    ds = SplitsDataset(args.splits_dir / f"{args.split}.csv", args.images_root, transform=transform)
    pin = args.device.startswith("cuda")
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=pin)
    model = build_model(args.arch, num_classes=args.num_classes, pretrained=args.pretrained).to(args.device)
    load_checkpoint(args.checkpoint, model, optimizer=None, scaler=None, map_location=args.device)
    run_inference(model, dl, num_classes=args.num_classes, device=args.device, filepaths=ds.filepaths, out_path=args.output_parquet, amp=args.amp)
    print(f"[evaluate] wrote {len(ds)} predictions to {args.output_parquet}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Write `foodnet/cli/analyze.py`**

```python
"""`foodnet-analyze`: read prediction parquet, write metrics + figures + dashboard.md."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from foodnet.data.splits import load_class_names
from foodnet.eval.metrics import topk_accuracy, macro_f1, per_class_accuracy
from foodnet.eval.confusion import confusion_matrix, most_confused_pairs
from foodnet.eval.calibration import expected_calibration_error, reliability_bins
from foodnet.eval.slices import worst_k_classes, hardest_examples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="foodnet-analyze")
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--classes", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--worst-k", type=int, default=10)
    p.add_argument("--hardest-k", type=int, default=20)
    p.add_argument("--n-bins", type=int, default=15)
    return p.parse_args(argv)


def _load(parquet: Path, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(parquet)
    cols = [f"p_{c}" for c in range(n_classes)]
    softmax = df[cols].to_numpy(dtype=np.float64)
    targets = df["label"].to_numpy(dtype=np.int64)
    losses = df["loss"].to_numpy(dtype=np.float64)
    filepaths = df["filepath"].to_numpy()
    return softmax, targets, losses, filepaths


def _plot_confusion(cm: np.ndarray, class_names: list[str], out: Path) -> None:
    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, n / 8), max(7, n / 8)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    sns.heatmap(cm_norm, ax=ax, cmap="Blues", cbar=True, square=True, xticklabels=False, yticklabels=False)
    ax.set_title(f"Confusion matrix (row-normalized, {n} classes)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_reliability(bins: dict, ece: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    centers = (np.array(bins["bin_lower"]) + np.array(bins["bin_upper"])) / 2
    ax.bar(centers, bins["bin_acc"], width=1 / len(centers), edgecolor="k", alpha=0.6, label="accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE={ece:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _write_dashboard(out: Path, metrics: dict, worst: list[dict], pairs: list[dict], hardest_correct: list[dict], hardest_incorrect: list[dict]) -> None:
    md = ["# Evaluation Dashboard\n"]
    md.append("## Headline metrics\n")
    md.append(f"- Top-1: {metrics['top1']:.4f}")
    md.append(f"- Top-5: {metrics['top5']:.4f}")
    md.append(f"- Macro-F1: {metrics['macro_f1']:.4f}")
    md.append(f"- ECE: {metrics['ece']:.4f}\n")
    md.append("## Worst-K classes\n")
    for r in worst:
        md.append(f"- {r['class']}: {r['accuracy']:.3f}")
    md.append("\n## Most-confused pairs (true → pred, count)\n")
    for r in pairs:
        md.append(f"- {r['true']} → {r['pred']}: {r['count']}")
    md.append("\n## Hardest correct examples\n")
    for r in hardest_correct:
        md.append(f"- idx={r['index']} loss={r['loss']:.3f} target={r['target']} pred={r['pred']}")
    md.append("\n## Hardest incorrect examples\n")
    for r in hardest_incorrect:
        md.append(f"- idx={r['index']} loss={r['loss']:.3f} target={r['target']} pred={r['pred']}")
    md.append("\n## Figures\n")
    md.append("- ![confusion](confusion.png)")
    md.append("- ![reliability](reliability.png)\n")
    out.write_text("\n".join(md))


def run(args: argparse.Namespace) -> int:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(args.classes)
    n_classes = len(class_names)
    softmax, targets, losses, _filepaths = _load(args.predictions, n_classes)

    top1 = topk_accuracy(softmax, targets, k=1)
    top5 = topk_accuracy(softmax, targets, k=5)
    f1 = macro_f1(softmax, targets, num_classes=n_classes)
    pca = per_class_accuracy(softmax, targets, num_classes=n_classes)
    cm = confusion_matrix(softmax, targets, num_classes=n_classes)
    pairs = most_confused_pairs(cm, k=20, class_names=class_names)
    bins = reliability_bins(softmax, targets, n_bins=args.n_bins)
    ece = expected_calibration_error(softmax, targets, n_bins=args.n_bins)
    worst = worst_k_classes(pca, k=args.worst_k, class_names=class_names)
    preds = softmax.argmax(axis=1)
    hardest_correct = hardest_examples(losses, preds, targets, k=args.hardest_k, kind="correct")
    hardest_incorrect = hardest_examples(losses, preds, targets, k=args.hardest_k, kind="incorrect")

    metrics = {
        "top1": top1, "top5": top5, "macro_f1": f1, "ece": ece,
        "per_class_acc": {class_names[i]: float(pca[i]) for i in range(n_classes)},
        "n": int(len(targets)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(out_dir / "confusion.csv")
    (out_dir / "most_confused.json").write_text(json.dumps(pairs, indent=2))
    (out_dir / "reliability.json").write_text(json.dumps({"ece": ece, **bins}, indent=2))
    _plot_confusion(cm, class_names, out_dir / "confusion.png")
    _plot_reliability(bins, ece, out_dir / "reliability.png")
    _write_dashboard(out_dir / "dashboard.md", metrics, worst, pairs[:20], hardest_correct, hardest_incorrect)
    print(f"[analyze] wrote dashboard to {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run both CLI tests**

Run: `PYTHONPATH=. pytest tests/test_cli_evaluate.py tests/test_cli_analyze.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add foodnet/cli/evaluate.py foodnet/cli/analyze.py tests/test_cli_evaluate.py tests/test_cli_analyze.py
git commit -m "feat(foodnet/cli): foodnet-evaluate (parquet) + foodnet-analyze (dashboard)"
```

---

## Task 21: Unified notebook + acceptance run

**Files:**
- Create: `notebooks/phases_3_4_5_pipeline.ipynb`

The notebook is a thin driver: section 0 detects environment, section 1 calls `foodnet-split`, section 2 calls `foodnet-train`, section 3 calls `foodnet-evaluate` + `foodnet-analyze`, section 4 renders `dashboard.md` summary.

- [ ] **Step 1: Build the notebook with `nbformat`**

Write `scripts/build_phase345_notebook.py` (a one-shot helper) — or run the equivalent inline:

```python
# Run in a python REPL or as a one-shot script.
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phases 3 + 4 + 5 — Swin-Tiny on Food-101\n"
    "Unified driver: split → train → evaluate → analyze.\n"
    "\n"
    "**Environment-aware:** detects Kaggle vs local at section 0."
))

cells.append(nbf.v4.new_markdown_cell("## 0. Environment + paths"))
cells.append(nbf.v4.new_code_cell(
    "from pathlib import Path\n"
    "from foodnet.utils.paths import detect_env\n"
    "ENV = detect_env()\n"
    "if ENV == 'kaggle':\n"
    "    IMAGES_ROOT = Path('/kaggle/input/food-101/food-101/images')\n"
    "    OUTPUT_DIR = Path('/kaggle/working/runs/baseline')\n"
    "    SPLITS_DIR = Path('/kaggle/working/splits')\n"
    "else:\n"
    "    IMAGES_ROOT = Path('data/food-101/images')\n"
    "    OUTPUT_DIR = Path('runs/baseline')\n"
    "    SPLITS_DIR = Path('splits')\n"
    "print({'env': ENV, 'images_root': str(IMAGES_ROOT), 'output_dir': str(OUTPUT_DIR), 'splits_dir': str(SPLITS_DIR)})"
))

cells.append(nbf.v4.new_markdown_cell("## 1. Generate (or refresh) 8:1:1 splits"))
cells.append(nbf.v4.new_code_cell(
    "REGENERATE_SPLITS = False  # set True to overwrite committed splits\n"
    "if REGENERATE_SPLITS or not (SPLITS_DIR / 'train.csv').exists():\n"
    "    !foodnet-split --images-root \"$IMAGES_ROOT\" --output-dir \"$SPLITS_DIR\" --seed 42 --ratios 0.8,0.1,0.1\n"
    "else:\n"
    "    print('splits already present at', SPLITS_DIR)"
))

cells.append(nbf.v4.new_markdown_cell("## 2. Train Swin-Tiny baseline"))
cells.append(nbf.v4.new_code_cell(
    "import os\n"
    "EPOCHS = 30\n"
    "BATCH = 64 if ENV == 'kaggle' else 32\n"
    "WANDB = '--wandb' if os.environ.get('WANDB_API_KEY') else '--no-wandb'\n"
    "!foodnet-train \\\n"
    "  --splits-dir \"$SPLITS_DIR\" \\\n"
    "  --images-root \"$IMAGES_ROOT\" \\\n"
    "  --output-dir \"$OUTPUT_DIR\" \\\n"
    "  --num-classes 101 --epochs $EPOCHS --batch-size $BATCH \\\n"
    "  --lr 5e-4 --weight-decay 0.05 --layer-decay 0.75 \\\n"
    "  --warmup-epochs 2 --grad-clip 1.0 --label-smoothing 0.1 \\\n"
    "  --mixup-alpha 0.8 --cutmix-alpha 1.0 --re-prob 0.25 \\\n"
    "  --early-stop --early-stop-patience 8 \\\n"
    "  --seed 42 --device cuda --amp $WANDB --wandb-project foodnet"
))

cells.append(nbf.v4.new_markdown_cell("## 3. Evaluate on test split + analyze"))
cells.append(nbf.v4.new_code_cell(
    "PRED_PARQUET = OUTPUT_DIR / 'test_preds.parquet'\n"
    "DASH_DIR = OUTPUT_DIR / 'analysis'\n"
    "!foodnet-evaluate \\\n"
    "  --checkpoint \"$OUTPUT_DIR/best.pt\" \\\n"
    "  --splits-dir \"$SPLITS_DIR\" --images-root \"$IMAGES_ROOT\" \\\n"
    "  --split test --output-parquet \"$PRED_PARQUET\" \\\n"
    "  --num-classes 101 --batch-size 128 --device cuda --amp\n"
    "!foodnet-analyze \\\n"
    "  --predictions \"$PRED_PARQUET\" \\\n"
    "  --classes \"$SPLITS_DIR/classes.txt\" \\\n"
    "  --output-dir \"$DASH_DIR\""
))

cells.append(nbf.v4.new_markdown_cell("## 4. Render dashboard summary"))
cells.append(nbf.v4.new_code_cell(
    "from IPython.display import Markdown, Image, display\n"
    "display(Markdown((DASH_DIR / 'dashboard.md').read_text()))"
))

nb["cells"] = cells
out = Path("notebooks/phases_3_4_5_pipeline.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
```

Run that script once:

```bash
PYTHONPATH=. python scripts/build_phase345_notebook.py
```

Or skip the helper and write the notebook by running the same code inline in a python -c block.

- [ ] **Step 2: Verify the notebook is valid JSON**

Run:

```bash
python -c "import nbformat; nb = nbformat.read('notebooks/phases_3_4_5_pipeline.ipynb', as_version=4); nbformat.validate(nb); print('valid', len(nb.cells), 'cells')"
```

Expected: `valid 9 cells` (or however many cells the spec adds).

- [ ] **Step 3: Run the full test suite**

Run:

```bash
PYTHONPATH=. pytest tests/ -v
```

Expected: all tests pass (~30 tests across the 21 test files).

- [ ] **Step 4: Commit the notebook**

```bash
git add notebooks/phases_3_4_5_pipeline.ipynb scripts/build_phase345_notebook.py 2>/dev/null
git commit -m "feat(notebooks): unified phase 3+4+5 pipeline driver"
```

- [ ] **Step 5: End-to-end CLI smoke (local, optional)**

If Food-101 is downloaded locally and time allows, run a 1-epoch end-to-end smoke:

```bash
foodnet-train --splits-dir splits --images-root data/food-101/images \
  --output-dir runs/smoke --num-classes 101 --epochs 1 --batch-size 32 \
  --no-wandb --no-early-stop --seed 42 --device cuda
foodnet-evaluate --checkpoint runs/smoke/best.pt --splits-dir splits \
  --images-root data/food-101/images --split test \
  --output-parquet runs/smoke/test_preds.parquet --num-classes 101 --device cuda
foodnet-analyze --predictions runs/smoke/test_preds.parquet \
  --classes splits/classes.txt --output-dir runs/smoke/analysis
```

Expected: all three CLIs exit 0; `runs/smoke/analysis/dashboard.md` exists with metrics.

- [ ] **Step 6: Final commit (no-op if Step 5 wrote nothing)**

```bash
git status
# If runs/ output remains uncommitted (it shouldn't, .gitignore covers it), stash or clean it.
```

---

## Acceptance criteria (from the spec)

After Task 21, verify each:

1. **Splits CLI is deterministic and produces 800/100/100 per class.** `foodnet-split --seed 42` twice produces the same `splits_meta.json` sha256.
2. **Train CLI produces last.pt, best.pt, args.json, train_log.csv.** Smoke test from Task 14 passes.
3. **Train CLI refuses to clobber.** Test in Task 14 verifies.
4. **Evaluate CLI writes a per-example parquet with N rows × softmax + loss.** Test in Task 20 verifies.
5. **Analyze CLI writes metrics.json, confusion.png/csv, reliability.png/json, most_confused.json, dashboard.md.** Test in Task 20 verifies.
6. **Notebook runs sections 0-4 in order on Kaggle and locally.** Manual on Kaggle.
7. **`pytest tests/` passes without Food-101 mounted.** Synthetic fixture only — Step 3 of Task 21 verifies.

## Self-Review Checklist

- **Spec coverage:** All 7 acceptance criteria are mapped to tasks (1, 14, 14, 19+20, 20, 21, 21).
- **Placeholder scan:** No TBD/TODO; every step has either runnable code or an exact command.
- **Type consistency:** `SplitsDataset.__init__(csv_path, images_root, transform)` matches all callers in `train.py` and `evaluate.py`. `build_model(arch, num_classes, pretrained, drop_path_rate)` matches both train and evaluate. `EarlyStopper(patience, mode, min_delta)` matches train CLI usage. `run_inference(model, loader, num_classes, device, filepaths, out_path, amp)` matches the evaluate CLI call.
- **Frequent commits:** Each task ends with a commit; each task is independently revertable.
- **TDD:** Every code-bearing task starts with a failing test → run-to-fail → minimal impl → run-to-pass → commit.
