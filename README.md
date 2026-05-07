# swin-transformer-classification

Swin-Tiny on Food-101 — phases 3+4+5 packaged as the pip-installable `foodnet`
library plus four console scripts: `foodnet-split`, `foodnet-train`,
`foodnet-evaluate`, `foodnet-analyze`.

## Install on Kaggle

Kaggle notebooks ship with `pip` and internet access (toggle **Settings →
Internet → On**). Add a cell at the top of the notebook:

```bash
!pip install -q "git+https://github.com/hungsvdut2k2/swin-transformer-classification.git@main"
```

Pin to the feature branch while the PR is open:

```bash
!pip install -q "git+https://github.com/hungsvdut2k2/swin-transformer-classification.git@feat/phases-3-5-foodnet"
```

Or pin to an exact commit for reproducibility:

```bash
!pip install -q "git+https://github.com/hungsvdut2k2/swin-transformer-classification.git@<commit-sha>"
```

After install, the four CLI entry points are on `PATH`:

```bash
!foodnet-split --help
!foodnet-train --help
!foodnet-evaluate --help
!foodnet-analyze --help
```

### Offline / no-internet Kaggle runs

Kaggle competitions sometimes disable internet. Two options:

1. **Upload the repo as a Kaggle Dataset** (`Add data → Upload`), then in the
   notebook:
   ```bash
   !pip install -q --no-build-isolation /kaggle/input/<your-dataset>/swin-transformer-classification
   ```
2. **Build a wheel locally and upload it as a dataset:**
   ```bash
   python -m build --wheel        # produces dist/foodnet-0.1.0-py3-none-any.whl
   ```
   Upload the wheel as a Kaggle Dataset and `pip install` from
   `/kaggle/input/<dataset>/foodnet-0.1.0-py3-none-any.whl`.

> Kaggle's base image already includes `torch`, `torchvision`, `timm`,
> `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `matplotlib`, `seaborn`,
> `tqdm`, and `wandb`, so only the `foodnet` package itself is downloaded.

## Local install

```bash
git clone https://github.com/hungsvdut2k2/swin-transformer-classification.git
cd swin-transformer-classification
pip install -e ".[dev]"
pytest -q
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.2.

## CLI quick reference

```bash
# 1. Stratified 8:1:1 split → writes train.csv / val.csv / test.csv + splits_meta.json
foodnet-split \
  --images-root /kaggle/input/food-101/images \
  --output-dir  /kaggle/working/splits \
  --ratios 0.8,0.1,0.1 --seed 42

# 2. Train Swin-Tiny with LLRD + AdamW + cosine + AMP + Mixup/CutMix
foodnet-train \
  --splits-dir  /kaggle/working/splits \
  --images-root /kaggle/input/food-101/images \
  --output-dir  /kaggle/working/runs/swin_tiny \
  --epochs 30 --batch-size 64 --lr 5e-4 \
  --no-wandb                       # or set WANDB_API_KEY and drop this flag

# 3. Run inference, write per-example parquet (filepath/label/loss/p_0..p_C-1)
foodnet-evaluate \
  --checkpoint   /kaggle/working/runs/swin_tiny/best.pt \
  --splits-dir   /kaggle/working/splits \
  --images-root  /kaggle/input/food-101/images \
  --split test \
  --output-parquet /kaggle/working/preds_test.parquet

# 4. Compute metrics + dashboard (top-1/5, macro-F1, ECE, confusion, hardest examples)
foodnet-analyze \
  --predictions /kaggle/working/preds_test.parquet \
  --classes     /kaggle/working/splits/classes.txt \
  --output-dir  /kaggle/working/runs/swin_tiny/analysis
```

The end-to-end flow is also packaged as
[`notebooks/phases_3_4_5_pipeline.ipynb`](notebooks/phases_3_4_5_pipeline.ipynb)
— it auto-detects Kaggle vs. local paths in section 0 and drives all four
CLIs.

## Package layout

```
foodnet/
  splitting/   # per-class stratified 8:1:1 splitter
  data/        # SplitsDataset + timm transforms + Mixup/CutMix collator
  models/      # Swin-Tiny factory + LLRD param groups (5 layer buckets)
  training/    # train/validate loops, optimizer/scheduler/scaler, EarlyStopper, checkpoint
  eval/        # top-k, macro-F1, confusion, calibration (ECE), slices, parquet runner
  cli/         # split / train / evaluate / analyze entry points
  utils/       # seed, env-detect, W&B wrapper, args.json dump
```
