# Phases 3 + 4 + 5 — `foodnet` Python Package (Design)

**Date:** 2026-05-07
**Plan reference:** [PLAN.md](../../../PLAN.md), Phases 3–5 (steps 8–16)
**Owner:** viethungnguyen2002@gmail.com

## Purpose

A single Python package, `foodnet`, that completes Phases 3, 4, and 5 of the Swin-Transformer-on-Food-101 project: the data pipeline + augmentation strategy (Phase 3), the baseline Swin-Tiny model + training loop (Phase 4), and the evaluation framework (Phase 5). The package is the authoritative driver — its CLIs produce every artifact later phases consume (checkpoints, predictions, metrics, plots).

## Scope

**In scope:** stratified 8:1:1 split CLI, `Dataset` + transforms (timm RandAugment / RandomErasing / Mixup / CutMix), Swin-Tiny baseline with LLRD + AdamW + cosine schedule + AMP, checkpointing + resume + early stopping, W&B logging, two-stage evaluation (inference once → analysis many), metrics dashboard (top-1 / top-5 / macro+micro F1 / per-class accuracy / confusion matrix / most-confused pairs / reliability + ECE / worst-K classes / hardest examples), unified notebook driver for Kaggle and local.

**Out of scope (deferred to Phase 6+):** RandAugment magnitude sweeps, class-balanced sampling, label-noise correction, TTA, EMA weights, distillation, models larger than Swin-Tiny, hosted CI, Hydra config composition.

## Runtime

- **Training:** Kaggle notebook on P100 (16 GB). Data mounted at `/kaggle/input/food-101/images`; splits at `/kaggle/working/splits/`; runs at `/kaggle/working/runs/<run_id>/`. W&B API key from Kaggle Secrets.
- **Phases 3 and 5 dev:** local CPU machine; same code paths as Kaggle; data at `data/food-101/images/`; splits at `splits/` (repo root); runs at `runs/<run_id>/` (repo root, gitignored).
- **Single unified notebook:** `notebooks/phases_3_4_5_pipeline.ipynb` drives the whole flow on either environment by detecting the env at startup and resolving paths.
- **`.gitignore` addition:** add `runs/` (and `!runs/.gitkeep`) so checkpoints and W&B-staged artifacts don't get committed locally.

## Splits decision (supersedes Phase 2 invariant)

Phase 2 locked an invariant — *"Test set is never touched until final reporting; `test.csv` mirrors the official Food-101 test list verbatim."* Phases 3+ deliberately supersede this:

- **New ratio:** 8:1:1 across all 101,000 Food-101 images, stratified per class. Per-class counts land **exactly** at 800 / 100 / 100 (1000 ÷ 0.8 / 0.1 / 0.1, no rounding).
- **Why:** uniform reproducible splits regenerable by a single CLI from any environment (Kaggle vs local). Single source of truth that travels with the package, not the previous phase.
- **Trade-off accepted:** PLAN.md's ≥88% top-1 target was published against Food-101's official 75,750 / 25,250 split. Our final number is no longer apples-to-apples with that literature. We rebaseline against the 8:1:1 test set when the first run completes.
- **Phase 2 splits/*.csv** stays committed but becomes stale; the `splits/` directory now contains *whatever the package CLI most recently wrote*. The committed test/val files are de facto archive only.

## File layout

```
foodnet/
  __init__.py
  splitting/
    __init__.py
    splitter.py            # per-class shuffle-and-slice; deterministic given seed
  data/
    __init__.py
    splits.py              # load_split(name, splits_dir, images_root) -> records
    dataset.py             # SplitsDataset; PIL-error log-once + resample
    transforms.py          # timm.data.create_transform recipes (train + eval)
    collate.py             # timm.data.Mixup wrapper
  models/
    __init__.py
    factory.py             # build_model() via timm.create_model
    param_groups.py        # llrd_param_groups(model, base_lr, decay)
  training/
    __init__.py
    optim.py               # AdamW + timm CosineLRScheduler + warmup
    loop.py                # train_one_epoch, validate, save_checkpoint, EarlyStopper
    amp.py                 # GradScaler + autocast(fp16) helpers
  eval/
    __init__.py
    runner.py              # run_inference -> predictions DataFrame
    metrics.py             # top-k, F1, per-class accuracy
    confusion.py           # confusion matrix + most_confused_pairs + heatmap
    calibration.py         # ECE + reliability diagram
    slices.py              # worst_classes, hardest_correct/incorrect, class_pair_examples
  cli/
    __init__.py
    split.py               # python -m foodnet.cli.split
    train.py               # python -m foodnet.cli.train
    evaluate.py            # python -m foodnet.cli.evaluate
    analyze.py             # python -m foodnet.cli.analyze
  utils/
    __init__.py
    paths.py               # repo/Kaggle env detection + path resolution
    seed.py                # set_seed: python/np/torch/cuda
    wandb_logger.py        # init/log/log_image_grid/finish; --no-wandb is no-op
    config_dump.py         # argparse Namespace -> JSON next to checkpoint

notebooks/
  phases_3_4_5_pipeline.ipynb   # the only Phase 3+4+5 driver

tests/
  conftest.py              # synthetic 32x32 PNG fixture; tiny class-name list
  test_splitter.py
  test_dataset.py
  test_transforms.py
  test_param_groups.py
  test_loop_smoke.py
  test_checkpointing.py
  test_amp.py
  test_metrics.py
  test_calibration.py
  test_confusion.py
  test_slices.py

pyproject.toml             # foodnet pip-installable; entry-point scripts
requirements.txt           # appended: timm, wandb, torchmetrics, pytest
```

`pyproject.toml` declares `[project.scripts]` so the CLIs are also callable as plain commands (`foodnet-split`, `foodnet-train`, `foodnet-evaluate`, `foodnet-analyze`).

## Phase 3 — data pipeline

### Splits CLI

```
python -m foodnet.cli.split \
    --images-root /kaggle/input/food-101/images \
    --out-dir    /kaggle/working/splits \
    --ratios     0.8 0.1 0.1 \
    --seed       42 \
    [--force]
```

1. Walks `images_root/<class>/*.jpg`; asserts 101,000 records, 101 classes.
2. Per-class shuffle (`np.random.default_rng(seed).permutation`) and slice into 800 / 100 / 100. Class-stratified by construction.
3. Writes `train.csv`, `val.csv`, `test.csv` with columns `filepath, label_idx, label_name`. `filepath` is **relative to `images_root`** (e.g. `apple_pie/1234.jpg`) — environment-agnostic. The dataset prepends `images_root` at load time.
4. Writes `splits_meta.json` with `{seed, ratios, sha256(train.csv), sha256(val.csv), sha256(test.csv), generated_at}`.
5. Writes `class_names.json` (sorted list of 101 class names) — eval and analyze read this rather than scanning the image directory.
6. Idempotent: skips if all three CSVs and `splits_meta.json` exist and `--force` is not passed.

### Dataset

```python
class SplitsDataset(Dataset):
    def __init__(self, records, images_root, transform, *, on_error="resample"):
        ...
    def __getitem__(self, i):
        r = self.records[i]
        try:
            img = Image.open(self.images_root / r.filepath).convert("RGB")
        except Exception as e:
            log_once(r.filepath, e)             # process-local set; warn at most once per path
            if self.on_error == "resample":
                return self[(i + 1) % len(self)]
            raise
        return self.transform(img), r.label_idx
```

- **No filtering** of `bad_files.json`. At training-CLI startup, a single warning prints the counts (corrupt / tiny / near-dupe) — informational only.
- **PIL-error resampling** walks forward (`(i + 1) % len(self)`) until success. Worker-safe (no shared mutation).

### Transforms (built via `timm.data.create_transform`)

- **Train:** `input_size=224`, `is_training=True`, `auto_augment="rand-m9-mstd0.5-inc1"`, `interpolation="bicubic"`, `re_prob=0.25`, `re_mode="pixel"`, `re_count=1`, mean/std = ImageNet stats.
- **Eval:** resize shorter side to 256, center crop 224, mean/std = ImageNet stats.
- **Normalization decision (deferred from Phase 2):** **ImageNet stats**, not Food-101 native stats. Reasoning: pretrained weights expect ImageNet input distribution; the ≈0.06 per-channel difference in means is small enough that pretraining advantage dominates.

### Mixup / CutMix collator

```python
mixup_fn = timm.data.Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0,
    prob=1.0, switch_prob=0.5, mode="batch",
    label_smoothing=0.1, num_classes=101,
)
```

Applied **after** the dataloader returns a batch, inside the train step. Loss = `timm.loss.SoftTargetCrossEntropy` (handles soft labels; identical to CE on hard labels). Eval never uses mixup.

### DataLoader knobs (CLI flags)

`--batch-size 64`, `--num-workers 4`, `--pin-memory` (auto-detect via `torch.cuda.is_available()`), `--persistent-workers` (default true).

## Phase 4 — baseline model + training

### Model

```python
def build_model(arch="swin_tiny_patch4_window7_224",
                num_classes=101, pretrained=True, drop_path_rate=0.2):
    return timm.create_model(arch, pretrained=pretrained,
                             num_classes=num_classes,
                             drop_path_rate=drop_path_rate)
```

`timm.create_model` replaces the 1000-class head automatically when `num_classes != 1000`. `drop_path_rate=0.2` is the Swin-Tiny paper default — small stochastic-depth regularizer.

### Layer-wise LR decay (LLRD)

`foodnet.models.param_groups.llrd_param_groups(model, base_lr, decay=0.75)` walks `model.named_parameters()`, groups by Swin stage (patch_embed, stages.0..3, head), assigns LR = `base_lr * decay^(num_layers - layer_idx)`. Returns a `param_groups` list compatible with `torch.optim.AdamW`.

### Optimizer + scheduler

- `AdamW(param_groups, weight_decay=0.05, betas=(0.9, 0.999))`
- `timm.scheduler.CosineLRScheduler(optimizer, t_initial=epochs, lr_min=1e-6, warmup_t=warmup_epochs, warmup_lr_init=1e-6)`. Stepped per epoch.

### Training step

- AMP: `torch.cuda.amp.GradScaler` + `autocast(dtype=torch.float16)`. P100 has no Tensor Cores → fp16 only; bf16 falls back to fp32.
- Gradient clipping: `clip_grad_norm_(params, 1.0)` after `scaler.unscale_(optimizer)`.
- Label smoothing 0.1 is folded into `Mixup` (don't double-apply).
- EMA: not enabled in baseline; deferred to Phase 6.

### Train loop (`foodnet/training/loop.py`)

```python
def train_one_epoch(model, loader, optimizer, scaler, mixup_fn, loss_fn,
                    device, *, epoch, log_every, wandb_run) -> dict[str, float]
def validate(model, loader, device, *, epoch, wandb_run) -> dict[str, float]
def save_checkpoint(model, optimizer, scheduler, scaler, *, epoch, best_top1,
                    args_dict, splits_meta, output_dir, name) -> Path
```

`train_one_epoch` returns `{"train/loss": ..., "train/lr_head": ...}`; logs every `log_every` steps.
`validate` returns `{"val/loss", "val/top1", "val/top5"}`; logs once per epoch.

### Checkpointing

- After each `validate`, compare `val/top1` to `best_top1`.
- Save `output_dir/<run_id>/checkpoints/best.pt` only on improvement; `last.pt` every epoch.
- Each checkpoint contains: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`, `best_top1`, `args` (effective CLI namespace as dict), `splits_meta` (the splits metadata read at training time).
- W&B: log `best.pt` as a **summary artifact only at the end of training**. `last.pt` stays local.

### Resume

`--resume <path>` loads model + optimizer + scheduler + scaler + epoch. If `--resume` is omitted but `output_dir/<run_id>/checkpoints/last.pt` exists, the CLI exits with an error asking for an explicit `--resume` or a fresh `--run-id`. Prevents accidental clobber.

### Early stopping

```python
class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 1e-3, mode: str = "max")
    def step(self, value: float) -> tuple[bool, bool]   # (improved, should_stop)
```

- **Tracked metric:** `val/top1`, mode=max.
- **Defaults:** patience=8 epochs, min_delta=1e-3 (0.1 percentage point).
- **Trigger behaviour:** logs `early-stop triggered at epoch N (best val/top1=X at epoch M)`; calls `wandb.finish()`; exits cleanly. `best.pt` preserved; `last.pt` reflects final trained epoch.
- **Patience rationale:** the cosine cool-down's last ~25% of epochs gives small slow improvements; patience=8 rides out cool-down without wasting compute when training has truly plateaued.

### Train CLI

```
python -m foodnet.cli.train \
    --splits-dir       /kaggle/working/splits \
    --images-root      /kaggle/input/food-101/images \
    --output-dir       /kaggle/working/runs \
    --run-id           baseline-001 \
    --arch             swin_tiny_patch4_window7_224 \
    --pretrained \
    --drop-path-rate   0.2 \
    --epochs           30 \
    --batch-size       64 \
    --num-workers      4 \
    --lr               1e-3 \
    --weight-decay     0.05 \
    --layer-decay      0.75 \
    --warmup-epochs    3 \
    --label-smoothing  0.1 \
    --mixup            0.8 \
    --cutmix           1.0 \
    --mixup-prob       1.0 \
    --mixup-switch-prob 0.5 \
    --rand-aug         rand-m9-mstd0.5-inc1 \
    --random-erasing   0.25 \
    --grad-clip        1.0 \
    --amp \
    --early-stop-patience  8 \
    --early-stop-min-delta 1e-3 \
    --seed             42 \
    --log-every        50 \
    --wandb-project    food101-swin \
    --wandb-entity     <user> \
    [--no-wandb] [--no-early-stop] [--resume <path>]
```

Args grouped (`data/`, `model/`, `optim/`, `aug/`, `runtime/`) for readable `--help`. Effective namespace dumped to `output_dir/<run_id>/args.json` before the first batch.

### Compute budget

Swin-Tiny @ 224, batch 64, fp16, ~80,800 train images → ~1,260 steps/epoch → ~6–8 min/epoch on P100. **30 epochs ≈ 3.5 hr** total — comfortably inside one Kaggle 9-hr session, with room for early stopping to cut compute further when val plateaus.

## Phase 5 — evaluation framework

### Two-stage flow

1. **Inference once** — `evaluate` CLI runs the model over a split, dumps `predictions.parquet` (per-example softmax + loss).
2. **Analysis many** — `analyze` CLI reads that parquet and produces metrics + plots. Re-renders cheaply (~seconds).

### Inference (`foodnet/eval/runner.py`)

```python
def run_inference(model, loader, device, *, num_classes) -> pd.DataFrame
```

Returns a row per example with `filepath, label_idx, top1_pred, top5_preds, loss, prob_0..prob_100` (the full softmax). Parquet ≈ 25 MB for 10,100 test rows × 101 classes.

### Metrics

- `top_k_accuracy`, `macro_f1`, `micro_f1` via `torchmetrics.functional`.
- `per_class_accuracy` via `np.bincount`.
- `metrics.json` summary with `top1, top5, macro_f1, micro_f1, loss_mean, per_class_acc_{mean,std}, ece, checkpoint_sha256, splits_meta_sha256, args_used`.

### Confusion + most-confused pairs

`confusion.py` produces a `[101, 101]` matrix and ranks off-diagonal cells two ways: by raw count and by `count / row_total`. Top-K under both rankings printed side-by-side. Heatmap (row-normalized) saved as `confusion_matrix.png`.

### Calibration

15-bin ECE on predicted-class confidence; reliability diagram with bin counts as a histogram beneath. Answers the PLAN.md question *"does the model know when it's wrong?"*

### Error slicing

- `worst_classes(per_class_acc, k=10)` — bottom-K classes by accuracy.
- `hardest_correct(predictions_df, k=10)` — high-loss but `top1_pred == label_idx` (PLAN.md step 15).
- `hardest_incorrect(predictions_df, k=10)` — high-loss and `top1_pred != label_idx` (Phase-6 label-noise candidates).
- `class_pair_examples(predictions_df, true, pred, k=10)` — drill into a confused pair for the dashboard.

### Eval CLIs

```
python -m foodnet.cli.evaluate \
    --ckpt        /kaggle/working/runs/<run_id>/checkpoints/best.pt \
    --split       test \
    --splits-dir  /kaggle/working/splits \
    --images-root /kaggle/input/food-101/images \
    --output-dir  /kaggle/working/runs/<run_id>/eval \
    --batch-size  128 \
    [--no-amp]

python -m foodnet.cli.analyze \
    --predictions /kaggle/working/runs/<run_id>/eval/predictions.parquet \
    --class-names /kaggle/working/splits/class_names.json \
    --output-dir  /kaggle/working/runs/<run_id>/eval/analysis \
    [--top-k-confused 20] [--n-bins 15] [--worst-k 10]
```

`evaluate` writes `predictions.parquet` + `metrics.json`. `analyze` writes `confusion_matrix.png`, `most_confused_pairs.csv`, `reliability.png`, `ece.txt`, `worst_classes.csv` + `worst_classes.png`, `hardest_correct.csv`, `hardest_incorrect.csv`, `dashboard.md`.

`evaluate` also logs the metrics JSON, confusion heatmap, reliability diagram, and a 4×4 hardest-incorrect grid to W&B as a summary artifact under the same `run_id`.

## Unified notebook

```
notebooks/phases_3_4_5_pipeline.ipynb

  0. Setup
     - Detect environment (Kaggle vs local) -> set IMAGES_ROOT, OUTPUT_ROOT
     - Clone repo (Kaggle only) + pip install -e .
     - Read W&B API key from Kaggle Secrets (Kaggle only)
     - Print resolved env: GPU name, image count, packages

  1. Phase 3 - data pipeline
     - !python -m foodnet.cli.split --ratios 0.8 0.1 0.1 --seed 42 ...
     - Verify splits exist; assert per-class counts exactly 800/100/100
     - Visual: 4x4 grid of augmented samples
     - Visual: 4x4 grid of mixup'd samples
     - Print bad_files.json startup warning

  2. Phase 4 - train baseline
     - If best.pt exists and not RETRAIN: skip; print "loaded existing checkpoint"
     - Else: !python -m foodnet.cli.train ... (Swin-Tiny baseline; early stop @ patience=8)
     - Print W&B URL

  3. Phase 5 - evaluation dashboard
     - !python -m foodnet.cli.evaluate --ckpt <best.pt> --split test ...
     - !python -m foodnet.cli.analyze --predictions <...>
     - Render dashboard.md inline (IPython.display.Markdown)
     - Display confusion heatmap, reliability diagram, worst-classes bar
     - Display 16 hardest-incorrect images as inline grid

  4. (Stretch) Repeat section 3 with --split val for unbiased model-selection numbers
```

Section 0 detects environment by checking `Path("/kaggle/input").exists()`. Falls back to `data/food-101/images/` locally.

## Acceptance criteria

Phases 3+4+5 are "done" when, on a clean Kaggle session, all of these hold:

1. `python -m foodnet.cli.split --ratios 0.8 0.1 0.1 --seed 42` produces three CSVs with **exactly** 800/100/100 per class; `splits_meta.json` and `class_names.json` exist.
2. `SplitsDataset` returns a tensor of shape `[3, 224, 224]`, dtype float32, normalized with ImageNet stats. Bad-file resampling logs each offending path at most once.
3. `python -m foodnet.cli.train` runs end-to-end on Kaggle P100 in **under 6 hours** (or stops early when `val/top1` hasn't improved by ≥1e-3 for 8 epochs); saves `best.pt` and `last.pt`; logs to W&B; checkpoint includes `args.json` and `splits_meta`.
4. `python -m foodnet.cli.evaluate --split test` produces `predictions.parquet` and `metrics.json`; baseline `top1` ≥ **80%** (loose floor — the 88% PLAN.md target was on the official Food-101 test split; we re-baseline once we see the actual number on the 8:1:1 test set).
5. `python -m foodnet.cli.analyze` produces all listed plots/CSVs and a `dashboard.md`.
6. Re-running the unified notebook end-to-end on the same machine reproduces identical splits and (modulo training stochasticity within seed) identical eval numbers from the same `best.pt`.
7. `pytest tests/` passes without requiring Food-101 to be present (synthetic-data fixtures only).

## Risks & mitigations

- **W&B API key on Kaggle.** Use Kaggle Secrets (`WANDB_API_KEY`). If unset, `--no-wandb` runs offline; the `WandBLogger` is a no-op when disabled.
- **Kaggle session timeouts** (9-hr wall clock, possibly flaky network). Mitigations: `last.pt` saved every epoch; `--resume` works on a partial run; early stopping caps wasted compute.
- **AMP-related NaN losses** (a real risk on a first Phase-4 run). Mitigations: AMP opt-in via `--amp`; `train_one_epoch` asserts loss is finite each step and exits with a useful message if not.
- **Splits drift between CLI runs.** `seed=42` hardcoded as default; `splits_meta.json` records sha256s; the train CLI rejects checkpoints whose `splits_meta` doesn't match the current splits dir.
- **Phase 2 invariant break** ("test = official Food-101 verbatim"). Explicit decision-note entry; flag the Phase 2 invariant as superseded; PLAN.md's 88% target re-baselined against the new 8:1:1 test set.
- **`/kaggle/working/` 20 GB cap.** Swin-Tiny ≈ 110 MB per checkpoint; predictions parquet ≈ 25 MB; figures < 5 MB total. Comfortably under cap. EMA + multi-checkpoint history would burst it; we don't enable them for the baseline.

## Implementation order

(Full step-by-step plan comes from the writing-plans skill.)

1. Repo scaffolding: `pyproject.toml`, `foodnet/` skeleton, `tests/conftest.py`, dependency additions.
2. `foodnet.splitting` + `foodnet.cli.split` + tests.
3. `foodnet.data` (splits, dataset, transforms, collate) + tests.
4. `foodnet.models` (factory, param_groups) + tests.
5. `foodnet.training` (optim, scheduler, loop, AMP, checkpointing, EarlyStopper) + smoke tests.
6. `foodnet.cli.train` end-to-end with W&B + tests.
7. `foodnet.eval` (metrics, confusion, calibration, slices, runner) + tests.
8. `foodnet.cli.evaluate` + `foodnet.cli.analyze` + tests.
9. Unified notebook (`phases_3_4_5_pipeline.ipynb`) — wires the CLIs end-to-end.
10. Acceptance-criteria validation (the 7 checks above).
