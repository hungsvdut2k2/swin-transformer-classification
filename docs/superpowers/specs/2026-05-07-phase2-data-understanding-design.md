# Phase 2 — Data Understanding Notebook (Design)

**Date:** 2026-05-07
**Plan reference:** [PLAN.md](../../../PLAN.md), Phase 2, steps 4–7
**Owner:** viethungnguyen2002@gmail.com

## Purpose

A single Jupyter notebook that completes Phase 2 of the Swin-Transformer-on-Food-101 project: acquire the dataset, characterize it, audit it for quality issues, and lock the train/val/test split. The notebook is the authoritative driver — it produces the artifacts every later phase consumes.

## Scope

**In scope:** dataset download + checksum verification, EDA (class distribution, image dimensions, per-channel pixel statistics, visual sampling for label noise), data quality audit (corrupt files, unusually small images, near-duplicates), stratified validation split.

**Out of scope:** PyTorch `Dataset` class, `DataLoader` configuration, augmentation logic, any model code, acting on the bad-files list (Phase 3 filters at training time), automatic label-noise correction.

## Runtime

- Local machine. Notebook downloads Food-101 itself on first run.
- All long-running cells are cached to disk; warm re-runs finish in under 60 seconds.

## File layout

```
notebooks/
  phase2_data_understanding.ipynb

data/
  food-101/                     # torchvision download target (gitignored, ~5 GB)
    images/<class>/<id>.jpg
    meta/

artifacts/phase2/
  eda_stats.json                # all numeric EDA outputs
  bad_files.json                # union of corrupt + tiny + near-duplicate
  phash_cache.parquet           # cached perceptual hashes (skip recompute on re-run)

splits/
  train.csv                     # columns: filepath, label_idx, label_name
  val.csv
  test.csv

figures/phase2/
  class_dist.png
  dims_hist.png
  pixel_stats.png
  samples/<class>.png           # 101 files, one grid per class
```

`.gitignore` covers `data/`, `artifacts/`, `figures/`. `splits/*.csv` is committed — small and reproducibility-critical.

## Notebook structure

```
0. Setup
   - Imports, seed=42, paths config (DATA_ROOT, ARTIFACTS_DIR, FIGURES_DIR)
   - Idempotency flags: FORCE_REBUILD = False, REGENERATE_SPLITS = False

1. Acquisition (Plan step 4)
   - torchvision.datasets.Food101(root=DATA_ROOT, split='train', download=True)
   - Same for split='test'. torchvision verifies the MD5 internally.
   - Print: dataset path, file counts, first 3 image paths.

2. Class distribution (Plan step 5a)
   - Counts per class for train and test → bar plot saved to figures/class_dist.png
   - Confirm balance (expect ~750 train + 250 test per class).

3. Image dimensions (Plan step 5b)
   - Walk file list, read width/height with PIL.Image.open(...).size (no decode).
   - Histograms of width, height, aspect ratio → figures/dims_hist.png
   - Summary stats fed into eda_stats.json.

4. Per-channel pixel stats (Plan step 5c)
   - Sample N=2000 random training images, compute per-channel mean/std at native resolution.
   - Compare to ImageNet stats (0.485/0.456/0.406, 0.229/0.224/0.225).
   - Save both to eda_stats.json. Normalization decision is surfaced — human picks in summary cell.

5. Visual sampling (Plan step 5d)
   - 10 random images per class arranged as a 2×5 grid; one PNG per class (101 PNGs total).
   - Saved to figures/samples/<class>.png.
   - No auto-flagging — for human eyeball review.

6. Data quality audit (Plan step 6)
   - Corrupt: PIL.Image.open(...).verify() on every file → corrupt_files.txt
   - Tiny: shorter side < 256 px → tiny_files.txt
   - Near-duplicates: imagehash.phash within each class only; Hamming ≤ 5 → near_dupes.csv
   - pHash matrix cached to artifacts/phase2/phash_cache.parquet
   - All three merged into bad_files.json.

7. Splits (Plan step 7)
   - Train (75,750) → 10% stratified validation → 68,175 train / 7,575 val.
   - sklearn.model_selection.train_test_split(stratify=labels, random_state=42)
   - Write splits/{train,val,test}.csv with columns: filepath, label_idx, label_name.
   - Sanity: per-class count in val within ±2 of expected 75.

8. Summary
   - One-screen recap: dataset size, balance verdict, dim distribution, normalization decision,
     audit counts, split sizes, label-noise hand-check verdict.
```

## Artifact formats

`eda_stats.json`:
```json
{
  "n_train": 75750, "n_test": 25250, "n_classes": 101,
  "class_counts": {"apple_pie": 750, "...": "..."},
  "dims": {
    "width":  {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0},
    "height": {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0},
    "aspect": {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0}
  },
  "pixel_stats_native": {"mean": [0, 0, 0], "std": [0, 0, 0], "n_sampled": 2000},
  "imagenet_stats": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
  "split_seed": 42
}
```

`bad_files.json`:
```json
{
  "corrupt": ["images/foo/123.jpg"],
  "tiny":    [{"path": "images/foo/124.jpg", "short_side": 198}],
  "near_duplicates": [
    {"a": "images/sushi/1.jpg", "b": "images/sushi/2.jpg", "phash_hamming": 3, "class": "sushi"}
  ]
}
```

Splits CSVs use **relative** `filepath` (so the data dir can be moved without breaking the splits).

## Implementation choices

**Libraries**
- `torchvision` — dataset download/extract.
- `Pillow` — image open, dimensions, integrity check via `Image.verify()`.
- `numpy`, `pandas` — stats, CSV.
- `matplotlib` — plots; saved at 150 DPI.
- `imagehash` — perceptual hashing (`phash`, 64-bit).
- `scikit-learn` — `train_test_split(stratify=...)`.
- `tqdm` — progress bars on long loops (audit, pHash).

**Constants** (declared at top of notebook)
- `SEED = 42`
- `VAL_FRACTION = 0.10`
- `PIXEL_STATS_SAMPLE = 2000`
- `TINY_SHORT_SIDE = 256`
- `PHASH_HAMMING_THRESHOLD = 5`
- `PHASH_SCOPE = "within_class"`

**Idempotency**
- Each section: `if artifact_exists and not FORCE_REBUILD: load and skip`.
- Long-running cells (audit, pHash) print a "loaded cached X" line so the user knows nothing recomputed.
- Splits are guarded by a separate `REGENERATE_SPLITS` flag — silently regenerating with a different seed would invalidate every downstream experiment.

**Performance budget** (rough, on a local Mac CPU)
- Acquisition: ~3–5 min one-time download.
- PIL dimension scan on 100k files: ~1 min.
- `Image.verify()` audit on 100k files: ~3–5 min.
- `phash` on 100k files: ~10–15 min.
- Cold full run: ~25 min. Warm re-run: under 1 min.

**Decisions left open for the human**
- "Use ImageNet normalization vs computed Food-101 stats?" — notebook prints both, user picks in the summary cell.
- "Is the visual label noise meaningful?" — per-class sample grids for human review; the notebook does not auto-flag.

## Acceptance criteria

Phase 2 is "done" when, by re-running the notebook end-to-end on a clean machine, all of these hold:

1. `data/food-101/` exists; torchvision's checksum check passed (no exception on `download=True`).
2. `splits/train.csv`, `splits/val.csv`, `splits/test.csv` exist; row counts are 68,175 / 7,575 / 25,250; per-class count in val is 75 ± 2.
3. `artifacts/phase2/eda_stats.json` exists and contains every key listed above.
4. `artifacts/phase2/bad_files.json` exists with three keys (`corrupt`, `tiny`, `near_duplicates`); each is a list (possibly empty).
5. `figures/phase2/` contains `class_dist.png`, `dims_hist.png`, `pixel_stats.png`, and a `samples/` directory with 101 PNGs.
6. The summary cell prints a one-screen recap covering: dataset size, class balance verdict, dim distribution, normalization decision, audit counts, split sizes, label-noise hand-check verdict.
7. Re-running with `FORCE_REBUILD=False` finishes in under 60 seconds.

## Risks & mitigations

- **pHash O(n²) blow-up.** Cross-class comparison on 100k images is ~5×10⁹ pairs. Mitigation: scope pHash to within-class only (~5.6×10⁷ pairs, ~3 orders of magnitude smaller). Cross-class duplicates are a different problem (label noise), addressed by visual sampling, not pHash.
- **Long cold-run time discourages re-runs.** Mitigation: aggressive disk caching of every expensive step, with a single `FORCE_REBUILD` escape hatch.
- **Accidental split regeneration.** A different seed re-orders val/train and silently invalidates every later comparison. Mitigation: separate `REGENERATE_SPLITS` flag, default `False`, prominently commented.
- **Code duplication into Phase 3.** Notebook-only logic means Phase 3 will re-implement CSV-loading and Dataset wrappers. Accepted trade-off — keeps Phase 2 simple; refactor into `src/` if the duplication becomes painful.
