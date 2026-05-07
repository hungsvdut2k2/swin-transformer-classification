# Phase 2 — Data Understanding Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single Jupyter notebook that completes Phase 2 of the Food-101 / Swin-Transformer project — acquire the dataset, characterize it, audit quality, lock the train/val/test split — emitting all artifacts every later phase consumes.

**Architecture:** One notebook (`notebooks/phase2_data_understanding.ipynb`) runs locally end-to-end. Each section writes its artifact to disk and ends with an assertion cell that verifies the artifact's shape. Long-running cells cache to disk and skip recompute when the artifact already exists. The notebook is built and modified in place using `nbformat` Python snippets (no jupytext, no build script — the `.ipynb` is the source of truth).

**Tech Stack:** Python 3.11+, Jupyter, `nbformat`, `torchvision`, `Pillow`, `numpy`, `pandas`, `matplotlib`, `imagehash`, `scikit-learn`, `tqdm`.

**Spec:** [docs/superpowers/specs/2026-05-07-phase2-data-understanding-design.md](../specs/2026-05-07-phase2-data-understanding-design.md)
**Decision note:** [docs/superpowers/specs/2026-05-07-phase2-data-understanding-decision.md](../specs/2026-05-07-phase2-data-understanding-decision.md)

---

## TDD Adaptation for Notebooks

Pure pytest-style TDD doesn't map cleanly to notebook development. Adapted pattern per task:

1. **Append the assertion cell first** (the "test"). Run the notebook. Observe failure (NameError, FileNotFoundError, AssertionError) — this confirms the assertion can fail.
2. **Append the implementation cells before the assertion**. Run the notebook again. Observe pass.
3. **Commit.**

The "run the notebook" step uses `jupyter nbconvert --to notebook --execute --inplace` so it's mechanical and CI-friendly. After the first cold run (acquisition + audit + pHash, ~25 min), warm re-runs finish in under 60 seconds because every expensive cell caches its result and skips recompute.

---

## Task 1: Repo scaffolding & dependency lock

**Files:**
- Create: `requirements.txt`
- Modify: `.gitignore`
- Create: `notebooks/phase2_data_understanding.ipynb` (empty scaffold)
- Create: `data/.gitkeep`, `artifacts/phase2/.gitkeep`, `figures/phase2/.gitkeep`, `splits/.gitkeep`

- [ ] **Step 1: Create `requirements.txt`**

```text
torch>=2.2
torchvision>=0.17
Pillow>=10.0
numpy>=1.26
pandas>=2.1
matplotlib>=3.8
imagehash>=4.3
scikit-learn>=1.4
tqdm>=4.66
jupyter>=1.0
nbformat>=5.9
pyarrow>=15.0
```

- [ ] **Step 2: Append data/artifact paths to `.gitignore`**

Append these lines to the existing `.gitignore`:

```text
# Phase 2 artifacts
data/food-101/
artifacts/
figures/
!**/.gitkeep
```

- [ ] **Step 3: Create directory scaffolding**

```bash
mkdir -p notebooks data artifacts/phase2 figures/phase2/samples splits
touch data/.gitkeep artifacts/phase2/.gitkeep figures/phase2/.gitkeep splits/.gitkeep
```

- [ ] **Step 4: Create the empty notebook scaffold**

Run this from the repo root:

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell(
        "# Phase 2 — Data Understanding\n\n"
        "Companion to PLAN.md Phase 2. Acquires Food-101, runs EDA, audits quality, "
        "and locks train/val/test splits. See "
        "`docs/superpowers/specs/2026-05-07-phase2-data-understanding-design.md`."
    )
]
nb["metadata"] = {
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
print("created empty notebook")
PY
```

Expected output: `created empty notebook`

- [ ] **Step 5: Install dependencies and verify Jupyter can execute the empty notebook**

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb
```

Expected: `[NbConvertApp] Writing ... bytes to notebooks/phase2_data_understanding.ipynb` with no errors.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore notebooks/ data/.gitkeep artifacts/ figures/ splits/.gitkeep
git commit -m "chore: scaffold phase 2 notebook + dependencies"
```

---

## Task 2: Section 0 — Setup, paths, constants, helpers

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

Run from the repo root:

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: setup primitives are defined and paths exist\n"
    "for name in ['SEED', 'VAL_FRACTION', 'PIXEL_STATS_SAMPLE', 'TINY_SHORT_SIDE',\n"
    "             'PHASH_HAMMING_THRESHOLD', 'PHASH_SCOPE', 'FORCE_REBUILD',\n"
    "             'REGENERATE_SPLITS', 'DATA_ROOT', 'ARTIFACTS_DIR', 'FIGURES_DIR',\n"
    "             'SPLITS_DIR', 'EDA_STATS_PATH', 'BAD_FILES_PATH', 'update_stats']:\n"
    "    assert name in dir(), f'missing setup symbol: {name}'\n"
    "for d in [DATA_ROOT, ARTIFACTS_DIR, FIGURES_DIR, SPLITS_DIR]:\n"
    "    assert d.exists(), f'missing dir: {d}'\n"
    "print('setup OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: execution fails with `NameError: name 'SEED' is not defined`. This confirms the assertion can fail.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()  # remove assertion temporarily

cells_to_add = [
    nbf.v4.new_markdown_cell("## 0. Setup"),
    nbf.v4.new_code_cell(
        "import json\n"
        "import random\n"
        "from pathlib import Path\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n"
        "from tqdm.auto import tqdm\n"
    ),
    nbf.v4.new_code_cell(
        "# Constants — change deliberately, every downstream phase depends on these.\n"
        "SEED = 42\n"
        "VAL_FRACTION = 0.10\n"
        "PIXEL_STATS_SAMPLE = 2000\n"
        "TINY_SHORT_SIDE = 256\n"
        "PHASH_HAMMING_THRESHOLD = 5\n"
        "PHASH_SCOPE = 'within_class'\n"
        "\n"
        "FORCE_REBUILD = False        # re-compute cached EDA / audit cells\n"
        "REGENERATE_SPLITS = False    # re-shuffle splits (DANGEROUS — invalidates downstream comparisons)\n"
        "\n"
        "random.seed(SEED)\n"
        "np.random.seed(SEED)\n"
    ),
    nbf.v4.new_code_cell(
        "# Paths — relative to repo root. Notebook is launched from repo root.\n"
        "REPO_ROOT = Path.cwd()\n"
        "if REPO_ROOT.name == 'notebooks':\n"
        "    REPO_ROOT = REPO_ROOT.parent\n"
        "DATA_ROOT = REPO_ROOT / 'data'\n"
        "ARTIFACTS_DIR = REPO_ROOT / 'artifacts' / 'phase2'\n"
        "FIGURES_DIR = REPO_ROOT / 'figures' / 'phase2'\n"
        "SPLITS_DIR = REPO_ROOT / 'splits'\n"
        "\n"
        "for d in [DATA_ROOT, ARTIFACTS_DIR, FIGURES_DIR, FIGURES_DIR / 'samples', SPLITS_DIR]:\n"
        "    d.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "EDA_STATS_PATH = ARTIFACTS_DIR / 'eda_stats.json'\n"
        "BAD_FILES_PATH = ARTIFACTS_DIR / 'bad_files.json'\n"
        "PHASH_CACHE_PATH = ARTIFACTS_DIR / 'phash_cache.parquet'\n"
        "AUDIT_CACHE_PATH = ARTIFACTS_DIR / 'audit_cache.parquet'\n"
        "\n"
        "print(f'REPO_ROOT = {REPO_ROOT}')\n"
    ),
    nbf.v4.new_code_cell(
        "def update_stats(key, value):\n"
        "    \"\"\"Read eda_stats.json, set one key, write back. Idempotent.\"\"\"\n"
        "    stats = {}\n"
        "    if EDA_STATS_PATH.exists():\n"
        "        stats = json.loads(EDA_STATS_PATH.read_text())\n"
        "    stats[key] = value\n"
        "    EDA_STATS_PATH.write_text(json.dumps(stats, indent=2, sort_keys=True))\n"
        "    return stats\n"
        "\n"
        "# Seed eda_stats.json with the constants we already know.\n"
        "update_stats('split_seed', SEED)\n"
        "update_stats('imagenet_stats', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook and verify the assertion passes**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -3
```

Expected: no errors. The final cell prints `setup OK`.

- [ ] **Step 4: Verify `eda_stats.json` got the seed values**

```bash
cat artifacts/phase2/eda_stats.json
```

Expected: JSON with keys `imagenet_stats` and `split_seed: 42`.

- [ ] **Step 5: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 0 — setup, paths, constants, helpers"
```

---

## Task 3: Section 1 — Acquisition

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: dataset acquired and inventoried\n"
    "assert (DATA_ROOT / 'food-101' / 'images').exists(), 'food-101 images dir missing'\n"
    "assert len(train_files) == 75750, f'expected 75750 train files, got {len(train_files)}'\n"
    "assert len(test_files) == 25250, f'expected 25250 test files, got {len(test_files)}'\n"
    "assert len(class_names) == 101, f'expected 101 classes, got {len(class_names)}'\n"
    "print('acquisition OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: `NameError: name 'train_files' is not defined`.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 1. Acquisition\n"
        "Downloads Food-101 from the official ETH Zurich URL via torchvision. "
        "torchvision verifies the MD5 internally; if the checksum fails it raises."
    ),
    nbf.v4.new_code_cell(
        "from torchvision.datasets import Food101\n"
        "\n"
        "# torchvision caches under DATA_ROOT/food-101/. download=True is a no-op if already present.\n"
        "train_ds = Food101(root=str(DATA_ROOT), split='train', download=True)\n"
        "test_ds  = Food101(root=str(DATA_ROOT), split='test',  download=True)\n"
        "\n"
        "class_names = sorted(train_ds.classes)\n"
        "label_to_idx = {c: i for i, c in enumerate(class_names)}\n"
    ),
    nbf.v4.new_code_cell(
        "# Build flat path lists. Use repo-relative paths so split CSVs survive moving the data dir.\n"
        "IMAGES_DIR = DATA_ROOT / 'food-101' / 'images'\n"
        "\n"
        "def _read_meta(split):\n"
        "    meta_file = DATA_ROOT / 'food-101' / 'meta' / f'{split}.txt'\n"
        "    return [line.strip() for line in meta_file.read_text().splitlines() if line.strip()]\n"
        "\n"
        "def _to_records(meta_lines):\n"
        "    out = []\n"
        "    for line in meta_lines:\n"
        "        cls, fname = line.split('/')\n"
        "        rel = f'data/food-101/images/{cls}/{fname}.jpg'\n"
        "        out.append({'filepath': rel, 'label_name': cls, 'label_idx': label_to_idx[cls]})\n"
        "    return out\n"
        "\n"
        "train_files = _to_records(_read_meta('train'))\n"
        "test_files  = _to_records(_read_meta('test'))\n"
        "\n"
        "print(f'train={len(train_files):,}  test={len(test_files):,}  classes={len(class_names)}')\n"
        "print('first 3 train paths:')\n"
        "for r in train_files[:3]:\n"
        "    print(' ', r['filepath'])\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook (cold run — first download takes ~5 min on a typical home connection)**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb --ExecutePreprocessor.timeout=1800 2>&1 | tail -5
```

Expected: download completes, assertion prints `acquisition OK`.

- [ ] **Step 4: Verify the data on disk**

```bash
ls data/food-101/images | wc -l         # should print 101
ls data/food-101/images/apple_pie | wc -l   # should print 1000
```

- [ ] **Step 5: Commit (notebook only — data/ is gitignored)**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 1 — acquire & inventory Food-101"
```

---

## Task 4: Section 2 — Class distribution

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: class distribution computed and persisted\n"
    "import json as _json\n"
    "stats = _json.loads(EDA_STATS_PATH.read_text())\n"
    "assert 'class_counts' in stats and 'n_train' in stats and 'n_test' in stats and 'n_classes' in stats\n"
    "assert stats['n_train'] == 75750\n"
    "assert stats['n_test']  == 25250\n"
    "assert stats['n_classes'] == 101\n"
    "assert len(stats['class_counts']) == 101\n"
    "assert (FIGURES_DIR / 'class_dist.png').exists()\n"
    "print('class distribution OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails on missing `class_counts` key.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 2. Class Distribution\n"
        "Food-101 ships balanced (750 train / 250 test per class). We confirm rather than assume."
    ),
    nbf.v4.new_code_cell(
        "from collections import Counter\n"
        "\n"
        "train_counts = Counter(r['label_name'] for r in train_files)\n"
        "test_counts  = Counter(r['label_name'] for r in test_files)\n"
        "\n"
        "balance_ok = (set(train_counts.values()) == {750}) and (set(test_counts.values()) == {250})\n"
        "print(f'balanced exactly 750/250 per class: {balance_ok}')\n"
    ),
    nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)\n"
        "ordered = sorted(class_names)\n"
        "axes[0].bar(range(len(ordered)), [train_counts[c] for c in ordered])\n"
        "axes[0].set_title('train images per class')\n"
        "axes[0].set_ylabel('count')\n"
        "axes[1].bar(range(len(ordered)), [test_counts[c] for c in ordered])\n"
        "axes[1].set_title('test images per class')\n"
        "axes[1].set_ylabel('count')\n"
        "axes[1].set_xlabel('class index')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / 'class_dist.png', dpi=150)\n"
        "plt.show()\n"
    ),
    nbf.v4.new_code_cell(
        "update_stats('n_train', len(train_files))\n"
        "update_stats('n_test', len(test_files))\n"
        "update_stats('n_classes', len(class_names))\n"
        "update_stats('class_counts', dict(train_counts))\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook and verify**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -3
```

Expected: `class distribution OK`. `figures/phase2/class_dist.png` exists.

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 2 — class distribution"
```

---

## Task 5: Section 3 — Image dimensions

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: dimension stats computed and persisted\n"
    "import json as _json\n"
    "stats = _json.loads(EDA_STATS_PATH.read_text())\n"
    "assert 'dims' in stats\n"
    "for k in ('width', 'height', 'aspect'):\n"
    "    assert k in stats['dims']\n"
    "    for stat in ('min', 'max', 'mean', 'p50', 'p95'):\n"
    "        assert stat in stats['dims'][k], f'missing dims.{k}.{stat}'\n"
    "assert (FIGURES_DIR / 'dims_hist.png').exists()\n"
    "assert (ARTIFACTS_DIR / 'dims_cache.parquet').exists()\n"
    "print('dimensions OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails on missing `dims` key.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 3. Image Dimensions\n"
        "Reads width/height (without decoding pixels) for every train+test image. "
        "Informs the resize strategy in Phase 3."
    ),
    nbf.v4.new_code_cell(
        "DIMS_CACHE = ARTIFACTS_DIR / 'dims_cache.parquet'\n"
        "\n"
        "if DIMS_CACHE.exists() and not FORCE_REBUILD:\n"
        "    dims_df = pd.read_parquet(DIMS_CACHE)\n"
        "    print(f'loaded cached dimensions for {len(dims_df):,} images — set FORCE_REBUILD=True to recompute')\n"
        "else:\n"
        "    rows = []\n"
        "    for r in tqdm(train_files + test_files, desc='reading dims'):\n"
        "        path = REPO_ROOT / r['filepath']\n"
        "        try:\n"
        "            with Image.open(path) as im:\n"
        "                w, h = im.size\n"
        "        except Exception:\n"
        "            w, h = -1, -1   # corrupt — picked up properly in section 6a\n"
        "        rows.append({'filepath': r['filepath'], 'width': w, 'height': h})\n"
        "    dims_df = pd.DataFrame(rows)\n"
        "    dims_df['aspect'] = dims_df['width'] / dims_df['height']\n"
        "    dims_df.to_parquet(DIMS_CACHE)\n"
        "    print(f'computed and cached dimensions for {len(dims_df):,} images')\n"
    ),
    nbf.v4.new_code_cell(
        "valid = dims_df[(dims_df['width'] > 0) & (dims_df['height'] > 0)]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "axes[0].hist(valid['width'], bins=60); axes[0].set_title('width (px)')\n"
        "axes[1].hist(valid['height'], bins=60); axes[1].set_title('height (px)')\n"
        "axes[2].hist(valid['aspect'].clip(0, 4), bins=60); axes[2].set_title('aspect ratio (clipped to [0,4])')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / 'dims_hist.png', dpi=150)\n"
        "plt.show()\n"
    ),
    nbf.v4.new_code_cell(
        "def _summary(s):\n"
        "    return {\n"
        "        'min':  float(s.min()), 'max':  float(s.max()),\n"
        "        'mean': float(s.mean()),\n"
        "        'p50':  float(s.quantile(0.50)), 'p95': float(s.quantile(0.95)),\n"
        "    }\n"
        "\n"
        "update_stats('dims', {\n"
        "    'width':  _summary(valid['width']),\n"
        "    'height': _summary(valid['height']),\n"
        "    'aspect': _summary(valid['aspect']),\n"
        "})\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -3
```

Expected: `dimensions OK`. First run takes ~1 min for the dim scan.

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 3 — image dimension distribution"
```

---

## Task 6: Section 4 — Per-channel pixel statistics

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: pixel stats computed and persisted\n"
    "import json as _json\n"
    "stats = _json.loads(EDA_STATS_PATH.read_text())\n"
    "ps = stats.get('pixel_stats_native')\n"
    "assert ps is not None and 'mean' in ps and 'std' in ps and 'n_sampled' in ps\n"
    "assert len(ps['mean']) == 3 and len(ps['std']) == 3\n"
    "assert ps['n_sampled'] == PIXEL_STATS_SAMPLE\n"
    "assert all(0.0 < m < 1.0 for m in ps['mean']), f'pixel means look wrong: {ps[\"mean\"]}'\n"
    "assert (FIGURES_DIR / 'pixel_stats.png').exists()\n"
    "print('pixel stats OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails on missing `pixel_stats_native`.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 4. Per-Channel Pixel Statistics\n"
        "Sample 2000 train images, compute RGB mean/std at native resolution. "
        "Compare to ImageNet stats — if close, we use ImageNet normalization in Phase 3."
    ),
    nbf.v4.new_code_cell(
        "rng = np.random.default_rng(SEED)\n"
        "sample_idx = rng.choice(len(train_files), size=PIXEL_STATS_SAMPLE, replace=False)\n"
        "sums = np.zeros(3, dtype=np.float64)\n"
        "sumsq = np.zeros(3, dtype=np.float64)\n"
        "n_pix = 0\n"
        "for i in tqdm(sample_idx, desc='pixel stats'):\n"
        "    p = REPO_ROOT / train_files[int(i)]['filepath']\n"
        "    try:\n"
        "        with Image.open(p) as im:\n"
        "            arr = np.asarray(im.convert('RGB'), dtype=np.float64) / 255.0\n"
        "    except Exception:\n"
        "        continue\n"
        "    sums  += arr.reshape(-1, 3).sum(axis=0)\n"
        "    sumsq += (arr.reshape(-1, 3) ** 2).sum(axis=0)\n"
        "    n_pix += arr.shape[0] * arr.shape[1]\n"
        "\n"
        "mean = sums / n_pix\n"
        "std  = np.sqrt(sumsq / n_pix - mean ** 2)\n"
        "print(f'computed mean = {mean}')\n"
        "print(f'computed std  = {std}')\n"
    ),
    nbf.v4.new_code_cell(
        "imnet_mean = np.array([0.485, 0.456, 0.406])\n"
        "imnet_std  = np.array([0.229, 0.224, 0.225])\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
        "x = np.arange(3); w = 0.35; channels = ['R', 'G', 'B']\n"
        "axes[0].bar(x - w/2, mean, w, label='Food-101')\n"
        "axes[0].bar(x + w/2, imnet_mean, w, label='ImageNet')\n"
        "axes[0].set_xticks(x); axes[0].set_xticklabels(channels); axes[0].set_title('mean')\n"
        "axes[0].legend()\n"
        "axes[1].bar(x - w/2, std, w, label='Food-101')\n"
        "axes[1].bar(x + w/2, imnet_std, w, label='ImageNet')\n"
        "axes[1].set_xticks(x); axes[1].set_xticklabels(channels); axes[1].set_title('std')\n"
        "axes[1].legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIGURES_DIR / 'pixel_stats.png', dpi=150)\n"
        "plt.show()\n"
    ),
    nbf.v4.new_code_cell(
        "update_stats('pixel_stats_native', {\n"
        "    'mean': mean.tolist(),\n"
        "    'std':  std.tolist(),\n"
        "    'n_sampled': PIXEL_STATS_SAMPLE,\n"
        "})\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -3
```

Expected: `pixel stats OK`. Pixel sampling takes ~1–2 min on cold run.

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 4 — per-channel pixel stats vs ImageNet"
```

---

## Task 7: Section 5 — Visual sampling (10 random per class)

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: 101 sample-grid PNGs exist (one per class)\n"
    "samples_dir = FIGURES_DIR / 'samples'\n"
    "pngs = sorted(samples_dir.glob('*.png'))\n"
    "assert len(pngs) == 101, f'expected 101 sample PNGs, got {len(pngs)}'\n"
    "expected_names = {f'{c}.png' for c in class_names}\n"
    "actual_names = {p.name for p in pngs}\n"
    "assert expected_names == actual_names, f'class/file mismatch: {expected_names ^ actual_names}'\n"
    "print('visual samples OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails (0 PNGs found).

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 5. Visual Sampling\n"
        "10 random images per class as a 2×5 grid; one PNG per class. "
        "For human eyeball review of label noise — Food-101 is known to have ~5%."
    ),
    nbf.v4.new_code_cell(
        "by_class = {c: [r for r in train_files if r['label_name'] == c] for c in class_names}\n"
        "rng = np.random.default_rng(SEED)\n"
        "samples_dir = FIGURES_DIR / 'samples'\n"
        "samples_dir.mkdir(parents=True, exist_ok=True)\n"
        "\n"
        "existing = {p.stem for p in samples_dir.glob('*.png')}\n"
        "to_render = [c for c in class_names if c not in existing or FORCE_REBUILD]\n"
        "print(f'rendering {len(to_render)} class grids ({len(existing)} already cached)')\n"
        "\n"
        "for cls in tqdm(to_render, desc='sample grids'):\n"
        "    rows = by_class[cls]\n"
        "    picks = rng.choice(len(rows), size=10, replace=False)\n"
        "    fig, axes = plt.subplots(2, 5, figsize=(12, 5))\n"
        "    for ax, idx in zip(axes.flatten(), picks):\n"
        "        path = REPO_ROOT / rows[int(idx)]['filepath']\n"
        "        try:\n"
        "            with Image.open(path) as im:\n"
        "                ax.imshow(im.convert('RGB'))\n"
        "        except Exception:\n"
        "            ax.text(0.5, 0.5, 'corrupt', ha='center', va='center')\n"
        "        ax.axis('off')\n"
        "    fig.suptitle(cls, fontsize=14)\n"
        "    fig.tight_layout()\n"
        "    fig.savefig(samples_dir / f'{cls}.png', dpi=120)\n"
        "    plt.close(fig)\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb --ExecutePreprocessor.timeout=900 2>&1 | tail -3
```

Expected: `visual samples OK`. First run renders 101 grids (~3–5 min).

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 5 — per-class visual sample grids"
```

---

## Task 8: Section 6a — Audit: corrupt + tiny

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: audit cache has expected columns; corrupt/tiny lists are computed\n"
    "assert AUDIT_CACHE_PATH.exists(), 'audit cache missing'\n"
    "ad = pd.read_parquet(AUDIT_CACHE_PATH)\n"
    "assert {'filepath', 'is_corrupt', 'short_side'}.issubset(ad.columns), f'columns: {ad.columns.tolist()}'\n"
    "assert isinstance(corrupt_files, list) and isinstance(tiny_files, list)\n"
    "for entry in tiny_files[:1]:\n"
    "    assert 'path' in entry and 'short_side' in entry\n"
    "print(f'audit pass 1 OK — corrupt={len(corrupt_files)} tiny={len(tiny_files)}')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails (audit cache missing or `corrupt_files` undefined).

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 6a. Audit — Corrupt + Tiny\n"
        "Walk every train+test file. `PIL.Image.open(...).verify()` flags structural corruption. "
        "`shorter_side < TINY_SHORT_SIDE` flags images that would force aggressive upscaling at 224."
    ),
    nbf.v4.new_code_cell(
        "def _check_one(rel_path):\n"
        "    p = REPO_ROOT / rel_path\n"
        "    try:\n"
        "        with Image.open(p) as im:\n"
        "            im.verify()\n"
        "        with Image.open(p) as im:\n"
        "            w, h = im.size\n"
        "        return False, min(w, h)\n"
        "    except Exception:\n"
        "        return True, -1\n"
        "\n"
        "if AUDIT_CACHE_PATH.exists() and not FORCE_REBUILD:\n"
        "    audit_df = pd.read_parquet(AUDIT_CACHE_PATH)\n"
        "    print(f'loaded cached audit for {len(audit_df):,} files')\n"
        "else:\n"
        "    rows = []\n"
        "    for r in tqdm(train_files + test_files, desc='audit'):\n"
        "        is_corrupt, short_side = _check_one(r['filepath'])\n"
        "        rows.append({'filepath': r['filepath'], 'is_corrupt': is_corrupt, 'short_side': short_side})\n"
        "    audit_df = pd.DataFrame(rows)\n"
        "    audit_df.to_parquet(AUDIT_CACHE_PATH)\n"
        "    print(f'audited {len(audit_df):,} files, cached')\n"
    ),
    nbf.v4.new_code_cell(
        "corrupt_files = audit_df.loc[audit_df['is_corrupt'], 'filepath'].tolist()\n"
        "tiny_mask = (~audit_df['is_corrupt']) & (audit_df['short_side'] < TINY_SHORT_SIDE)\n"
        "tiny_files = [\n"
        "    {'path': r.filepath, 'short_side': int(r.short_side)}\n"
        "    for r in audit_df[tiny_mask].itertuples()\n"
        "]\n"
        "print(f'corrupt: {len(corrupt_files)}')\n"
        "print(f'tiny (shorter side < {TINY_SHORT_SIDE}): {len(tiny_files)}')\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb --ExecutePreprocessor.timeout=900 2>&1 | tail -3
```

Expected: `audit pass 1 OK — corrupt=N tiny=M`. Audit takes ~3–5 min cold.

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 6a — corrupt + tiny file audit"
```

---

## Task 9: Section 6b — Audit: near-duplicates (within-class pHash)

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: pHash cache exists; near_dupes computed within-class only\n"
    "assert PHASH_CACHE_PATH.exists(), 'pHash cache missing'\n"
    "ph = pd.read_parquet(PHASH_CACHE_PATH)\n"
    "assert {'filepath', 'phash_hex', 'class'}.issubset(ph.columns)\n"
    "assert isinstance(near_dupes, list)\n"
    "if near_dupes:\n"
    "    e = near_dupes[0]\n"
    "    assert {'a', 'b', 'phash_hamming', 'class'}.issubset(e.keys())\n"
    "    assert e['phash_hamming'] <= PHASH_HAMMING_THRESHOLD\n"
    "    a_class = e['a'].split('/')[-2]\n"
    "    b_class = e['b'].split('/')[-2]\n"
    "    assert a_class == b_class == e['class'], 'pHash escaped within-class scope'\n"
    "print(f'audit pass 2 OK — near_duplicate pairs={len(near_dupes)}')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails on missing pHash cache or undefined `near_dupes`.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 6b. Audit — Near-Duplicates (within-class pHash)\n"
        "Compute a 64-bit perceptual hash per image. Compare pairwise *within each class only* — "
        "cross-class is O(n²) on 100k images and the architectural meaning is different "
        "(cross-class dups are label noise, addressed by visual sampling)."
    ),
    nbf.v4.new_code_cell(
        "import imagehash\n"
        "\n"
        "all_records = train_files + test_files\n"
        "\n"
        "if PHASH_CACHE_PATH.exists() and not FORCE_REBUILD:\n"
        "    phash_df = pd.read_parquet(PHASH_CACHE_PATH)\n"
        "    print(f'loaded cached pHashes for {len(phash_df):,} images')\n"
        "else:\n"
        "    rows = []\n"
        "    for r in tqdm(all_records, desc='pHash'):\n"
        "        p = REPO_ROOT / r['filepath']\n"
        "        try:\n"
        "            with Image.open(p) as im:\n"
        "                h = str(imagehash.phash(im.convert('RGB')))\n"
        "        except Exception:\n"
        "            h = None\n"
        "        rows.append({'filepath': r['filepath'], 'phash_hex': h, 'class': r['label_name']})\n"
        "    phash_df = pd.DataFrame(rows)\n"
        "    phash_df.to_parquet(PHASH_CACHE_PATH)\n"
        "    print(f'computed pHash for {len(phash_df):,} images, cached')\n"
    ),
    nbf.v4.new_code_cell(
        "def _hex_to_int(s):\n"
        "    return int(s, 16) if s else None\n"
        "\n"
        "def _hamming(a, b):\n"
        "    return bin(a ^ b).count('1')\n"
        "\n"
        "near_dupes = []\n"
        "valid = phash_df.dropna(subset=['phash_hex']).copy()\n"
        "valid['phash_int'] = valid['phash_hex'].map(_hex_to_int)\n"
        "\n"
        "for cls, grp in tqdm(valid.groupby('class'), desc='within-class pairs'):\n"
        "    paths = grp['filepath'].tolist()\n"
        "    hashes = grp['phash_int'].tolist()\n"
        "    n = len(hashes)\n"
        "    for i in range(n):\n"
        "        for j in range(i + 1, n):\n"
        "            d = _hamming(hashes[i], hashes[j])\n"
        "            if d <= PHASH_HAMMING_THRESHOLD:\n"
        "                near_dupes.append({\n"
        "                    'a': paths[i], 'b': paths[j],\n"
        "                    'phash_hamming': int(d), 'class': cls,\n"
        "                })\n"
        "print(f'found {len(near_dupes)} near-duplicate pairs (hamming <= {PHASH_HAMMING_THRESHOLD})')\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook (cold pHash takes ~10–15 min)**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb --ExecutePreprocessor.timeout=1800 2>&1 | tail -3
```

Expected: `audit pass 2 OK — near_duplicate pairs=N`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 6b — within-class pHash near-duplicates"
```

---

## Task 10: Section 6c — Merge → bad_files.json

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: bad_files.json exists with three keys, each a list\n"
    "import json as _json\n"
    "assert BAD_FILES_PATH.exists(), 'bad_files.json missing'\n"
    "bf = _json.loads(BAD_FILES_PATH.read_text())\n"
    "for k in ('corrupt', 'tiny', 'near_duplicates'):\n"
    "    assert k in bf, f'missing key: {k}'\n"
    "    assert isinstance(bf[k], list), f'{k} must be a list'\n"
    "print('bad_files.json OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails on missing bad_files.json.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 6c. Merge → bad_files.json\n"
        "Aggregate the three issue lists into the single artifact Phase 3 will read."
    ),
    nbf.v4.new_code_cell(
        "bad_files = {\n"
        "    'corrupt':         corrupt_files,\n"
        "    'tiny':            tiny_files,\n"
        "    'near_duplicates': near_dupes,\n"
        "}\n"
        "BAD_FILES_PATH.write_text(json.dumps(bad_files, indent=2, sort_keys=True))\n"
        "print(f'wrote {BAD_FILES_PATH}')\n"
        "print(f'  corrupt:         {len(bad_files[\"corrupt\"])}')\n"
        "print(f'  tiny:            {len(bad_files[\"tiny\"])}')\n"
        "print(f'  near_duplicates: {len(bad_files[\"near_duplicates\"])}')\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -3
```

Expected: `bad_files.json OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 6c — merge audit results to bad_files.json"
```

---

## Task 11: Section 7 — Stratified train/val split + write split CSVs

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: split CSVs exist with correct row counts, columns, and val per-class balance\n"
    "for split, expected in [('train', 68175), ('val', 7575), ('test', 25250)]:\n"
    "    p = SPLITS_DIR / f'{split}.csv'\n"
    "    assert p.exists(), f'missing {p}'\n"
    "    df = pd.read_csv(p)\n"
    "    assert list(df.columns) == ['filepath', 'label_idx', 'label_name'], f'{split} columns: {df.columns.tolist()}'\n"
    "    assert len(df) == expected, f'{split} rows: {len(df)} != {expected}'\n"
    "    assert df['label_idx'].nunique() == 101, f'{split} class count: {df[\"label_idx\"].nunique()}'\n"
    "val_per_class = pd.read_csv(SPLITS_DIR / 'val.csv').groupby('label_name').size()\n"
    "assert val_per_class.between(73, 77).all(), f'val per-class out of band: min={val_per_class.min()} max={val_per_class.max()}'\n"
    "print('splits OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails on missing splits CSVs.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 7. Splits\n"
        "Carve a 10% stratified validation set out of train. Test is untouched (mirrors the official Food-101 test list). "
        "**REGENERATE_SPLITS** is a hard guard — flipping the seed silently invalidates every downstream comparison."
    ),
    nbf.v4.new_code_cell(
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "splits_exist = all((SPLITS_DIR / f'{s}.csv').exists() for s in ('train', 'val', 'test'))\n"
        "if splits_exist and not REGENERATE_SPLITS:\n"
        "    print('splits already exist — set REGENERATE_SPLITS=True to overwrite (this invalidates downstream comparisons)')\n"
        "else:\n"
        "    train_df = pd.DataFrame(train_files)\n"
        "    test_df  = pd.DataFrame(test_files)\n"
        "    tr_idx, val_idx = train_test_split(\n"
        "        np.arange(len(train_df)),\n"
        "        test_size=VAL_FRACTION,\n"
        "        stratify=train_df['label_idx'].values,\n"
        "        random_state=SEED,\n"
        "    )\n"
        "    cols = ['filepath', 'label_idx', 'label_name']\n"
        "    train_df.iloc[tr_idx][cols].to_csv(SPLITS_DIR / 'train.csv', index=False)\n"
        "    train_df.iloc[val_idx][cols].to_csv(SPLITS_DIR / 'val.csv',   index=False)\n"
        "    test_df[cols].to_csv(SPLITS_DIR / 'test.csv', index=False)\n"
        "    print(f'wrote splits: train={len(tr_idx)}  val={len(val_idx)}  test={len(test_df)}')\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -3
```

Expected: `splits OK`.

- [ ] **Step 4: Verify split files on disk**

```bash
wc -l splits/*.csv
```

Expected: `68176 splits/train.csv`, `7576 splits/val.csv`, `25251 splits/test.csv` (each includes one header row).

- [ ] **Step 5: Commit (notebook + splits — splits CSVs are committed for reproducibility)**

```bash
git add notebooks/phase2_data_understanding.ipynb splits/train.csv splits/val.csv splits/test.csv
git commit -m "feat(phase2): notebook section 7 + lock train/val/test splits"
```

---

## Task 12: Section 8 — Summary recap

**Files:**
- Modify: `notebooks/phase2_data_understanding.ipynb`

- [ ] **Step 1: Append the assertion cell first**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
nb["cells"].append(nbf.v4.new_code_cell(
    "# ASSERT: summary cell ran (a sentinel string is printed)\n"
    "assert summary_printed is True, 'summary did not run'\n"
    "import json as _json\n"
    "stats = _json.loads(EDA_STATS_PATH.read_text())\n"
    "required = {'n_train', 'n_test', 'n_classes', 'class_counts', 'dims',\n"
    "            'pixel_stats_native', 'imagenet_stats', 'split_seed'}\n"
    "assert required.issubset(stats.keys()), f'eda_stats.json missing keys: {required - stats.keys()}'\n"
    "print('summary OK')"
))
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -5
```

Expected: assertion fails on missing `summary_printed`.

- [ ] **Step 2: Insert implementation cells before the assertion**

```bash
python - <<'PY'
import nbformat as nbf
nb = nbf.read("notebooks/phase2_data_understanding.ipynb", as_version=4)
assertion = nb["cells"].pop()
cells_to_add = [
    nbf.v4.new_markdown_cell(
        "## 8. Summary\n"
        "One-screen recap. The two human-decision lines — **normalization choice** and "
        "**label-noise verdict** — are filled in below by the engineer after reviewing "
        "section 4 (pixel stats vs ImageNet) and section 5 (per-class sample grids)."
    ),
    nbf.v4.new_code_cell(
        "stats = json.loads(EDA_STATS_PATH.read_text())\n"
        "bad   = json.loads(BAD_FILES_PATH.read_text())\n"
        "\n"
        "n_classes = stats['n_classes']\n"
        "n_train, n_test = stats['n_train'], stats['n_test']\n"
        "balance = 'balanced exactly' if set(stats['class_counts'].values()) == {750} else 'imbalanced'\n"
        "\n"
        "print('=' * 60)\n"
        "print('PHASE 2 SUMMARY — Food-101 / Swin Transformer')\n"
        "print('=' * 60)\n"
        "print(f'classes:                  {n_classes}')\n"
        "print(f'train / test images:      {n_train:,} / {n_test:,}')\n"
        "print(f'class balance:            {balance}')\n"
        "print(f'width  p50/p95 (px):      {stats[\"dims\"][\"width\"][\"p50\"]:.0f} / {stats[\"dims\"][\"width\"][\"p95\"]:.0f}')\n"
        "print(f'height p50/p95 (px):      {stats[\"dims\"][\"height\"][\"p50\"]:.0f} / {stats[\"dims\"][\"height\"][\"p95\"]:.0f}')\n"
        "print(f'aspect p05/p95:           {stats[\"dims\"][\"aspect\"][\"min\"]:.2f} / {stats[\"dims\"][\"aspect\"][\"p95\"]:.2f}')\n"
        "print(f'pixel mean (Food-101):    {[round(x, 3) for x in stats[\"pixel_stats_native\"][\"mean\"]]}')\n"
        "print(f'pixel mean (ImageNet):    {stats[\"imagenet_stats\"][\"mean\"]}')\n"
        "print(f'pixel std  (Food-101):    {[round(x, 3) for x in stats[\"pixel_stats_native\"][\"std\"]]}')\n"
        "print(f'pixel std  (ImageNet):    {stats[\"imagenet_stats\"][\"std\"]}')\n"
        "print(f'audit corrupt:            {len(bad[\"corrupt\"])}')\n"
        "print(f'audit tiny (<{TINY_SHORT_SIDE} px):     {len(bad[\"tiny\"])}')\n"
        "print(f'audit near-duplicates:    {len(bad[\"near_duplicates\"])}')\n"
        "print(f'split seed:               {stats[\"split_seed\"]}')\n"
        "print(f'split sizes (tr/val/te):  68,175 / 7,575 / 25,250')\n"
        "print('-' * 60)\n"
        "print('HUMAN DECISIONS (fill in after reviewing sections 4 and 5):')\n"
        "print('  normalization choice:   <ImageNet | Food-101>')\n"
        "print('  label-noise verdict:    <acceptable | flag classes: ...>')\n"
        "print('=' * 60)\n"
        "summary_printed = True\n"
    ),
]
nb["cells"].extend(cells_to_add)
nb["cells"].append(assertion)
nbf.write(nb, "notebooks/phase2_data_understanding.ipynb")
PY
```

- [ ] **Step 3: Run the notebook**

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb 2>&1 | tail -3
```

Expected: `summary OK`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/phase2_data_understanding.ipynb
git commit -m "feat(phase2): notebook section 8 — summary recap"
```

---

## Task 13: End-to-end warm-run validation

**Files:**
- (none — verification only)

- [ ] **Step 1: Confirm all artifacts exist**

```bash
ls -la artifacts/phase2/eda_stats.json artifacts/phase2/bad_files.json \
       artifacts/phase2/phash_cache.parquet artifacts/phase2/dims_cache.parquet \
       artifacts/phase2/audit_cache.parquet \
       splits/train.csv splits/val.csv splits/test.csv \
       figures/phase2/class_dist.png figures/phase2/dims_hist.png figures/phase2/pixel_stats.png
ls figures/phase2/samples | wc -l   # should print 101
```

- [ ] **Step 2: Verify warm re-run finishes under 60 seconds**

```bash
time jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_data_understanding.ipynb
```

Expected: real time under 60 s; every section prints its "loaded cached X" message.

- [ ] **Step 3: Verify all 7 acceptance criteria from the spec**

Run a one-liner sanity check covering every spec acceptance criterion:

```bash
python - <<'PY'
import json
from pathlib import Path
import pandas as pd

p = Path('.')
# (1) data acquired
assert (p / 'data/food-101/images').exists(), 'data/food-101/images missing'
# (2) splits with right counts and val balance
counts = {s: len(pd.read_csv(p / f'splits/{s}.csv')) for s in ('train', 'val', 'test')}
assert counts == {'train': 68175, 'val': 7575, 'test': 25250}, counts
val = pd.read_csv(p / 'splits/val.csv').groupby('label_name').size()
assert val.between(73, 77).all(), val.describe()
# (3) eda_stats.json keys
stats = json.loads((p / 'artifacts/phase2/eda_stats.json').read_text())
required = {'n_train', 'n_test', 'n_classes', 'class_counts', 'dims',
            'pixel_stats_native', 'imagenet_stats', 'split_seed'}
assert required.issubset(stats.keys()), required - stats.keys()
# (4) bad_files.json keys
bad = json.loads((p / 'artifacts/phase2/bad_files.json').read_text())
assert {'corrupt', 'tiny', 'near_duplicates'}.issubset(bad.keys())
# (5) figures present
for f in ['class_dist.png', 'dims_hist.png', 'pixel_stats.png']:
    assert (p / 'figures/phase2' / f).exists(), f
assert len(list((p / 'figures/phase2/samples').glob('*.png'))) == 101
print('ALL 7 ACCEPTANCE CRITERIA PASS')
PY
```

Expected: `ALL 7 ACCEPTANCE CRITERIA PASS`.

- [ ] **Step 4: (Optional) Tag the Phase 2 milestone**

```bash
git tag -a phase2-complete -m "Phase 2 — Data Understanding complete"
```

---

## Self-Review

**Spec coverage:** Every spec section maps to a task — Setup → Task 2; Acquisition → Task 3; Class distribution → Task 4; Dimensions → Task 5; Pixel stats → Task 6; Visual sampling → Task 7; Audit (corrupt/tiny/dupes) → Tasks 8-10; Splits → Task 11; Summary → Task 12; Acceptance criteria → Task 13.

**Placeholders:** None. Every step contains executable code or commands.

**Type/name consistency:** Constants (`SEED`, `VAL_FRACTION`, `PIXEL_STATS_SAMPLE`, `TINY_SHORT_SIDE`, `PHASH_HAMMING_THRESHOLD`, `PHASH_SCOPE`, `FORCE_REBUILD`, `REGENERATE_SPLITS`) and paths (`DATA_ROOT`, `ARTIFACTS_DIR`, `FIGURES_DIR`, `SPLITS_DIR`, `EDA_STATS_PATH`, `BAD_FILES_PATH`, `PHASH_CACHE_PATH`, `AUDIT_CACHE_PATH`, `DIMS_CACHE`) are consistent across tasks. Stats keys match the spec's `eda_stats.json` shape exactly. Variable names (`train_files`, `test_files`, `class_names`, `corrupt_files`, `tiny_files`, `near_dupes`, `bad_files`) flow correctly from one task's implementation to the next task's assertion.
