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
