# Decision Note — Phases 3 + 4 + 5 `foodnet` Python Package

**Date:** 2026-05-07
**Spec:** [2026-05-07-phases-3-4-5-package-design.md](2026-05-07-phases-3-4-5-package-design.md)

## Context

Phase 2 finished with a verified Food-101 dataset on disk, a locked Phase-2 train/val/test split (committed `splits/*.csv`), an EDA characterization (`artifacts/phase2/eda_stats.json`), and a quality audit (`bad_files.json`). PLAN.md Phases 3, 4, 5 now turn that into a working baseline: data pipeline + augmentation, Swin-Tiny model + training, and evaluation framework. The user articulated four pivotal constraints during brainstorming: (1) all three phases as one cohesive Python package, not three separate efforts; (2) CLI-driven entry points (pure argparse, no YAML / Hydra) plus a thin notebook driver; (3) training on Kaggle P100 with the Food-101 mounted as a Kaggle dataset; (4) W&B as the experiment tracker. The user also chose **not** to filter `bad_files.json` at training time — log a warning, let the dataset's PIL-error path resample on the fly. Most consequentially, the user asked to resplit the dataset 8:1:1 across all 101,000 images via a package CLI — overriding Phase 2's locked invariant that "test = official Food-101 test verbatim." Prior byterover context for Phases 3-5 was empty (confirmed via `brv query`), so this brainstorm establishes the architectural memory from scratch for the modeling half of the project.

## Choice

A single pip-installable Python package `foodnet/` with modules for `splitting/`, `data/`, `models/`, `training/`, `eval/`, `cli/`, `utils/`. Four CLIs — `foodnet.cli.split` (regenerates 8:1:1 stratified splits with exactly 800/100/100 per class), `foodnet.cli.train` (Swin-Tiny baseline with LLRD + AdamW + cosine + warmup + fp16 AMP + grad-clip + Mixup/CutMix + early stopping at patience=8 on val/top1, full W&B logging), `foodnet.cli.evaluate` (inference pass that dumps per-example softmax + loss to a parquet), and `foodnet.cli.analyze` (reads that parquet and produces metrics JSON, confusion heatmap, reliability/ECE diagram, worst-K classes, hardest-correct + hardest-incorrect example tables, and a summary `dashboard.md`). A single notebook `notebooks/phases_3_4_5_pipeline.ipynb` detects environment (Kaggle vs local) and drives the four CLIs sequentially. Splits CSVs use paths relative to `images_root` (not repo-relative) so the same CSV works in both environments unchanged. Tests use synthetic 32×32 PNG fixtures in pytest; Food-101 is not required for `pytest tests/`.

## Alternatives Rejected

- **Three separate packages or three independent efforts (one per phase):** Rejected — the phases share configuration, dataset, and checkpoints; cohesion outweighs separation cost.
- **Phase-named modules (`phase3.py`, `phase4.py`, `phase5.py`):** Rejected — encourages god-files; capability-layered modules survive Phase 6 ablations better.
- **Trainer god-class (Lightning-style without the framework):** Rejected — adds an abstraction layer this project doesn't need with one model and one task.
- **Hydra / YAML configs:** Rejected by user choice — pure CLI flags. Reproducibility lives in `args.json` dumped alongside each checkpoint.
- **TensorBoard or JSON-only experiment tracking:** Rejected by user choice — W&B is the tracker.
- **Filter `bad_files.json` at dataset-load time (any of: corrupt, tiny, near-dupes):** Rejected by user choice — keep all rows, log startup warning, let `__getitem__` resample on PIL error.
- **Notebook-driven, package as library (Phase 2 style):** Rejected — Kaggle and reproducibility both want CLIs; the notebook becomes a thin driver instead of where logic accretes.
- **Three separate notebooks (one per phase):** Rejected by user — single unified notebook fits Kaggle's "one session = one notebook" model and matches the Phase 2 precedent.
- **Keep Phase 2's official-test-verbatim splits:** Rejected by user — uniform 8:1:1 reproducible from a CLI on any environment was preferred over apples-to-apples literature comparability. Trade-off documented (PLAN.md ≥88% target re-baselined against the 8:1:1 test set).
- **EMA model weights for the baseline:** Deferred to Phase 6 — doubles checkpoint size, adds a moving piece; the baseline doesn't need it.
- **TTA (test-time augmentation):** Deferred — not in PLAN.md; baseline reports plain single-crop eval.

## Invariants Preserved

- **Reproducibility from a single seed.** `--seed 42` is the package-wide default. The splits CLI is deterministic given the seed; `splits_meta.json` records sha256s of all three CSVs; checkpoints record the splits_meta sha256 they trained on. Drift is detectable.
- **`__init__.py`-defined module boundaries.** Each `foodnet/` subpackage has one job: `splitting/` produces splits, `data/` consumes them, `models/` builds architectures, `training/` orchestrates the loop, `eval/` analyzes outputs, `cli/` is the user-facing surface. Phase 6 changes augmentation in `data/` without touching `training/` or `models/`.
- **Two-stage evaluation: inference once → analyze many.** `evaluate` writes a per-example parquet that captures full softmax + loss; `analyze` re-renders any plot from that parquet in seconds. Phase 6 dashboards iterate on plotting without re-running the model.
- **`args.json` + `splits_meta` traveling with every checkpoint.** A checkpoint can always be matched to the exact CLI invocation and exact split that produced it. Critical for Phase-6 ablations where multiple runs need to be compared.
- **Kaggle and local share one code path.** The only difference is the `--images-root` and `--output-dir` flags; environment detection lives in the notebook's section 0, not scattered through the package.
- **The training CLI refuses to silently clobber prior runs.** If `last.pt` exists for a `run_id` and `--resume` is not passed, the CLI exits with an error. Users must opt into either resuming or starting fresh.

## Superseded Invariants (Phase 2)

- **"Test set is never touched until final reporting; `test.csv` mirrors the official Food-101 test list verbatim."** Superseded by the user's choice of an 8:1:1 stratified split across all 101,000 images. The Phase 2 entry will be flagged for byterover review during curation.

## In-flight Refinements

None yet — implementation has not started.
