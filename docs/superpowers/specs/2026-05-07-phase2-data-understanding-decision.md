# Decision Note — Phase 2 Data Understanding Notebook

**Date:** 2026-05-07
**Spec:** [2026-05-07-phase2-data-understanding-design.md](2026-05-07-phase2-data-understanding-design.md)

## Context

Project is at the start of Phase 2 of a planned seven-phase Swin-Transformer-on-Food-101 pipeline (see [PLAN.md](../../../PLAN.md)). Repo is fresh: `PLAN.md`, empty `README.md`, no prior code, no prior byterover entries. Phase 1 fixed the runtime budget (Colab T4 for training, 224×224 input) and success metrics (top-1 ≥ 88%). Phase 2 must produce: a verified dataset on disk, an EDA characterization, a quality audit, and a locked train/val/test split — all consumed by Phase 3+. The user opted to run Phase 2 locally (not Colab) and wants the notebook itself to handle the download. The notebook is the authoritative driver; Phase 3+ reads the artifacts it writes.

## Choice

A single notebook `notebooks/phase2_data_understanding.ipynb` runs locally and executes the eight sections of Phase 2 top-to-bottom (setup → acquire → class distribution → dimensions → pixel stats → visual sampling → quality audit → splits → summary). Each section writes its artifact to disk before the next begins. Acquisition uses `torchvision.datasets.Food101` (handles MD5 verification). Splits go to committed `splits/*.csv`; numeric EDA goes to `artifacts/phase2/eda_stats.json`; quality issues go to `artifacts/phase2/bad_files.json`; figures go to `figures/phase2/`. Long-running cells cache to disk and skip recompute on re-runs unless `FORCE_REBUILD=True`. All Phase 2 logic lives in the notebook — no `src/` extraction yet.

## Alternatives Rejected

- **Run on Colab + Drive cache:** Rejected — user prefers local for faster EDA iteration; Colab is reserved for Phase 4 training where the GPU matters.
- **Manual tar download + checksum:** Rejected — torchvision wraps the same download with checksum verification in one call; manual mechanics teach nothing of value here.
- **HuggingFace `datasets` loader:** Rejected — Arrow-backed storage hides the on-disk file structure that the integrity audit (corrupt files, near-duplicates by path) needs to walk.
- **Extract logic into `src/data/` modules now:** Rejected — premature; Phase 3 will reveal what training actually needs from a `Dataset` class. Accept the duplication risk and refactor when concrete.
- **Cross-class perceptual hashing:** Rejected — O(n²) on 100k images is ~5×10⁹ pairs, infeasible. Cross-class duplicates are a label-noise problem, addressed by visual sampling instead.
- **Multiple notebooks (one per Phase 2 step):** Rejected — the user asked for "a notebook" (singular); the steps share state (file lists, label maps) and split well into sequential sections within one file.

## Invariants Preserved

- **Test set is never touched until final reporting (Phase 7).** Stratified split carves val out of train only; `test.csv` mirrors the official Food-101 test list verbatim.
- **Split seed is locked at 42 and the regeneration path is gated.** A separate `REGENERATE_SPLITS=False` flag prevents silent re-shuffling that would invalidate every downstream comparison.
- **Splits use relative `filepath`.** The dataset directory can be moved without breaking committed split CSVs.
- **Phase 2 produces artifacts only — it does not act on them.** Filtering bad files happens in Phase 3 at dataset-load time. Phase 2 surfaces; later phases decide.
- **Reproducibility over convenience.** All randomness is seeded; all expensive computation is cached deterministically; the summary cell is a single source of human-readable truth for what Phase 2 concluded.

## In-flight Refinements

None. The plan executed end-to-end on the first attempt with no deviations. The only minor variance was commit count (11 commits for 13 tasks — Tasks 9+10 batched into one commit, Task 13 verification-only had no commit), which doesn't change behavior. All 7 acceptance criteria pass; warm re-run is 32 s (target was <60 s).
