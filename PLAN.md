# Swin Transformer on Food-101 — Project Plan

A data-driven image classification pipeline. Data work bookends modeling: EDA shapes design choices upfront, and error analysis drives iteration afterward.

## Phase 1 — Problem framing & success criteria

1. **Define the task precisely**: 101-way single-label classification, top-1 accuracy as headline metric, top-5 as secondary.
2. **Set targets**: e.g. baseline Swin-Tiny ≥ 88% top-1, stretch ≥ 91%. Pick from published numbers so success is verifiable.
3. **Lock constraints**: Colab T4 budget, single session, 224×224 input, batch size that fits VRAM.

## Phase 2 — Data understanding (the heart of "data-driven")

4. **Acquire Food-101** and verify checksums / file integrity.
5. **EDA on the dataset**:
   - Class distribution (Food-101 is balanced — confirm, don't assume).
   - Image resolution & aspect ratio histograms — informs resize strategy.
   - Per-channel pixel statistics — confirm we should use ImageNet normalization stats.
   - Visual sampling: 10 random images per class, eyeball for label noise (Food-101 is known to have ~5% noisy labels).
6. **Data quality audit**: detect corrupt files, near-duplicates (perceptual hash), unusually small images.
7. **Splits**: Food-101 ships with fixed train (75,750) / test (25,250). Carve a validation set out of train (e.g. 10% stratified) — never touch test until final.

## Phase 3 — Preprocessing & augmentation strategy

8. **Preprocessing**: resize-shorter-side → center crop for eval; resize → random resized crop for train. Normalize with ImageNet mean/std.
9. **Augmentation policy**: start with a known-good recipe from the Swin paper (RandAugment + Mixup + CutMix + Random Erasing). Justify each based on EDA findings (e.g. if EDA shows tight crops, lean harder on RandomResizedCrop).

## Phase 4 — Baseline model

10. **Pick the model**: Swin-Tiny, ImageNet-1K pretrained, from `timm`. Replace the 1000-class head with 101-class.
11. **Decide what to fine-tune**: full network vs. head-only vs. layer-wise LR decay. Default: full fine-tune with LLRD.
12. **Training config**: AdamW, cosine schedule with warmup, mixed precision, gradient clipping, label smoothing 0.1.
13. **Train the baseline** end to end. This is the number every later change is measured against.

## Phase 5 — Evaluation framework (built before iteration, not after)

14. **Metrics dashboard**: top-1, top-5, macro/micro F1, per-class accuracy, confusion matrix.
15. **Error slicing**: worst 10 classes, most-confused pairs, hardest individual examples (highest-loss correctly-labeled images).
16. **Calibration check**: reliability diagram — does the model know when it's wrong?

## Phase 6 — Data-driven iteration (this is where it earns the name)

17. **Error analysis → hypothesis**: e.g. "steak vs. filet mignon" confusion → visual inspection → are these genuinely ambiguous, or is augmentation cropping out the discriminative region?
18. **Targeted intervention**: change augmentation, add class-balanced sampling for confused pairs, clean obvious label noise — *one change per experiment*.
19. **Re-evaluate** against the same dashboard. Keep an experiment log.

## Phase 7 — Final reporting

20. **Final test-set evaluation** (once, at the end).
21. **Write up**: dataset insights from EDA, design choices justified by data, error analysis findings, what worked / what didn't.
