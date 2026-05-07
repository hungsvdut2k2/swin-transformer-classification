---
title: Project Overview
summary: Swin Transformer classification project structure, dataset, and phase-based development lifecycle
tags: []
related: [project_overview/project_details/project_details.md, project_overview/phase_2_data_understanding/phase_2_data_understanding.md]
keywords: []
createdAt: '2026-05-07T09:20:21.924Z'
updatedAt: '2026-05-07T11:36:26.201Z'
---
## Reason
Curate project overview and scope from RLM context

## Raw Concept
**Task:**
Document Swin Transformer classification project

**Changes:**
- Initial project documentation

**Files:**
- docs/
- .brv/
- data/food-101/
- notebooks/phase2_data_understanding.ipynb
- splits/train.csv
- splits/test.csv
- splits/val.csv
- requirements.txt

**Flow:**
Phase 2 (Data Understanding) -> Phase 3 (Preprocessing) -> Phase 4 (Baseline Model) -> Phase 5 (Evaluation)

**Timestamp:** 2026-05-07

## Narrative
### Structure
Project organized into phases with notebooks in notebooks/ and results in artifacts/.

### Highlights
Swin Transformer model for Food-101 classification.

### Rules
Follow phased development lifecycle.

## Facts
- **foodnet package**: The foodnet package is a single pip-installable library with capability-layered modules.
- **CLI**: The project uses four CLI entry points: split, train, evaluate, and analyze.
- **Reproducibility**: Reproducibility is enforced via a fixed seed of 42 and tracking splits_meta sha256s in checkpoints.
- **Evaluation**: The evaluation process is split into two stages: inference to generate predictions.parquet, followed by analysis.
- **Early Stopping**: Early stopping is configured with patience=8 and min_delta=1e-3.
- **Training**: The training process utilizes AdamW, CosineLRScheduler, and AMP GradScaler.
- **Data Splitting**: The dataset splitting strategy uses per-class shuffle-and-slice with an 8:1:1 ratio.
- **Clobber-prevention**: The training CLI prevents overwriting existing run_ids unless the --resume flag is used.
- **Model**: The model architecture uses timm Swin-Tiny with LLRD decay of 0.75.
- **Environment**: The system supports both Kaggle and local environments using the same code path via --images-root and --output-dir.
