---
title: Working Module Overview
summary: Overview of working module structure, components, and current status
tags: []
related: []
keywords: []
createdAt: '2026-05-07T13:28:49.314Z'
updatedAt: '2026-05-07T13:29:19.668Z'
---
## Reason
Curate working module knowledge from RLM context

## Raw Concept
**Task:**
Document Working Module

**Changes:**
- Consolidated working module findings
- Extracted module overview

**Timestamp:** 2026-05-07

## Narrative
### Structure
Working module comprises core components for analysis, evaluation, and training.

### Highlights
Centralized module structure for project phases.

## Facts
- **SplitsDataset**: The SplitsDataset class handles corrupt or missing image files by catching PIL/IO errors and recursively resampling a random index from the dataset.
- **SplitsDataset**: SplitsDataset logs warnings for unreadable files only once per file path using a class-level set named _warned.
- **SplitsDataset**: The length of the SplitsDataset remains unchanged even when corrupt images are encountered.
- **run_inference**: The run_inference function requires that the length of filepaths matches the number of inference rows to ensure data alignment in the output parquet file.
- **run_inference**: The run_inference function performs single-pass inference and saves per-example softmax probabilities and losses to a parquet file.
- **SplitsDataset**: SplitsDataset uses the filepaths list as the authoritative source for the dataset size (N).
