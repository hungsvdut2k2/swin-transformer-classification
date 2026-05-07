- The project implements a Swin Transformer (timm Swin-Tiny) for image classification on the Food-101 dataset.
- Development follows a strict phased lifecycle: Data Understanding, Preprocessing, Baseline Modeling, and Evaluation.
- Reproducibility is prioritized through a fixed seed (42), SHA256 tracking of metadata, and explicit CLI entry points.
- Training utilizes advanced optimization techniques including AdamW, CosineLRScheduler, AMP GradScaler, and LLRD decay (0.75).
- The system supports environment flexibility (Kaggle vs. local) and includes safety mechanisms like clobber-prevention for run IDs.

Structure / Sections Summary:
- Project Structure: Organized into directories for documentation, data, notebooks, and splits.
- Narrative: Defines the phased development approach and core model architecture.
- Facts: Details technical specifications, including the foodnet package, CLI commands, early stopping parameters (patience=8), and data splitting ratios (8:1:1).

Notable Entities, Patterns, and Decisions:
- Entities: Swin-Tiny (timm), Food-101 dataset, AdamW, CosineLRScheduler.
- Patterns: Phased development lifecycle, per-class shuffle-and-slice splitting, two-stage evaluation (inference followed by analysis).
- Decisions: Enforcing reproducibility via fixed seeds and metadata hashing; preventing accidental overwrites of training runs via CLI flags.