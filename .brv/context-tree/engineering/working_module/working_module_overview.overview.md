Key Points:
- SplitsDataset maintains a constant dataset length even when encountering corrupt or missing image files.
- Corrupt files are handled via recursive resampling, with errors caught during PIL/IO operations.
- A class-level set (_warned) is utilized to prevent redundant logging of unreadable file warnings.
- The run_inference function enforces strict alignment between input filepaths and inference rows.
- Inference results, including softmax probabilities and losses, are persisted in a single-pass parquet file.

Structure / Sections Summary:
- Reason: Defines the objective of curating RLM context.
- Raw Concept: Outlines the task, changes, and timestamp.
- Narrative: Provides a high-level summary of the module's structure and purpose.
- Facts: Details specific technical behaviors of the SplitsDataset and run_inference components.

Notable Entities, Patterns, or Decisions:
- Entities: SplitsDataset (class), run_inference (function).
- Patterns: Recursive resampling for data robustness; class-level state tracking for logging suppression.
- Decisions: Using the filepaths list as the single source of truth for dataset size (N) to ensure consistency.