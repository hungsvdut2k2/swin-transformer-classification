- Implemented CUDA gating for `AMP GradScaler` to ensure CPU compatibility.
- `GradScaler.is_enabled()` assertions are now gated by `torch.cuda.is_available()`.
- Prevents runtime crashes in non-CUDA/CPU-only environments.
- Ensures test suite portability across different hardware configurations.
- Applies to `foodnet/training/optim.py` and associated tests.

Structure:
- Reason: Documentation of the CUDA gating fix.
- Raw Concept: Task details, file changes, and implementation flow.
- Narrative: Explanation of the fix, dependencies, and coding rules.
- Facts: Technical behavior of GradScaler and portability conventions.

Notable Entities/Patterns:
- Component: `torch.cuda.amp.GradScaler`.
- Pattern: Conditional gating of hardware-specific assertions.
- Decision: Prioritize cross-platform portability for training utilities.