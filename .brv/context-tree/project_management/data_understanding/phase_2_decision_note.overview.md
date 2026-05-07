- Established strict reproducibility invariants for the project.
- Locked the random split seed at 42 to ensure consistency.
- Implemented a "Test Set Policy" prohibiting access to test data until Phase 7.
- Defined a clear separation of concerns: Phase 2 generates artifacts but does not act upon them.
- Standardized the use of relative filepaths for split CSVs.

Structure:
- Reason: Documentation of design decisions.
- Raw Concept: Workflow steps (setup through summary) and file references.
- Narrative: Summary of execution strategy and reproducibility rules.
- Rules: Explicit constraints on test set usage and seeding.
- Facts: Strategy, random seed, and test set policy.

Notable Entities/Patterns:
- Author: meowso.
- Strategy: Single notebook execution for all data understanding steps.
- Constraint: Test set preserved until Phase 7.
- Convention: Split seed = 42.