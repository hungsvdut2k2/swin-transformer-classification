---
title: Pytest Slow Marker Registration
summary: Pytest slow marker must be registered in pyproject.toml to avoid PytestUnknownMarkWarning.
tags: []
related: []
keywords: []
createdAt: '2026-05-07T13:28:51.685Z'
updatedAt: '2026-05-07T13:28:51.685Z'
---
## Reason
Documenting pytest slow marker registration requirement for silencing PytestUnknownMarkWarning.

## Raw Concept
**Task:**
Document Pytest slow marker registration

**Changes:**
- Registered slow marker in pyproject.toml to silence warning

**Files:**
- pyproject.toml

**Timestamp:** 2026-05-07

## Narrative
### Structure
The slow marker is registered under [tool.pytest.ini_options] in pyproject.toml.

### Highlights
Prevents PytestUnknownMarkWarning during test execution.

### Rules
Register all custom markers in pyproject.toml under [tool.pytest.ini_options].

## Facts
- **pytest_marker**: When using @pytest.mark.slow, the marker must be registered in pyproject.toml. [convention]
