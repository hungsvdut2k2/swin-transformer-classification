---
title: Foodnet Package Architecture
summary: 'Foodnet package architecture organized by capability: splitting, data, models, training, evaluation, cli, and utils.'
tags: []
related: []
keywords: []
createdAt: '2026-05-07T13:29:01.215Z'
updatedAt: '2026-05-07T13:29:01.215Z'
---
## Reason
Curate package structure and capability-based architectural patterns

## Raw Concept
**Task:**
Document foodnet package architecture

**Changes:**
- Defined architectural structure
- Mapped capability layers

**Files:**
- foodnet/
- pyproject.toml

**Flow:**
cli orchestrates capability layers (splitting, data, models, training, eval, utils)

**Timestamp:** 2026-05-07

## Narrative
### Structure
The foodnet package is organized by capability: splitting/, data/, models/, training/, eval/, cli/, and utils/. The cli/ layer acts as a thin orchestrator.

### Highlights
Capability-based architecture; thin CLI orchestration; clean separation of concerns.

### Rules
CLI layer must not own business logic; it wires together pure functions from other layers.

## Facts
- **architecture**: foodnet package organized by capability layers [project]
- **cli_role**: CLI layer is a thin orchestrator [project]
