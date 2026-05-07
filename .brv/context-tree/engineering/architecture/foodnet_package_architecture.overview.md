- Organizes the "foodnet" package into distinct capability-based layers.
- Implements a "thin CLI" pattern where the CLI layer acts only as an orchestrator.
- Enforces a strict separation of concerns by keeping business logic out of the CLI layer.
- Defines core modules: splitting, data, models, training, evaluation, cli, and utils.
- Ensures modularity by wiring together pure functions from underlying layers.

Structure:
- Reason and Raw Concept: Defines the architectural goal and package structure.
- Narrative: Describes the capability-based organization and CLI orchestration rules.
- Facts: Summarizes the architectural design and the specific role of the CLI.

Notable Entities/Patterns:
- Capability Layers: splitting/, data/, models/, training/, eval/, cli/, utils/.
- Pattern: Thin CLI orchestration (CLI must not own business logic).