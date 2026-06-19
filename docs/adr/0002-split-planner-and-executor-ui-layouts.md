# Split planner and executor UI layouts

Planner search and executor search need different UI inputs, so CueRouter keeps separate layout types and selectors for them. Executor layout captures a fully specified shot intent for cue-action search, while planner layout captures physical table state only; this prevents planner-facing code from depending on target-pocket-region fields that belong to executor requests.

## Consequences

- Planner search consumes `PlannerLayout` as its stable table-state interface.
- Executor search consumes `ExecutorLayout` and converts it to `SearchRequest`.
- Shared table drawing stays separate from selector state machines so planner and executor selectors can evolve independently.
