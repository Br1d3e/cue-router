# Keep planner search pure

Planner search generates strategic shot intent only: ranked `ShotOption`s and executor-ready `SearchRequest`s. It must not call executor search, inspect low-level `ShotAction`s, or use executor simulation as an oracle, because those responsibilities belong to the executor and a future coordinator can combine planner ranking with executor verification without collapsing the boundary.

## Consequences

- Planner scores are heuristic intent scores, not guaranteed execution success.
- Executor remains responsible for physical cue-parameter search and simulation.
- If executor verification is needed for final ranking, it happens outside both planner and executor in a coordinator layer.
