# Keep continuation value maps planner-owned

CueRouter will represent continuation intent with a planner-side `PositionValueMap`, not a `core.geometry.PositionTarget`. The map stores one final future-position heatmap per candidate next shot, selects the highest-scoring candidate heatmap or configurable top-k heatmap set, and exports that same scalar `HeatmapTarget` to executor search; this keeps planner score and executor target aligned while preserving the planner/executor boundary.

## Consequences

- Executor code must not read next-shot identities or planner continuation metadata.
- `PositionValueMap` owns candidate heatmap selection, top-k combination, and thresholding before exporting a `HeatmapTarget`.
- Planner scoring and executor targeting use the same selected candidate heatmap score.
