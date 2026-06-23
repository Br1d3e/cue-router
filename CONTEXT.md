# CueRouter Project Context

CueRouter is a billiards / Chinese 8-ball shot planning tool. The goal is to recommend a high-level pool “思路” and then compute an executable cue action using `pooltool`.

Architectural split:

* `planner/` = brain / 思路 layer.

  * Input: full table state `S` with all ball positions.
  * Output: high-level `SearchRequest`: target ball, target pocket, route type, and cue-ball target region / line / future-position target.
  * Planner should evaluate strategic value: next-shot availability, ball order, angle preservation, fallback options, route risk, and future table value.
  * Planner should not output low-level cue parameters.

* `executor/` = cue / hand layer.

  * Input: a `SearchRequest` produced by the planner.
  * Output: `ShotAction(phi, V0, a, b)`.
  * Executor searches physical cue parameters using stochastic search, grid search, pattern search, and pooltool simulation.
  * Executor evaluator scores execution quality: target potted, cue-ball reaches target, scratch/foul avoidance, route cleanliness, action ease, and robustness.
  * Executor should not choose the strategic target ball/pocket/position idea.

Current state:

* Executor is mostly implemented.
* Existing executor modules include `action.py`, `evaluation.py`, `search.py`, `geometry.py`, `table.py`, `visual.py`, and `search_method/`.
* `ShotAction(phi, V0, a, b)` represents cue direction, speed, side spin, and top/back spin.
* `PolicySearch` can run grid/stochastic search, pattern refinement, and robustness reranking.
* Stochastic search is now faster than grid and should remain deterministic when a seed is provided.
* `ShotEvaluation` already tracks score breakdowns such as position quality, target route quality, cue action ease, cue route quality, near-miss quality, base score, and robustness.

Coding guidelines:

* Keep planner and executor strictly separated.
* Do not let executor decide strategic shot choice.
* Do not let planner depend on low-level cue parameters.
* Prefer small, targeted edits over large rewrites.
* Preserve current working executor behavior unless the task explicitly asks to refactor it.
* Use deterministic seeds for stochastic search during debugging.
* Avoid overengineering with HRL/MCTS for now; first build a deterministic planner with option generation, position value maps, and beam-style lookahead.

## Glossary

* **Pot quality / shot ease**: A planner-side estimate of how playable a target ball and pocket are from a cue-ball position, considering pot geometry, visibility, and cueing room.
* **Position target**: The planner's desired future cue-ball location, represented as a region, line, or table-wide quality map.
* **Heatmap position target**: A position target that assigns a quality score to each cue-ball location across the table. Executor-facing heatmaps zero out weak cells below a planner threshold.
* **Legal target balls**: The set of balls the planner is allowed to consider for the current turn, supplied by upstream game-state/rules logic rather than derived inside the planner.
* **Planner score**: A heuristic score for a shot idea before executor verification. For non-terminal shots, it combines current pot quality and future position value multiplicatively so weak continuation can outweigh an easy pot. Terminal shots use future position value `1.0`.
* **Next-shot value heatmap**: A heatmap position target whose score at each cue-ball location estimates the best available next shot from that location. It excludes table-geometry impossible locations such as out-of-table points and pocket interiors.
* **Continuation value map**: A planner-side value map that stores one final future-position heatmap for each candidate next shot. It is not an executor-facing position target; it selects one or more top candidate heatmaps and exports their combined scalar `HeatmapTarget` when executor search needs a target region.
* **Candidate future-position heatmap**: The final cue-ball landing value grid for one candidate next shot, including shot geometry, option richness, contrast, and cue-ball positioning distance ease.
* **Continuation identity**: The intended next-shot option or top-k option set whose candidate future-position heatmap is selected as the current shot's executor target. It is planner metadata used for continuity and visualization, not data the executor reads.
* **Problem ball**: A legal target ball whose current table context makes it strategically urgent to solve before the endgame.
* **Dead ball**: A problem ball with no currently available direct potting pocket.
* **Hard ball**: A problem ball with at least one available direct potting pocket but low pot quality or awkward cushion proximity.
* **Development shot**: A shot whose strategic value includes improving or opening a problem ball, not only potting the current target.
* **Breakout objective**: An executor search objective for a development shot. The current target ball still must be first-hit legal and potted in the requested pocket, but the cue-ball objective is to disturb one dead problem ball rather than land at a final position target.
* **Breakout target**: The executor-facing target for a breakout objective. It names exactly one dead ball to develop and may include an internal path region that guides the cue ball through an area without requiring it to finish there.
* **Helper ball**: A non-problem legal target ball selected as the object ball for preparing or executing a breakout on a high-priority problem ball.
* **Breakout position option**: A planner option that pots a current ball and leaves the cue ball in a region that creates a future natural breakout angle on a selected helper ball.
* **Breakout shot option**: A planner option that pots a helper ball now and asks executor search to develop a specific dead ball through a breakout objective.
* **Breakout geometry quality**: A planner-side estimate of whether cue ball position, helper ball, pocket, and dead ball geometry form a natural development angle. It compares the helper ball's object-to-pocket tangent direction against the helper-to-dead-ball direction, cut-angle window, distance quality, and simple clearance.
* **Direct breakout collision**: A pooltool event-chain outcome where the cue ball directly collides with the breakout target ball after first legally hitting the current target ball.
* **Indirect breakout collision**: A pooltool event-chain outcome where collisions caused after the legal first hit eventually reach the breakout target ball, including chains such as cue ball to target ball to dead ball. Indirect collision is valid for breakout scoring when allowed by the breakout target.
* **Breakout path region**: An internal `PositionTarget` on a breakout target that scores whether the cue-ball path passes through a useful development area. It is not a final cue-ball landing target.
* **Running side**: A breakout sampling preference for side spin that increases the cue ball's reflection angle after cushion contact so the cue ball escapes the developed area. It is a search bias, not an executor scoring term in the first breakout implementation.
* **Aiming pocket position**: The pocket target point used by potting-angle heuristics. In `ChineseTableSpecs.POCKET_POS`, these are fake ghost-ball aim targets, not physical pocket centers.
* **Physical pocket area**: The actual pocket center and radius from the pooltool table object. Heatmap validity checks use this area to reject cue-ball locations inside pockets.
* **Legal cue-ball center area**: The area inside the table where the cue ball's center can exist, excluding cushion-overlap edges and physical pocket interiors.
* **Option richness**: Extra planner value from having multiple playable next shots, beyond the single best next shot. It uses a diminishing-returns bonus so the second playable option matters more than later options.
* **Target-region robustness**: Planner value from a broad high-quality cue-ball landing area. Shot options use the mean of the top heatmap cells rather than only the single best cell.
* **Cut-angle pruning**: Planner skips a target ball and pocket candidate when the cue-ball-to-object and object-to-pocket angle is too thin. The initial cutoff is `85` degrees.
* **Shot visibility**: Planner line-of-sight checks ignore the cue ball and target ball as obstacles. Other balls block cue-to-target and target-to-pocket paths.
* **Route type**: The path family for the target ball. Planner v1 enumerates only direct routes but keeps `route_type` on shot options and search requests for future bank-shot expansion.
* **Lookahead depth**: How many future turns the planner estimates beyond the current shot. Planner search keeps this configurable and starts with default depth `1`.
* **Future candidate cap**: Planner future-value evaluation considers visible direct next-shot candidates and caps them to the top five candidates by pot quality before applying option richness.
* **Planner config**: The dataclass of tunable planner heuristics, including heatmap size, lookahead depth, candidate caps, cut-angle threshold, and option-richness weights.
* **Physical table state**: The current cue-ball and object-ball positions independent of rules legality. `PoolLayout` is the current UI-oriented representation and may later migrate to a `(cue_xy, balls_info)` shape.
* **Planner ball map**: The planner's normalized physical-state shape: `cue_xy` plus a mapping from object-ball IDs to positions, excluding the cue ball.
* **Executor layout**: A target-specific UI layout for executor search, containing cue ball, chosen target ball, target pocket, route type, other balls, and target region.
* **Planner layout**: A physical table-state UI layout for planner search, containing cue ball and object-ball positions only.
* **Planner layout import**: A planner layout can be built from a pooltool shot/system so planner search can consume active table state through the same stable layout interface. Pocketed object balls are excluded.
* **Shot option list**: The planner returns a ranked top-K list of shot options so downstream execution can try fallbacks when the best planner idea is not executable.
* **Shot option**: A lightweight planner intent containing target ball, target pocket, route type, position target, heuristic score, and planner score breakdown. It does not own a pooltool system or cue action.
* **Search request conversion**: A shot option can create a `SearchRequest` when given the current shot/system, but it does not store that system itself.
* **No planner option**: If no valid direct shot option exists, planner search returns an empty option list.
