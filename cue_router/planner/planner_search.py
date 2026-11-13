from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from cue_router.core.table import ChineseTableSpecs
from cue_router.planner.breakout_planner import BreakoutPlanner
from cue_router.planner.landing_simulation import LandingSimulator
from cue_router.planner.planner_evaluation import PlannerEvaluation, ShotOption
from cue_router.planner.position_value_map import ContinuationOption, PositionValueMap
from cue_router.planner.problem_ball_profile import ProblemBallAnalyzer
from cue_router.ui.layout import PlannerLayout

@dataclass
class PlannerConfig:
    display_k: int = 5
    lookahead_depth: int = 1
    top_k_future_candidates: int = 5
    target_heatmap_k: int = 2
    playable_threshold: float = 0.25
    option_bonus_weight: float = 0.10
    top_heatmap_percent: float = 0.15
    target_region_threshold: float = 0.0
    future_position_contrast: float = 1.1
    max_cut_angle_deg: float = 80.0
    grid_m: int = 60
    grid_n: int = 30
    pocket_mask_radius: float = 1.5 * ChineseTableSpecs.BALL_RADIUS
    cue_target_distance_free: float = 0.25      # cue ball positioning distance penalty
    cue_target_distance_softness: float = 0.55
    cue_target_distance_min_quality: float = 0.75
    normal_position_weight: float = 1.0
    breakout_position_weight: float = 1.8
    breakout_shot_weight: float = 5.0
    problem_priority_weight: float = 1.4
    breakout_problem_priority_threshold: float = 0.3   # min priority value for a problem ball to be considered dead
    breakout_problem_top_k: int = 6
    breakout_helper_top_k: int = 10
    breakout_shot_min_pot_quality: float = 0.1
    breakout_shot_min_geometry_quality: float = 0.1
    breakout_position_threshold: float = 0.0
    breakout_position_robustness_threshold: float = 0.25
    breakout_position_robustness_floor: float = 0.2
    clear_problem_weight: float = 0.3
    remaining_problem_weight: float = 0.8
    table_simplification_min: float = 0.2
    table_simplification_max: float = 1.25
    bank_shot_cap: float = 0.5
    bank_route_weight: float = 0.88
    continuity_bonus_weight: float = 0.30
    continuity_heatmap_bonus_weight: float = 0.5
    continuity_min_factor: float = 0.85
    continuity_max_factor: float = 1.40
    cue_landing_tolerance: float = 0.15
    landing_support_floor: float = 0.25


@dataclass(frozen=True)
class PlannerContext:
    previous_shot: ShotOption | None = None
    intended_next_shots: tuple[ContinuationOption, ...] = ()
    confidence: float = 1.0
    breakout_stage: str | None = None
    breakout_target_ball_id: str | None = None

    @classmethod
    def from_option(cls, option):
        intended_next_shots = option.score_breakdown.get("intended_next_shots", ())
        if intended_next_shots is None:
            intended_next_shots = ()
        confidence = cls._scalar_confidence(option.score_breakdown.get("future_position_value", 1.0))
        return cls(
            previous_shot=option,
            intended_next_shots=tuple(intended_next_shots),
            confidence=confidence,
        )

    @staticmethod
    def _scalar_confidence(value):
        values = np.asarray(value, dtype=float)
        if values.size == 0:
            return 0.0
        return float(np.clip(np.mean(values), 0.0, 1.0))


class PlannerSearch:
    EIGHT_BALL_ID = "8"

    def __init__(self, config=None):
        self.config = config or PlannerConfig()
        self.problem_profiles = {}

    def search(self, layout, legal_target_balls, display_k=None, context=None):
        if not isinstance(layout, PlannerLayout):
            raise TypeError(f"layout must be PlannerLayout, got {type(layout)!r}")
        cue_xy = PlannerEvaluation._xy(layout.cue_ball_pos)
        balls = {
            str(ball_id): PlannerEvaluation._xy(pos)
            for ball_id, pos in layout.balls_info.items()
        }
        legal_targets = [ball_id for ball_id in legal_target_balls if ball_id in balls]
        self.problem_profiles = ProblemBallAnalyzer.analyze(
            balls,
            legal_targets,
        )
        for key, value in self.problem_profiles.items():
            print(f"ball #{key}")
            print(value)
        table_simplification_factors = {
            ball_id: self._table_simplification_factor(ball_id, legal_targets, self.problem_profiles)
            for ball_id in legal_targets
        }
        options = []
        options.extend(BreakoutPlanner.build_breakout_shot_options(
            cue_xy,
            balls,
            legal_targets,
            self.problem_profiles,
            self.config,
            table_simplification_factors=table_simplification_factors,
        ))

        for target_ball_id in legal_targets:
            future_balls = {ball_id: xy for ball_id, xy in balls.items() if ball_id != target_ball_id}
            future_legal_targets = self._future_legal_targets(
                balls,
                legal_targets,
                target_ball_id,
            )
            target_xy = balls[target_ball_id]
            value_map = self._build_position_value_map(future_balls, future_legal_targets, target_xy)
            for target_pocket_id in ChineseTableSpecs.POCKET_ID:
                option = self._build_shot_option(
                    cue_xy,
                    balls,
                    legal_targets,
                    target_ball_id,
                    target_pocket_id,
                    value_map=value_map,
                    problem_profile=self.problem_profiles.get(target_ball_id),
                    context=context,
                )
                if option is not None:
                    option = self._reevaluate_shot_option(
                        option=option,
                        balls=balls,
                        position_map=value_map,
                        legal_targets=legal_targets,
                        context=context,
                        cue_xy=cue_xy
                    )
                    options.append(option)
                    options.extend(BreakoutPlanner.build_breakout_position_options(
                        cue_xy,
                        balls,
                        legal_targets,
                        self.problem_profiles,
                        target_ball_id,
                        target_pocket_id,
                        option.score_breakdown["current_pot_quality"],
                        self.config,
                        table_simplification_factors=table_simplification_factors,
                    ))
                # else:
                options.extend(self._build_bank_shot_options(
                    cue_xy,
                    balls,
                    legal_targets,
                    target_ball_id,
                    target_pocket_id,
                    value_map=value_map,
                    problem_profile=self.problem_profiles.get(target_ball_id),
                    context=context,
                ))

        k = self.config.display_k if display_k is None else display_k
        return sorted(options, key=lambda option: option.score, reverse=True)[:k]

    def _build_shot_option(
            self,
            cue_xy,
            balls,
            legal_targets,
            target_ball_id,
            target_pocket_id,
            value_map=None,
            problem_profile=None,
            context=None,
    ):
        target_xy = balls[target_ball_id]
        if not PlannerEvaluation.is_direct_candidate(
                cue_xy,
                target_xy,
                target_pocket_id,
                obstacles=balls,
                max_cut_angle_deg=self.config.max_cut_angle_deg,
        ):
            return None

        current_pot_quality = PlannerEvaluation.evaluate_shot_ease(
            cue_xy,
            target_xy,
            target_pocket_id,
            obstacles=balls,
        )
        target_region, future_position_value, intended_next_shot = value_map.select_best_target_heatmap(
            threshold=self.config.target_region_threshold,
            top_k=self.config.target_heatmap_k,
            top_fraction=self.config.top_heatmap_percent,
            preferred_options=self._context_intended_next_shots(context),
            preferred_weight=self._context_heatmap_bonus_weight(context),
        )
        table_simplification_factor = self._table_simplification_factor(
            target_ball_id,
            legal_targets,
            self.problem_profiles,
        )
        base_score = current_pot_quality * future_position_value * self.config.normal_position_weight
        continuity_factor = self._continuity_factor(
            target_ball_id,
            target_pocket_id,
            "direct",
            None,
            context,
        )
        score = base_score * table_simplification_factor * continuity_factor
        intended_next_shot_label = None
        if len(intended_next_shot) > 0:
            intended_next_shot_label = ", ".join(option.label() for option in intended_next_shot)

        return ShotOption(
            target_ball_id=target_ball_id,
            target_pocket_id=target_pocket_id,
            target_region=target_region,
            score=score,
            route_type="direct",
            option_family="normal_position",
            score_breakdown={
                "option_family": "normal_position",
                "current_pot_quality": current_pot_quality,
                "future_position_value": future_position_value,
                "target_region_value": future_position_value,
                "table_simplification_factor": table_simplification_factor,
                "continuity_factor": continuity_factor,
                "base_score": base_score,
                "intended_next_shot": intended_next_shot_label,
                "intended_next_shots": intended_next_shot,
                "cut_angle_deg": PlannerEvaluation.cut_angle_deg(cue_xy, target_xy, target_pocket_id),
                "problem_priority": 0.0 if problem_profile is None else problem_profile.problem_priority,
                "is_problem_ball": False if problem_profile is None else problem_profile.is_hard,
                "require_breakout": False if problem_profile is None else problem_profile.require_breakout,
                "problem_reasons": () if problem_profile is None else problem_profile.reasons,
            },
        )

    def _reevaluate_shot_option(self, option: ShotOption, balls, position_map, legal_targets, context, cue_xy):
        if not position_map.candidates:
            return option  # Terminal position: there is no continuation to condition.
        option = deepcopy(option)

        landing_map = self._build_landing_support_map(option, balls, cue_xy)
        combined_map = self._combine_value_maps(position_map, landing_map, eps=self.config.landing_support_floor)

        target_region, future_position_value, intended_next_shot = combined_map.select_best_target_heatmap(
            threshold=self.config.target_region_threshold,
            top_k=self.config.target_heatmap_k,
            top_fraction=self.config.top_heatmap_percent,
            preferred_options=self._context_intended_next_shots(context),
            preferred_weight=self._context_heatmap_bonus_weight(context),
        )

        current_pot_quality = option.score_breakdown["current_pot_quality"]
        base_score = current_pot_quality * future_position_value * self.config.normal_position_weight

        table_simplification_factor = self._table_simplification_factor(
            option.target_ball_id,
            legal_targets,
            self.problem_profiles,
        )
        continuity_factor = self._continuity_factor(
            option.target_ball_id,
            option.target_pocket_id,
            option.route_type,
            None,
            context,
        )

        score = base_score * table_simplification_factor * continuity_factor
        intended_next_shot_label = None
        if len(intended_next_shot) > 0:
            intended_next_shot_label = ", ".join(option.label() for option in intended_next_shot)

        return ShotOption(
            target_ball_id=option.target_ball_id,
            target_pocket_id=option.target_pocket_id,
            target_region=target_region,
            score=score,
            route_type=option.route_type,
            option_family=option.option_family,
            score_breakdown=option.score_breakdown | {
                "option_family": "normal_position",
                "current_pot_quality": current_pot_quality,
                "future_position_value": future_position_value,
                "target_region_value": future_position_value,
                "table_simplification_factor": table_simplification_factor,
                "continuity_factor": continuity_factor,
                "base_score": base_score,
                "intended_next_shot": intended_next_shot_label,
                "intended_next_shots": intended_next_shot,
            },
        )

    def _build_bank_shot_options(
            self,
            cue_xy,
            balls,
            legal_targets,
            target_ball_id,
            target_pocket_id,
            value_map=None,
            problem_profile=None,
            context=None,
    ):
        target_xy = balls[target_ball_id]
        target_region, future_position_value, intended_next_shot = value_map.select_best_target_heatmap(
            threshold=self.config.target_region_threshold,
            top_k=self.config.target_heatmap_k,
            top_fraction=self.config.top_heatmap_percent,
            preferred_options=self._context_intended_next_shots(context),
            preferred_weight=self._context_heatmap_bonus_weight(context),
        )
        table_simplification_factor = self._table_simplification_factor(
            target_ball_id,
            legal_targets,
            self.problem_profiles,
        )
        intended_next_shot_label = None
        if len(intended_next_shot) > 0:
            intended_next_shot_label = ", ".join(option.label() for option in intended_next_shot)

        options = []
        for rail in PlannerEvaluation.BANK_RAILS:
            bank_quality = PlannerEvaluation.evaluate_bank_shot_ease(
                cue_xy,
                target_xy,
                target_pocket_id,
                rail,
                obstacles=balls,
                cap=self.config.bank_shot_cap,
            )
            if bank_quality <= 0:
                continue

            bank_geometry = PlannerEvaluation.bank_geometry(
                cue_xy,
                target_xy,
                target_pocket_id,
                rail,
                obstacles=balls,
            )
            base_score = (
                bank_quality
                * future_position_value
                * self.config.normal_position_weight
                * self.config.bank_route_weight
            )
            continuity_factor = self._continuity_factor(
                target_ball_id,
                target_pocket_id,
                "bank",
                rail,
                context,
            )
            score = base_score * table_simplification_factor * continuity_factor
            options.append(ShotOption(
                target_ball_id=target_ball_id,
                target_pocket_id=target_pocket_id,
                target_region=target_region,
                score=score,
                route_type="bank",
                bank_rail=rail,
                option_family="normal_position",
                score_breakdown={
                    "option_family": "normal_position",
                    "current_pot_quality": bank_quality,
                    "future_position_value": future_position_value,
                    "target_region_value": future_position_value,
                    "table_simplification_factor": table_simplification_factor,
                    "continuity_factor": continuity_factor,
                    "bank_route_weight": self.config.bank_route_weight,
                    "bank_shot_cap": self.config.bank_shot_cap,
                    "base_score": base_score,
                    "intended_next_shot": intended_next_shot_label,
                    "intended_next_shots": intended_next_shot,
                    "cut_angle_deg": PlannerEvaluation.cut_angle_deg(cue_xy, target_xy, target_pocket_id),
                    "bank_rail": rail,
                    "bank_travel_distance": 0.0 if bank_geometry is None else bank_geometry["travel_distance"],
                    "problem_priority": 0.0 if problem_profile is None else problem_profile.problem_priority,
                    "is_problem_ball": False if problem_profile is None else problem_profile.is_hard,
                    "require_breakout": False if problem_profile is None else problem_profile.require_breakout,
                    "problem_reasons": () if problem_profile is None else problem_profile.reasons,
                },
            ))

        return options

    def _table_simplification_factor(self, target_ball_id, legal_targets, profiles):
        remaining = [ball_id for ball_id in legal_targets if ball_id != target_ball_id]
        n_remaining = len(remaining)

        target_profile = profiles.get(target_ball_id)
        cleared_priority = 0.0 if target_profile is None else target_profile.problem_priority

        if n_remaining == 0:
            remaining_pressure = 0.0
            urgency = 0.0
        else:
            remaining_pressure = sum(
                profiles[ball_id].problem_priority
                for ball_id in remaining
                if ball_id in profiles
            ) / n_remaining
            urgency = 1.0 / n_remaining

        clear_bonus = self.config.clear_problem_weight * cleared_priority
        pressure_penalty = self.config.remaining_problem_weight * remaining_pressure * urgency
        return float(np.clip(
            1.0 + clear_bonus - pressure_penalty,
            self.config.table_simplification_min,
            self.config.table_simplification_max,
        ))

    def _continuity_factor(self, target_ball_id, target_pocket_id, route_type, bank_rail, context):
        if context is None:
            return 1.0
        if not self._matches_context_intent(target_ball_id, target_pocket_id, route_type, bank_rail, context):
            return 1.0

        confidence = float(np.clip(context.confidence, 0.0, 1.0))
        factor = 1.0 + self.config.continuity_bonus_weight * confidence
        return float(np.clip(
            factor,
            self.config.continuity_min_factor,
            self.config.continuity_max_factor,
        ))

    def _matches_context_intent(self, target_ball_id, target_pocket_id, route_type, bank_rail, context):
        for option in self._context_intended_next_shots(context):
            if option.target_ball_id != target_ball_id:
                continue
            if option.target_pocket_id != target_pocket_id:
                continue
            if option.route_type != route_type:
                continue
            option_bank_rail = getattr(option, "bank_rail", None)
            if option_bank_rail is not None and option_bank_rail != bank_rail:
                continue
            return True
        return False

    def _context_intended_next_shots(self, context):
        if context is None:
            return ()
        return tuple(context.intended_next_shots)

    def _context_heatmap_bonus_weight(self, context):
        if context is None:
            return 0.0
        confidence = float(np.clip(context.confidence, 0.0, 1.0))
        return self.config.continuity_heatmap_bonus_weight * confidence

    def _build_position_value_map(self, balls, legal_targets, target_ball_pos):
        if self.config.lookahead_depth <= 0 or len(legal_targets) == 0:
            return self._terminal_value_map()

        candidates = [
            ContinuationOption(ball_id, pocket_id, route_type="direct")
            for ball_id in legal_targets
            if ball_id in balls
            for pocket_id in ChineseTableSpecs.POCKET_ID
        ]
        return PositionValueMap.from_grid_evaluator(
            self._future_position_value_vectorized,
            candidates=candidates,
            target_ball_pos=target_ball_pos,
            target_pocket_id=None,
            obstacles={
                "balls": balls,
                "legal_targets": legal_targets,
            },
            m=self.config.grid_m,
            n=self.config.grid_n,
            pocket_mask_radius=self.config.pocket_mask_radius,
            top_fraction=self.config.top_heatmap_percent,
        )

    def _build_landing_support_map(self, option: ShotOption, balls, cue_xy) -> PositionValueMap:
        return PositionValueMap.from_grid_evaluator(
            self._cue_landing_position_value_vectorized,
            cue_xy=cue_xy,
            balls=balls,
            option=option,
            candidates=(),
            target_ball_pos=balls[option.target_ball_id],
            target_pocket_id=option.target_pocket_id,
            m=self.config.grid_m,
            n=self.config.grid_n,
            pocket_mask_radius=self.config.pocket_mask_radius,
            top_fraction=self.config.top_heatmap_percent,
        )

    def _combine_value_maps(self, position_map: PositionValueMap, landing_map: PositionValueMap, eps=0.25) -> PositionValueMap:
        position_grid = position_map.value_grid
        landing_grid = landing_map.value_grid
        if landing_grid.shape != (1, *position_grid.shape[1:]):
            raise ValueError(f"Expected one support grid aligned with {position_grid.shape}, got {landing_grid.shape}")
        combined_grid = position_grid * (eps + (1 - eps) * landing_grid)
        return PositionValueMap(
            combined_grid,
            candidates=PositionValueMap._score_candidates(
                position_map.candidates, combined_grid, self.config.top_heatmap_percent,
            ),
            table_width=position_map.table_width,
            table_length=position_map.table_length,
            pocket_mask_radius=position_map.pocket_mask_radius,
        )

    def _terminal_value_map(self):
        return PositionValueMap.terminal(
            m=self.config.grid_m,
            n=self.config.grid_n,
            pocket_mask_radius=self.config.pocket_mask_radius,
        )

    def _future_position_value_vectorized(
            self,
            cue_X,
            cue_Y,
            target_ball_pos,
            target_pocket_id,
            candidates=(),
            obstacles=None,
            **kwargs,
    ):
        balls = obstacles["balls"]
        legal_targets = obstacles["legal_targets"]
        qualities = []

        for candidate in candidates:
            if candidate.target_ball_id not in balls:
                continue
            shot_quality = PlannerEvaluation.evaluate_position_grid(
                cue_X,
                cue_Y,
                balls[candidate.target_ball_id],
                candidate.target_pocket_id,
                obstacles=balls,
            )
            cut_angle_deg = PlannerEvaluation.cut_angle_deg_grid(
                cue_X,
                cue_Y,
                balls[candidate.target_ball_id],
                candidate.target_pocket_id,
            )
            problem_profile = self.problem_profiles.get(candidate.target_ball_id)
            priority_factor = 1.0 + (self.config.problem_priority_weight * problem_profile.problem_priority
                             if problem_profile is not None else 0)
            table_factor = self._table_simplification_factor(
                candidate.target_ball_id,
                legal_targets,
                self.problem_profiles,
            )
            quality = shot_quality * priority_factor * table_factor
            qualities.append(np.where(
                cut_angle_deg < self.config.max_cut_angle_deg,
                quality,
                0.0,
            ))

        if len(qualities) == 0:
            return np.zeros((0, *cue_X.shape), dtype=float)

        raw_value_grid = np.stack(qualities)
        k = min(self.config.top_k_future_candidates, raw_value_grid.shape[0])
        top_qualities = np.sort(raw_value_grid, axis=0)[-k:]
        best = top_qualities[-1]
        playable_count = np.sum(
            (top_qualities > 0) & (top_qualities >= self.config.playable_threshold),
            axis=0,
        )
        option_bonus = self.config.option_bonus_weight * np.log1p(np.maximum(0, playable_count - 1))
        distance_quality = self._cue_target_distance_quality_grid(cue_X, cue_Y, target_ball_pos)
        value_grid = np.where(
            raw_value_grid > 0,
            (np.clip(raw_value_grid + option_bonus, 0.0, 1.0) ** self.config.future_position_contrast)
            * distance_quality,
            0.0,
        )
        return value_grid

    def _cue_landing_position_value_vectorized(self, cue_X, cue_Y, *, option=None, balls=None, R=None, cue_xy=None,
                                               **kwargs):
        if R is None:
            R = self.config.cue_landing_tolerance
        landings = LandingSimulator.simulate(option, {**balls, "cue": cue_xy})

        if len(landings) == 0:
            return np.zeros_like(cue_X, dtype=float)

        # cue_X: [M, N]
        # cue_Y: [M, N]
        # landings: [S, 2]
        # distance_sq: [S, M, N]
        distance_sq = (
            (cue_X[None, :, :] - landings[:, 0, None, None]) ** 2
            + (cue_Y[None, :, :] - landings[:, 1, None, None]) ** 2
        )

        maps = np.exp(-np.log(2) * distance_sq / R**2)
        return np.max(maps, axis=0)


    def _cue_target_distance_quality_grid(self, cue_X, cue_Y, target_ball_pos):
        if target_ball_pos is None:
            return np.ones_like(cue_X, dtype=float)

        target_xy = PlannerEvaluation._xy(target_ball_pos)
        dist = np.sqrt((cue_X - target_xy[0]) ** 2 + (cue_Y - target_xy[1]) ** 2)
        excess = np.maximum(0.0, dist - self.config.cue_target_distance_free)
        decay = np.exp(-((excess / self.config.cue_target_distance_softness) ** 2))
        min_quality = self.config.cue_target_distance_min_quality
        return np.where(
            excess == 0.0,
            1.0,
            min_quality + (1.0 - min_quality) * decay,
        )

    def _future_legal_targets(self, balls, legal_targets, target_ball_id):
        future_legal_targets = [ball_id for ball_id in legal_targets if ball_id != target_ball_id]
        if len(future_legal_targets) > 0:
            return future_legal_targets
        if target_ball_id == self.EIGHT_BALL_ID:
            return []
        if len(legal_targets) == 1 and self.EIGHT_BALL_ID in balls:
            return [self.EIGHT_BALL_ID]
        return []
