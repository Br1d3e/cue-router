from dataclasses import dataclass

import numpy as np

from cue_router.core.table import ChineseTableSpecs
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
    option_bonus_weight: float = 0.05
    top_heatmap_percent: float = 0.05
    target_region_threshold: float = 0.0
    future_position_contrast: float = 1.0
    max_cut_angle_deg: float = 80.0
    grid_m: int = 60
    grid_n: int = 30
    pocket_mask_radius: float = 1.5 * ChineseTableSpecs.BALL_RADIUS
    cue_target_distance_free: float = 0.45
    cue_target_distance_softness: float = 0.65
    cue_target_distance_min_quality: float = 0.75


class PlannerSearch:
    EIGHT_BALL_ID = "8"

    def __init__(self, config=None):
        self.config = config or PlannerConfig()
        self.problem_profiles = {}

    def search(self, layout, legal_target_balls, display_k=None):
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
        options = []

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
                )
                if option is not None:
                    options.append(option)

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
        )
        score = current_pot_quality * future_position_value
        intended_next_shot_label = None
        if len(intended_next_shot) > 0:
            intended_next_shot_label = ", ".join(option.label() for option in intended_next_shot)

        return ShotOption(
            target_ball_id=target_ball_id,
            target_pocket_id=target_pocket_id,
            target_region=target_region,
            score=score,
            route_type="direct",
            score_breakdown={
                "current_pot_quality": current_pot_quality,
                "future_position_value": future_position_value,
                "target_region_value": future_position_value,
                "intended_next_shot": intended_next_shot_label,
                "cut_angle_deg": PlannerEvaluation.cut_angle_deg(cue_xy, target_xy, target_pocket_id),
                "problem_priority": 0.0 if problem_profile is None else problem_profile.problem_priority,
                "is_problem_ball": False if problem_profile is None else problem_profile.is_hard,
                "require_breakout": False if problem_profile is None else problem_profile.require_breakout,
                "problem_reasons": () if problem_profile is None else problem_profile.reasons,
            },
        )

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
    ):
        balls = obstacles["balls"]
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
            qualities.append(np.where(
                cut_angle_deg < self.config.max_cut_angle_deg,
                shot_quality,
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
