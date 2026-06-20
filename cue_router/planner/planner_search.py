from dataclasses import dataclass

import numpy as np

from cue_router.core.geometry import HeatmapTarget
from cue_router.core.table import ChineseTableSpecs
from cue_router.planner.planner_evaluation import PlannerEvaluation, ShotOption
from cue_router.ui.layout import PlannerLayout

@dataclass
class PlannerConfig:
    display_k: int = 5
    lookahead_depth: int = 1
    top_k_future_candidates: int = 5
    playable_threshold: float = 0.25
    option_bonus_weight: float = 0.05
    top_heatmap_percent: float = 0.05
    target_region_threshold: float = 0.4
    future_position_contrast: float = 2.0
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

    def search(self, layout, legal_target_balls, display_k=None):
        if not isinstance(layout, PlannerLayout):
            raise TypeError(f"layout must be PlannerLayout, got {type(layout)!r}")
        cue_xy = PlannerEvaluation._xy(layout.cue_ball_pos)
        balls = {
            str(ball_id): PlannerEvaluation._xy(pos)
            for ball_id, pos in layout.balls_info.items()
        }
        legal_targets = [ball_id for ball_id in legal_target_balls if ball_id in balls]
        options = []

        for target_ball_id in legal_targets:
            for target_pocket_id in ChineseTableSpecs.POCKET_ID:
                option = self._build_shot_option(
                    cue_xy,
                    balls,
                    legal_targets,
                    target_ball_id,
                    target_pocket_id,
                )
                if option is not None:
                    options.append(option)

        k = self.config.display_k if display_k is None else display_k
        return sorted(options, key=lambda option: option.score, reverse=True)[:k]

    def _build_shot_option(self, cue_xy, balls, legal_targets, target_ball_id, target_pocket_id):
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
        future_balls = {ball_id: xy for ball_id, xy in balls.items() if ball_id != target_ball_id}
        future_legal_targets = self._future_legal_targets(
            balls,
            legal_targets,
            target_ball_id,
        )
        heatmap = self._build_future_heatmap(future_balls, future_legal_targets, target_xy)
        future_position_value = heatmap.mean_top_fraction(self.config.top_heatmap_percent)
        score = current_pot_quality * future_position_value

        return ShotOption(
            target_ball_id=target_ball_id,
            target_pocket_id=target_pocket_id,
            target_region=heatmap.thresholded(self.config.target_region_threshold),
            score=score,
            route_type="direct",
            score_breakdown={
                "current_pot_quality": current_pot_quality,
                "future_position_value": future_position_value,
                "target_region_value": future_position_value,
                "cut_angle_deg": PlannerEvaluation.cut_angle_deg(cue_xy, target_xy, target_pocket_id),
            },
        )

    def _build_future_heatmap(self, balls, legal_targets, target_ball_pos):
        if self.config.lookahead_depth <= 0 or len(legal_targets) == 0:
            return self._terminal_heatmap()

        return HeatmapTarget.from_evaluator(
            self._future_position_value,
            target_ball_pos=target_ball_pos,
            target_pocket_id=None,
            obstacles={
                "balls": balls,
                "legal_targets": legal_targets,
            },
            m=self.config.grid_m,
            n=self.config.grid_n,
            pocket_mask_radius=self.config.pocket_mask_radius,
        )

    def _terminal_heatmap(self):
        return HeatmapTarget.from_evaluator(
            lambda cue_xy, target_ball_pos, target_pocket_id, obstacles=None: 1.0,
            target_ball_pos=None,
            target_pocket_id=None,
            m=self.config.grid_m,
            n=self.config.grid_n,
            pocket_mask_radius=self.config.pocket_mask_radius,
        )

    def _future_position_value(self, cue_xy, target_ball_pos, target_pocket_id, obstacles=None):
        balls = obstacles["balls"]
        legal_targets = obstacles["legal_targets"]
        qualities = []

        for ball_id in legal_targets:
            if ball_id not in balls:
                continue
            for pocket_id in ChineseTableSpecs.POCKET_ID:
                if not PlannerEvaluation.is_direct_candidate(
                        cue_xy,
                        balls[ball_id],
                        pocket_id,
                        obstacles=balls,
                        max_cut_angle_deg=self.config.max_cut_angle_deg,
                ):
                    continue
                shot_ease = PlannerEvaluation.evaluate_position(
                    cue_xy,
                    balls[ball_id],
                    pocket_id,
                    obstacles=balls,
                )
                qualities.append(shot_ease)

        if len(qualities) == 0:
            return 0.0

        qualities = sorted(qualities, reverse=True)[:self.config.top_k_future_candidates]
        best = qualities[0]
        playable_count = sum(q >= self.config.playable_threshold for q in qualities)
        option_bonus = self.config.option_bonus_weight * np.log1p(max(0, playable_count - 1))
        position_value = np.clip(best + option_bonus, 0.0, 1.0) ** self.config.future_position_contrast
        distance_quality = self._cue_target_distance_quality(cue_xy, target_ball_pos)
        return float(position_value * distance_quality)

    def _cue_target_distance_quality(self, cue_xy, target_ball_pos):
        if target_ball_pos is None:
            return 1.0

        dist = np.linalg.norm(PlannerEvaluation._xy(cue_xy) - PlannerEvaluation._xy(target_ball_pos))
        excess = max(0.0, dist - self.config.cue_target_distance_free)
        if excess == 0.0:
            return 1.0

        decay = np.exp(-((excess / self.config.cue_target_distance_softness) ** 2))
        min_quality = self.config.cue_target_distance_min_quality
        return float(min_quality + (1.0 - min_quality) * decay)

    def _future_legal_targets(self, balls, legal_targets, target_ball_id):
        future_legal_targets = [ball_id for ball_id in legal_targets if ball_id != target_ball_id]
        if len(future_legal_targets) > 0:
            return future_legal_targets
        if target_ball_id == self.EIGHT_BALL_ID:
            return []
        if len(legal_targets) == 1 and self.EIGHT_BALL_ID in balls:
            return [self.EIGHT_BALL_ID]
        return []
