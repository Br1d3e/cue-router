from dataclasses import dataclass, field

import numpy as np

from cue_router.core.table import ChineseTableSpecs
from cue_router.planner.planner_evaluation import PlannerEvaluation


@dataclass(frozen=True)
class ProblemBallProfile:
    ball_id: str
    best_shot_ease: float
    best_pocket_id: str | None
    available_pockets: int
    cue_access_samples: int
    nearest_available_pocket_distance: float
    nearest_cushion_distance: float
    near_long_rail: bool
    near_short_rail: bool
    is_dead: bool
    is_hard: bool
    require_breakout: bool
    problem_priority: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


class ProblemBallAnalyzer:
    NEAR_CUSHION_DISTANCE = 0.08
    HARD_SHOT_EASE = 0.45
    GHOST_LATERAL_OFFSET = 0.05
    GHOST_CUE_DISTANCES = (
        ChineseTableSpecs.BALL_RADIUS,
        2 * ChineseTableSpecs.BALL_RADIUS + 0.02,
        0.2,
        0.4,
    )
    CUSHION_POCKET_HARD_SEVERITY = 0.25

    @classmethod
    def analyze(
            cls,
            balls,
            legal_target_balls,
    ):
        return {
            ball_id: cls.analyze_ball(
                ball_id,
                balls,
            )
            for ball_id in legal_target_balls
            if ball_id in balls
        }

    @classmethod
    def analyze_ball(
            cls,
            ball_id,
            balls,
    ):
        ball_xy = PlannerEvaluation._xy(balls[ball_id])
        best_shot_ease = 0.0
        best_pocket_id = None
        available_pockets = 0
        cue_access_samples = 0
        nearest_available_pocket_distance = float("inf")

        for pocket_id in PlannerEvaluation.available_pocket_ids_for_target(ball_xy):
            pocket_xy = PlannerEvaluation._aim_pocket_xy(pocket_id)
            if not PlannerEvaluation.has_line_of_sight(
                    ball_xy,
                    pocket_xy,
                    balls,
                    exclude=(ball_xy,),
            ):
                continue

            nearest_available_pocket_distance = min(
                nearest_available_pocket_distance,
                float(np.linalg.norm(pocket_xy - ball_xy)),
            )
            pocket_best_shot_ease, pocket_cue_access_samples = cls._ghost_sample_pocket_access(
                ball_xy,
                pocket_id,
                balls,
            )
            cue_access_samples += pocket_cue_access_samples
            if pocket_cue_access_samples == 0:
                continue

            available_pockets += 1
            if pocket_best_shot_ease > best_shot_ease:
                best_shot_ease = pocket_best_shot_ease
                best_pocket_id = pocket_id

        nearest_cushion_distance = PlannerEvaluation._nearest_cushion_distance(ball_xy)
        cushion_pocket_severity = PlannerEvaluation.target_cushion_pocket_severity(
            ball_xy,
            nearest_available_pocket_distance,
        )
        near_long_rail, near_short_rail = cls._near_cushion_flags(ball_xy)
        is_dead = available_pockets == 0
        is_hard = (
            is_dead
            or best_shot_ease < cls.HARD_SHOT_EASE
            or cushion_pocket_severity >= cls.CUSHION_POCKET_HARD_SEVERITY
        )
        require_breakout = is_dead
        reasons = cls._reasons(
            is_dead,
            best_shot_ease,
            near_long_rail,
            near_short_rail,
            cushion_pocket_severity,
        )
        problem_priority = cls._problem_priority(
            best_shot_ease,
            available_pockets,
            is_dead,
            cushion_pocket_severity,
        )

        return ProblemBallProfile(
            ball_id=ball_id,
            best_shot_ease=float(best_shot_ease),
            best_pocket_id=best_pocket_id,
            available_pockets=available_pockets,
            cue_access_samples=cue_access_samples,
            nearest_available_pocket_distance=float(nearest_available_pocket_distance),
            nearest_cushion_distance=float(nearest_cushion_distance),
            near_long_rail=near_long_rail,
            near_short_rail=near_short_rail,
            is_dead=is_dead,
            is_hard=is_hard,
            require_breakout=require_breakout,
            problem_priority=problem_priority,
            reasons=reasons,
        )

    @classmethod
    def _ghost_sample_pocket_access(cls, ball_xy, pocket_id, balls):
        pocket_xy = PlannerEvaluation._aim_pocket_xy(pocket_id)
        object_to_pocket = pocket_xy - PlannerEvaluation._xy(ball_xy)
        v_op = np.linalg.norm(object_to_pocket)
        if v_op == 0:
            return 0.0, 0

        cue_axis = -object_to_pocket / v_op
        lateral_axis = np.asarray((-cue_axis[1], cue_axis[0]), dtype=float)
        best_shot_ease = 0.0
        cue_access_samples = 0
        for cue_distance in cls.GHOST_CUE_DISTANCES:
            for lateral_offset in (-cls.GHOST_LATERAL_OFFSET, 0.0, cls.GHOST_LATERAL_OFFSET):
                cue_xy = ball_xy + cue_axis * cue_distance + lateral_axis * lateral_offset
                if not ChineseTableSpecs.in_table(cue_xy):
                    continue
                if not cls._cue_position_clear(cue_xy, balls):
                    continue
                shot_ease = PlannerEvaluation.evaluate_shot_ease(
                    cue_xy,
                    ball_xy,
                    pocket_id,
                    obstacles=balls,
                )
                if shot_ease <= 0:
                    continue
                cue_access_samples += 1
                best_shot_ease = max(best_shot_ease, shot_ease)

        return best_shot_ease, cue_access_samples

    @staticmethod
    def _cue_position_clear(cue_xy, balls):
        cue_xy = PlannerEvaluation._xy(cue_xy)
        min_distance = 2 * ChineseTableSpecs.BALL_RADIUS
        for ball_xy in balls.values():
            if np.linalg.norm(cue_xy - PlannerEvaluation._xy(ball_xy)) < min_distance:
                return False
        return True

    @classmethod
    def _near_cushion_flags(cls, xy):
        x, y = PlannerEvaluation._xy(xy)
        r = ChineseTableSpecs.BALL_RADIUS
        long_rail_distance = min(
            x - r,
            ChineseTableSpecs.TABLE_WIDTH - r - x,
        )
        short_rail_distance = min(
            y - r,
            ChineseTableSpecs.TABLE_LENGTH - r - y,
        )
        return (
            bool(long_rail_distance < cls.NEAR_CUSHION_DISTANCE),
            bool(short_rail_distance < cls.NEAR_CUSHION_DISTANCE),
        )

    @classmethod
    def _problem_priority(cls, best_shot_ease, available_pockets, is_dead, cushion_pocket_severity):
        if is_dead:
            return 1.0

        difficulty = 1.0 - float(np.clip(best_shot_ease, 0.0, 1.0))
        priority = 0.1 + 0.45 * difficulty + 0.35 * cushion_pocket_severity
        return float(np.clip(priority, 0.05, 0.85))

    @classmethod
    def _reasons(cls, is_dead, best_shot_ease, near_long_rail, near_short_rail, cushion_pocket_severity):
        reasons = []
        if is_dead:
            reasons.append("dead")
        elif best_shot_ease < cls.HARD_SHOT_EASE:
            reasons.append("low_shot_ease")
        if cushion_pocket_severity >= cls.CUSHION_POCKET_HARD_SEVERITY:
            reasons.append("near_cushion_far_pocket")
        if near_long_rail:
            reasons.append("near_long_rail")
        if near_short_rail:
            reasons.append("near_short_rail")
        return tuple(reasons)
