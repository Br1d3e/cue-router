from dataclasses import dataclass

import numpy as np

from cue_router.core.geometry import HeatmapTarget
from cue_router.core.request import BreakoutTarget, ShotObjective
from cue_router.core.table import ChineseTableSpecs
from cue_router.planner.planner_evaluation import PlannerEvaluation, ShotOption


@dataclass(frozen=True)
class BreakoutCandidate:
    helper_ball_id: str
    helper_pocket_id: str
    dead_ball_id: str
    support_score: float
    problem_priority: float

    def label(self):
        return f"{self.helper_ball_id}:{self.helper_pocket_id}->breakout:{self.dead_ball_id}"


class BreakoutPlanner:
    MIN_CUT_ANGLE_DEG = 5.0
    MAX_CUT_ANGLE_DEG = 45.0
    CUT_ANGLE_SOFTNESS_DEG = 30.0
    TANGENT_SOFTNESS_DEG = 45.0
    MAX_BREAKOUT_DISTANCE = 1.0

    @classmethod
    def development_targets(cls, problem_profiles, threshold, top_k):
        targets = [
            profile
            for profile in problem_profiles.values()
            if profile.problem_priority >= threshold
        ]
        targets.sort(key=lambda profile: profile.problem_priority, reverse=True)
        return targets[:top_k]

    @classmethod
    def helper_ball_ids(cls, legal_targets, problem_profiles, balls, exclude=()):
        candidates = [
            ball_id
            for ball_id in legal_targets
            if ball_id in balls and ball_id not in exclude
        ]

        easy = [
            ball_id
            for ball_id in candidates
            if not cls._is_dead_or_breakout_required(problem_profiles.get(ball_id))
            and not cls._is_hard(problem_profiles.get(ball_id))
        ]
        if len(easy) > 0:
            return easy

        return [
            ball_id
            for ball_id in candidates
            if not cls._is_dead_or_breakout_required(problem_profiles.get(ball_id))
        ]

    @classmethod
    def top_helper_candidates(
            cls,
            balls,
            legal_targets,
            problem_profiles,
            dead_profile,
            top_k,
            exclude=(),
    ):
        candidates = []
        for helper_ball_id in cls.helper_ball_ids(legal_targets, problem_profiles, balls, exclude=exclude):
            for pocket_id in ChineseTableSpecs.POCKET_ID:
                score = cls.breakout_helper_quality(
                    balls[helper_ball_id],
                    pocket_id,
                    balls[dead_profile.ball_id],
                    balls,
                )
                if score <= 0:
                    continue
                candidates.append(BreakoutCandidate(
                    helper_ball_id=helper_ball_id,
                    helper_pocket_id=pocket_id,
                    dead_ball_id=dead_profile.ball_id,
                    support_score=score,
                    problem_priority=dead_profile.problem_priority,
                ))

        candidates.sort(key=lambda candidate: candidate.support_score, reverse=True)
        return candidates[:top_k]

    @classmethod
    def build_breakout_shot_options(
            cls,
            cue_xy,
            balls,
            legal_targets,
            problem_profiles,
            config,
    ):
        options = []
        for dead_profile in cls.development_targets(
                problem_profiles,
                config.breakout_problem_priority_threshold,
                config.breakout_problem_top_k,
        ):
            if dead_profile.ball_id not in balls:
                continue

            for candidate in cls.top_helper_candidates(
                    balls,
                    legal_targets,
                    problem_profiles,
                    dead_profile,
                    config.breakout_helper_top_k,
                    exclude=(dead_profile.ball_id,),
            ):
                helper_xy = balls[candidate.helper_ball_id]
                pot_quality = PlannerEvaluation.evaluate_shot_ease(
                    cue_xy,
                    helper_xy,
                    candidate.helper_pocket_id,
                    obstacles=balls,
                )
                if pot_quality <= config.breakout_shot_min_pot_quality:
                    continue

                geometry_quality = cls.breakout_geometry_quality(
                    cue_xy,
                    helper_xy,
                    candidate.helper_pocket_id,
                    balls[candidate.dead_ball_id],
                    balls,
                    check_cue_line=True,
                    check_helper_pocket_line=True,
                )
                if geometry_quality < config.breakout_shot_min_geometry_quality:
                    continue

                problem_bonus = config.problem_priority_weight * candidate.problem_priority
                score = (
                    pot_quality
                    * geometry_quality
                    * candidate.support_score
                    * (1.0 + problem_bonus)
                    * config.breakout_shot_weight
                )
                options.append(ShotOption(
                    target_ball_id=candidate.helper_ball_id,
                    target_pocket_id=candidate.helper_pocket_id,
                    target_region=None,
                    score=np.clip(float(score), 0, 1),
                    route_type="direct",
                    objective=ShotObjective.BREAKOUT,
                    breakout_target=BreakoutTarget(candidate.dead_ball_id),
                    option_family="breakout_shot",
                    score_breakdown={
                        "option_family": "breakout_shot",
                        "helper_ball_id": candidate.helper_ball_id,
                        "dead_ball_id": candidate.dead_ball_id,
                        "helper_pocket_id": candidate.helper_pocket_id,
                        "helper_pot_quality": pot_quality,
                        "breakout_geometry_quality": geometry_quality,
                        "helper_support_score": candidate.support_score,
                        "problem_priority": candidate.problem_priority,
                        "problem_bonus": problem_bonus,
                    },
                ))

        return options

    @classmethod
    def build_breakout_position_options(
            cls,
            cue_xy,
            balls,
            legal_targets,
            problem_profiles,
            target_ball_id,
            target_pocket_id,
            current_pot_quality,
            config,
    ):
        options = []
        future_balls = {ball_id: xy for ball_id, xy in balls.items() if ball_id != target_ball_id}
        future_legal_targets = [ball_id for ball_id in legal_targets if ball_id != target_ball_id]

        for dead_profile in cls.development_targets(
                problem_profiles,
                config.breakout_problem_priority_threshold,
                config.breakout_problem_top_k,
        ):
            if dead_profile.ball_id not in future_balls:
                continue

            for candidate in cls.top_helper_candidates(
                    future_balls,
                    future_legal_targets,
                    problem_profiles,
                    dead_profile,
                    config.breakout_helper_top_k,
                    exclude=(dead_profile.ball_id,),
            ):
                heatmap = cls.breakout_position_heatmap(
                    future_balls,
                    candidate,
                    config.grid_m,
                    config.grid_n,
                    config.pocket_mask_radius,
                    threshold=config.breakout_position_threshold,
                )
                peak_value = float(np.max(heatmap.scores))
                if peak_value <= 0:
                    continue

                problem_bonus = config.problem_priority_weight * candidate.problem_priority
                score = (
                    current_pot_quality
                    * peak_value
                    * candidate.support_score
                    * (1.0 + problem_bonus)
                    * config.breakout_position_weight
                )
                options.append(ShotOption(
                    target_ball_id=target_ball_id,
                    target_pocket_id=target_pocket_id,
                    target_region=heatmap,
                    score=float(score),
                    route_type="direct",
                    objective=ShotObjective.POSITION,
                    option_family="breakout_position",
                    score_breakdown={
                        "option_family": "breakout_position",
                        "current_pot_quality": current_pot_quality,
                        "target_region_value": peak_value,
                        "future_position_value": peak_value,
                        "intended_next_shot": candidate.label(),
                        "helper_ball_id": candidate.helper_ball_id,
                        "dead_ball_id": candidate.dead_ball_id,
                        "helper_pocket_id": candidate.helper_pocket_id,
                        "helper_support_score": candidate.support_score,
                        "problem_priority": candidate.problem_priority,
                        "problem_bonus": problem_bonus,
                    },
                ))

        return options

    @classmethod
    def breakout_position_heatmap(
            cls,
            balls,
            candidate,
            m,
            n,
            pocket_mask_radius,
            threshold=0.0,
    ):
        heatmap = HeatmapTarget(np.zeros((m, n), dtype=float), pocket_mask_radius=pocket_mask_radius)
        cue_X, cue_Y = np.meshgrid(heatmap.x_coords, heatmap.y_coords)
        scores = cls.breakout_geometry_quality_grid(
            cue_X,
            cue_Y,
            balls[candidate.helper_ball_id],
            candidate.helper_pocket_id,
            balls[candidate.dead_ball_id],
            balls,
        )
        scores = np.where(heatmap.valid_position_mask(cue_X, cue_Y), scores, 0.0)
        if threshold > 0:
            scores = np.where(scores >= threshold, scores, 0.0)
        heatmap.scores = np.clip(scores, 0.0, 1.0)

        return heatmap

    @classmethod
    def breakout_helper_quality(cls, helper_xy, helper_pocket_id, dead_xy, balls):
        helper_xy = PlannerEvaluation._xy(helper_xy)
        dead_xy = PlannerEvaluation._xy(dead_xy)
        pocket_xy = PlannerEvaluation._aim_pocket_xy(helper_pocket_id)

        if not PlannerEvaluation.has_line_of_sight(
                helper_xy,
                pocket_xy,
                balls,
                exclude=(helper_xy,),
        ):
            return 0.0

        pot_dir = cls._unit(pocket_xy - helper_xy)
        desired_dir = cls._unit(dead_xy - helper_xy)
        if pot_dir is None or desired_dir is None:
            return 0.0

        tangent_axis = np.asarray((-pot_dir[1], pot_dir[0]), dtype=float)
        tangent_alignment = abs(float(np.dot(tangent_axis, desired_dir)))
        angle_error = np.degrees(np.arccos(np.clip(tangent_alignment, -1.0, 1.0)))
        angle_quality = np.exp(-((angle_error / cls.TANGENT_SOFTNESS_DEG) ** 2))
        distance_quality = cls._breakout_distance_quality(np.linalg.norm(dead_xy - helper_xy))
        return float(np.clip(angle_quality * distance_quality, 0.0, 1.0))

    @classmethod
    def breakout_geometry_quality(
            cls,
            cue_xy,
            helper_xy,
            helper_pocket_id,
            dead_xy,
            balls,
            check_cue_line=False,
            check_helper_pocket_line=False,
    ):
        cue_xy = PlannerEvaluation._xy(cue_xy)
        helper_xy = PlannerEvaluation._xy(helper_xy)
        dead_xy = PlannerEvaluation._xy(dead_xy)
        pocket_xy = PlannerEvaluation._aim_pocket_xy(helper_pocket_id)

        if check_cue_line and not PlannerEvaluation.has_line_of_sight(
                cue_xy,
                helper_xy,
                balls,
                exclude=(cue_xy, helper_xy),
        ):
            return 0.0
        if check_helper_pocket_line and not PlannerEvaluation.has_line_of_sight(
                helper_xy,
                pocket_xy,
                balls,
                exclude=(helper_xy,),
        ):
            return 0.0

        pot_dir = cls._unit(pocket_xy - helper_xy)
        cue_in_dir = cls._unit(helper_xy - cue_xy)
        desired_dir = cls._unit(dead_xy - helper_xy)
        if pot_dir is None or cue_in_dir is None or desired_dir is None:
            return 0.0

        tangent_axis = np.asarray((-pot_dir[1], pot_dir[0]), dtype=float)
        actual_tangent = tangent_axis if np.dot(cue_in_dir, tangent_axis) >= 0 else -tangent_axis
        tangent_alignment = max(0.0, float(np.dot(actual_tangent, desired_dir)))
        tangent_error = np.degrees(np.arccos(np.clip(tangent_alignment, -1.0, 1.0)))
        tangent_quality = np.exp(-((tangent_error / cls.TANGENT_SOFTNESS_DEG) ** 2))

        cut_angle_deg = PlannerEvaluation.cut_angle_deg(cue_xy, helper_xy, helper_pocket_id)
        cut_quality = cls._breakout_cut_angle_quality(cut_angle_deg)
        distance_quality = cls._breakout_distance_quality(np.linalg.norm(dead_xy - helper_xy))
        clearance_quality = cls._helper_dead_path_clearance(helper_xy, dead_xy, balls)

        return float(np.clip(tangent_quality * cut_quality * distance_quality * clearance_quality, 0.0, 1.0))

    @classmethod
    def breakout_geometry_quality_grid(
            cls,
            cue_X,
            cue_Y,
            helper_xy,
            helper_pocket_id,
            dead_xy,
            balls,
    ):
        helper_xy = PlannerEvaluation._xy(helper_xy)
        dead_xy = PlannerEvaluation._xy(dead_xy)
        pocket_xy = PlannerEvaluation._aim_pocket_xy(helper_pocket_id)

        pot_dir = cls._unit(pocket_xy - helper_xy)
        desired_dir = cls._unit(dead_xy - helper_xy)
        if pot_dir is None or desired_dir is None:
            return np.zeros_like(cue_X, dtype=float)

        cue_to_helper_x = helper_xy[0] - cue_X
        cue_to_helper_y = helper_xy[1] - cue_Y
        v_ch = np.sqrt(cue_to_helper_x ** 2 + cue_to_helper_y ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            cue_in_x = cue_to_helper_x / v_ch
            cue_in_y = cue_to_helper_y / v_ch

        tangent_axis = np.asarray((-pot_dir[1], pot_dir[0]), dtype=float)
        tangent_component = cue_in_x * tangent_axis[0] + cue_in_y * tangent_axis[1]
        actual_tangent_x = np.where(tangent_component >= 0, tangent_axis[0], -tangent_axis[0])
        actual_tangent_y = np.where(tangent_component >= 0, tangent_axis[1], -tangent_axis[1])
        tangent_alignment = np.maximum(0.0, actual_tangent_x * desired_dir[0] + actual_tangent_y * desired_dir[1])
        tangent_error = np.degrees(np.arccos(np.clip(tangent_alignment, -1.0, 1.0)))
        tangent_quality = np.exp(-((tangent_error / cls.TANGENT_SOFTNESS_DEG) ** 2))

        cut_angle_deg = PlannerEvaluation.cut_angle_deg_grid(cue_X, cue_Y, helper_xy, helper_pocket_id)
        cut_quality = cls._breakout_cut_angle_quality_grid(cut_angle_deg)
        distance_quality = cls._breakout_distance_quality(np.linalg.norm(dead_xy - helper_xy))
        clearance_quality = cls._helper_dead_path_clearance(helper_xy, dead_xy, balls)

        cue_visible = PlannerEvaluation.has_line_of_sight_grid(
            cue_X,
            cue_Y,
            helper_xy,
            balls,
            fixed_exclude=(helper_xy,),
            grid_exclude=((cue_X, cue_Y),),
        )
        valid = (v_ch > 0) & cue_visible
        return np.where(
            valid,
            tangent_quality * cut_quality * distance_quality * clearance_quality,
            0.0,
        )

    @classmethod
    def _breakout_cut_angle_quality(cls, cut_angle_deg):
        if cls.MIN_CUT_ANGLE_DEG <= cut_angle_deg <= cls.MAX_CUT_ANGLE_DEG:
            return 1.0
        if cut_angle_deg < cls.MIN_CUT_ANGLE_DEG:
            error = cls.MIN_CUT_ANGLE_DEG - cut_angle_deg
        else:
            error = cut_angle_deg - cls.MAX_CUT_ANGLE_DEG
        return float(np.exp(-((error / cls.CUT_ANGLE_SOFTNESS_DEG) ** 2)))

    @classmethod
    def _breakout_cut_angle_quality_grid(cls, cut_angle_deg):
        low_error = np.maximum(0.0, cls.MIN_CUT_ANGLE_DEG - cut_angle_deg)
        high_error = np.maximum(0.0, cut_angle_deg - cls.MAX_CUT_ANGLE_DEG)
        error = np.maximum(low_error, high_error)
        return np.exp(-((error / cls.CUT_ANGLE_SOFTNESS_DEG) ** 2))

    @classmethod
    def _breakout_distance_quality(cls, distance):
        return float(np.exp(-((distance / cls.MAX_BREAKOUT_DISTANCE) ** 2)))

    @staticmethod
    def _helper_dead_path_clearance(helper_xy, dead_xy, balls):
        if PlannerEvaluation.has_line_of_sight(
                helper_xy,
                dead_xy,
                balls,
                exclude=(helper_xy, dead_xy),
        ):
            return 1.0
        return 0.65

    @staticmethod
    def _unit(vec):
        length = np.linalg.norm(vec)
        if length == 0:
            return None
        return np.asarray(vec, dtype=float) / length

    @staticmethod
    def _is_dead_or_breakout_required(profile):
        return bool(profile and (profile.is_dead or profile.require_breakout))

    @staticmethod
    def _is_hard(profile):
        return bool(profile and profile.is_hard)
