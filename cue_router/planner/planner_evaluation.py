from dataclasses import dataclass, field

import numpy as np

from cue_router.core.geometry import HeatmapTarget, PositionTarget
from cue_router.core.request import SearchRequest
from cue_router.core.table import ChineseTableSpecs



@dataclass
class ShotOption:
    target_ball_id: str
    target_pocket_id: str
    target_region: PositionTarget
    score: float
    route_type: str = "direct"
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_search_request(self, shot):
        return SearchRequest(
            shot=shot,
            target_ball_id=self.target_ball_id,
            target_pocket_id=self.target_pocket_id,
            target_region=self.target_region,
            route_type=self.route_type,
        )


class PlannerEvaluation:
    BALL_CLEARANCE = 2 * ChineseTableSpecs.BALL_RADIUS
    OPTIMAL_CO_RANGE = (0.15, 0.6)
    CUSHION_SAFE_DISTANCE = 0.1
    ALPHA_FREE_DEG = 30.0
    ALPHA_HARD_CUTOFF_DEG = 85.0
    MIN_PLAYABLE_QUALITY = 0.08
    POSITION_ALPHA_FREE_DEG = 15.0
    POSITION_ALPHA_SOFTNESS_DEG = 25.0
    POSITION_OPTIMAL_CO_RANGE = (0.15, 0.6)
    POSITION_CO_CLOSE_SOFTNESS = 0.18
    POSITION_CO_FAR_SOFTNESS = 0.55

    @classmethod
    def evaluate_position(
            cls,
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            obstacles=None,
    ) -> float:
        shot_ease = cls.evaluate_shot_ease(
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            obstacles=obstacles,
        )
        if shot_ease <= 0.0:
            return 0.0

        cue_xy = cls._xy(cue_ball_pos)
        target_xy = cls._xy(target_ball_pos)
        v_co = np.linalg.norm(target_xy - cue_xy)
        alpha_deg = cls.cut_angle_deg(cue_xy, target_xy, target_pocket_id)

        position_quality = (
            shot_ease
            * cls._position_cut_angle_quality(alpha_deg)
            * cls._position_cue_object_distance_quality(v_co)
        )
        return float(np.clip(position_quality, 0.0, 1.0))

    @classmethod
    def evaluate_position_grid(
            cls,
            cue_X,
            cue_Y,
            target_ball_pos,
            target_pocket_id,
            obstacles=None,
    ):
        shot_ease = cls.evaluate_shot_ease_grid(
            cue_X,
            cue_Y,
            target_ball_pos,
            target_pocket_id,
            obstacles=obstacles,
        )
        target_xy = cls._xy(target_ball_pos)
        v_co = np.sqrt((target_xy[0] - cue_X) ** 2 + (target_xy[1] - cue_Y) ** 2)
        alpha_deg = cls.cut_angle_deg_grid(cue_X, cue_Y, target_xy, target_pocket_id)

        position_quality = (
            shot_ease
            * cls._position_cut_angle_quality_grid(alpha_deg)
            * cls._position_cue_object_distance_quality_grid(v_co)
        )
        return np.clip(position_quality, 0.0, 1.0)

    @classmethod
    def evaluate_shot_ease(
            cls,
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            obstacles=None,
    ) -> float:
        cue_xy = cls._xy(cue_ball_pos)
        target_xy = cls._xy(target_ball_pos)
        pocket_xy = cls._aim_pocket_xy(target_pocket_id)

        if not ChineseTableSpecs.in_table(cue_xy) or not ChineseTableSpecs.in_table(target_xy):
            return 0.0
        if not cls.has_line_of_sight(cue_xy, target_xy, obstacles, exclude=(cue_xy, target_xy)):
            return 0.0
        if not cls.has_line_of_sight(target_xy, pocket_xy, obstacles, exclude=(cue_xy, target_xy)):
            return 0.0

        cue_to_object = target_xy - cue_xy
        object_to_pocket = pocket_xy - target_xy
        v_co = np.linalg.norm(cue_to_object)
        v_op = np.linalg.norm(object_to_pocket)
        if v_co == 0 or v_op == 0:
            return 0.0

        cos_alpha = np.dot(cue_to_object, object_to_pocket) / (v_co * v_op)
        cos_alpha = max(float(cos_alpha), 0.0)

        alpha_deg = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
        if alpha_deg >= cls.ALPHA_HARD_CUTOFF_DEG:
            return 0.0

        alpha_quality = cls._cut_angle_quality(alpha_deg)
        co_quality = cls._cue_object_distance_quality(v_co)
        op_quality = cls._object_pocket_distance_quality(v_op)

        pot_quality = alpha_quality * co_quality * op_quality

        if cls._is_side_pocket(target_pocket_id):
            cos_gamma = abs(float(object_to_pocket[0] / v_op))
            pot_quality *= 0.65 + 0.35 * cos_gamma

        pot_quality *= cls._cue_cushion_quality(cue_xy)
        if pot_quality <= 0:
            return 0.0
        pot_quality = cls.MIN_PLAYABLE_QUALITY + (1 - cls.MIN_PLAYABLE_QUALITY) * pot_quality
        return float(np.clip(pot_quality, 0.0, 1.0))

    @classmethod
    def evaluate_shot_ease_grid(
            cls,
            cue_X,
            cue_Y,
            target_ball_pos,
            target_pocket_id,
            obstacles=None
    ):
        cue_X = np.asarray(cue_X, dtype=float)
        cue_Y = np.asarray(cue_Y, dtype=float)
        scores = np.zeros_like(cue_X, dtype=float)
        target_xy = cls._xy(target_ball_pos)
        pocket_xy = cls._aim_pocket_xy(target_pocket_id)

        if not ChineseTableSpecs.in_table(target_xy):
            return scores

        object_to_pocket = pocket_xy - target_xy
        v_op = np.linalg.norm(object_to_pocket)
        if v_op == 0:
            return scores

        cue_to_object_x = target_xy[0] - cue_X
        cue_to_object_y = target_xy[1] - cue_Y
        v_co = np.sqrt(cue_to_object_x ** 2 + cue_to_object_y ** 2)

        with np.errstate(divide="ignore", invalid="ignore"):
            cos_alpha = (
                cue_to_object_x * object_to_pocket[0]
                + cue_to_object_y * object_to_pocket[1]
            ) / (v_co * v_op)
        cos_alpha = np.maximum(cos_alpha, 0.0)
        alpha_deg = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))

        cue_target_visible = cls.has_line_of_sight_grid(
            cue_X,
            cue_Y,
            target_xy,
            obstacles,
            fixed_exclude=(target_xy,),
            grid_exclude=((cue_X, cue_Y),),
        )
        target_pocket_visible = cls.has_line_of_sight_grid(
            np.full_like(cue_X, target_xy[0], dtype=float),
            np.full_like(cue_Y, target_xy[1], dtype=float),
            pocket_xy,
            obstacles,
            fixed_exclude=(target_xy,),
            grid_exclude=((cue_X, cue_Y),),
        )

        valid = (
            ChineseTableSpecs.in_table_mask(cue_X, cue_Y)
            & cue_target_visible
            & target_pocket_visible
            & (v_co != 0)
            & (alpha_deg < cls.ALPHA_HARD_CUTOFF_DEG)
        )
        if not np.any(valid):
            return scores

        pot_quality = (
            cls._cut_angle_quality_grid(alpha_deg)
            * cls._cue_object_distance_quality_grid(v_co)
            * cls._object_pocket_distance_quality_grid(v_op)
        )

        if cls._is_side_pocket(target_pocket_id):
            cos_gamma = abs(float(object_to_pocket[0] / v_op))
            pot_quality *= 0.65 + 0.35 * cos_gamma

        pot_quality *= cls._cue_cushion_quality_grid(cue_X, cue_Y)
        positive = valid & (pot_quality > 0)
        scores[positive] = (
            cls.MIN_PLAYABLE_QUALITY
            + (1 - cls.MIN_PLAYABLE_QUALITY) * pot_quality[positive]
        )
        return np.clip(scores, 0.0, 1.0)

    @classmethod
    def _cut_angle_quality(cls, alpha_deg):
        if alpha_deg <= cls.ALPHA_FREE_DEG:
            return 1.0
        t = (alpha_deg - cls.ALPHA_FREE_DEG) / (cls.ALPHA_HARD_CUTOFF_DEG - cls.ALPHA_FREE_DEG)
        t = float(np.clip(t, 0.0, 1.0))
        return float(1.0 - 0.9 * (t ** 1.35))

    @classmethod
    def _cut_angle_quality_grid(cls, alpha_deg):
        t = (alpha_deg - cls.ALPHA_FREE_DEG) / (cls.ALPHA_HARD_CUTOFF_DEG - cls.ALPHA_FREE_DEG)
        t = np.clip(t, 0.0, 1.0)
        return np.where(alpha_deg <= cls.ALPHA_FREE_DEG, 1.0, 1.0 - 0.9 * (t ** 1.35))

    @classmethod
    def _cue_object_distance_quality(cls, v_co):
        co_min, co_max = cls.OPTIMAL_CO_RANGE
        if co_min <= v_co <= co_max:
            return 1.0
        if v_co < co_min:
            t = (co_min - v_co) / co_min
            return float(1.0 - 0.75 * np.clip(t, 0.0, 1.0) ** 1.4)
        t = (v_co - co_max) / (ChineseTableSpecs.TABLE_LENGTH - co_max)
        return float(1.0 - 0.55 * np.clip(t, 0.0, 1.0) ** 1.2)

    @classmethod
    def _cue_object_distance_quality_grid(cls, v_co):
        co_min, co_max = cls.OPTIMAL_CO_RANGE
        quality = np.ones_like(v_co, dtype=float)
        close = v_co < co_min
        far = v_co > co_max
        close_t = np.clip((co_min - v_co) / co_min, 0.0, 1.0)
        far_t = np.clip((v_co - co_max) / (ChineseTableSpecs.TABLE_LENGTH - co_max), 0.0, 1.0)
        quality[close] = 1.0 - 0.75 * close_t[close] ** 1.4
        quality[far] = 1.0 - 0.55 * far_t[far] ** 1.2
        return quality

    @classmethod
    def _position_cut_angle_quality(cls, alpha_deg):
        excess = max(0.0, alpha_deg - cls.POSITION_ALPHA_FREE_DEG)
        return float(np.exp(-((excess / cls.POSITION_ALPHA_SOFTNESS_DEG) ** 2)))

    @classmethod
    def _position_cut_angle_quality_grid(cls, alpha_deg):
        excess = np.maximum(0.0, alpha_deg - cls.POSITION_ALPHA_FREE_DEG)
        return np.exp(-((excess / cls.POSITION_ALPHA_SOFTNESS_DEG) ** 2))

    @classmethod
    def _position_cue_object_distance_quality(cls, v_co):
        co_min, co_max = cls.POSITION_OPTIMAL_CO_RANGE
        too_close = max(0.0, co_min - v_co)
        too_far = max(0.0, v_co - co_max)
        return float(np.exp(-(
            (too_close / cls.POSITION_CO_CLOSE_SOFTNESS) ** 2
            + (too_far / cls.POSITION_CO_FAR_SOFTNESS) ** 2
        )))

    @classmethod
    def _position_cue_object_distance_quality_grid(cls, v_co):
        co_min, co_max = cls.POSITION_OPTIMAL_CO_RANGE
        too_close = np.maximum(0.0, co_min - v_co)
        too_far = np.maximum(0.0, v_co - co_max)
        return np.exp(-(
            (too_close / cls.POSITION_CO_CLOSE_SOFTNESS) ** 2
            + (too_far / cls.POSITION_CO_FAR_SOFTNESS) ** 2
        ))

    @staticmethod
    def _object_pocket_distance_quality(v_op):
        return float(0.4 + 0.6 / (1.0 + (v_op / 1.15) ** 1.6))

    @staticmethod
    def _object_pocket_distance_quality_grid(v_op):
        return 0.4 + 0.6 / (1.0 + (v_op / 1.15) ** 1.6)

    @classmethod
    def cut_angle_deg(cls, cue_ball_pos, target_ball_pos, target_pocket_id):
        cue_xy = cls._xy(cue_ball_pos)
        target_xy = cls._xy(target_ball_pos)
        pocket_xy = cls._aim_pocket_xy(target_pocket_id)
        cue_to_object = target_xy - cue_xy
        object_to_pocket = pocket_xy - target_xy
        v_co = np.linalg.norm(cue_to_object)
        v_op = np.linalg.norm(object_to_pocket)
        if v_co == 0 or v_op == 0:
            return 180.0
        cos_alpha = np.dot(cue_to_object, object_to_pocket) / (v_co * v_op)
        cos_alpha = float(np.clip(cos_alpha, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_alpha)))

    @classmethod
    def cut_angle_deg_grid(cls, cue_X, cue_Y, target_ball_pos, target_pocket_id):
        target_xy = cls._xy(target_ball_pos)
        pocket_xy = cls._aim_pocket_xy(target_pocket_id)
        cue_to_object_x = target_xy[0] - cue_X
        cue_to_object_y = target_xy[1] - cue_Y
        object_to_pocket = pocket_xy - target_xy
        v_co = np.sqrt(cue_to_object_x ** 2 + cue_to_object_y ** 2)
        v_op = np.linalg.norm(object_to_pocket)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos_alpha = (
                cue_to_object_x * object_to_pocket[0]
                + cue_to_object_y * object_to_pocket[1]
            ) / (v_co * v_op)
        alpha_deg = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
        return np.where((v_co == 0) | (v_op == 0), 180.0, alpha_deg)

    @classmethod
    def is_direct_candidate(
            cls,
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            obstacles=None,
            max_cut_angle_deg=85.0,
    ):
        if cls.cut_angle_deg(cue_ball_pos, target_ball_pos, target_pocket_id) >= max_cut_angle_deg:
            return False
        return cls.evaluate_shot_ease(
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            obstacles=obstacles,
        ) > 0

    @classmethod
    def has_line_of_sight(cls, start_xy, end_xy, obstacles, exclude=()) -> bool:
        start_xy = cls._xy(start_xy)
        end_xy = cls._xy(end_xy)
        segment = end_xy - start_xy
        length_sq = np.dot(segment, segment)
        if length_sq == 0:
            return False

        for obstacle_xy in cls._obstacle_positions(obstacles):
            if any(np.linalg.norm(obstacle_xy - cls._xy(point)) < ChineseTableSpecs.BALL_RADIUS for point in exclude):
                continue
            t = np.dot(obstacle_xy - start_xy, segment) / length_sq
            if t <= 0 or t >= 1:
                continue
            closest = start_xy + t * segment
            if np.linalg.norm(obstacle_xy - closest) < cls.BALL_CLEARANCE:
                return False
        return True

    @classmethod
    def has_line_of_sight_grid(
            cls,
            start_X,
            start_Y,
            end_xy,
            obstacles,
            fixed_exclude=(),
            grid_exclude=(),
    ):
        start_X = np.asarray(start_X, dtype=float)
        start_Y = np.asarray(start_Y, dtype=float)
        end_xy = cls._xy(end_xy)
        segment_x = end_xy[0] - start_X
        segment_y = end_xy[1] - start_Y
        length_sq = segment_x ** 2 + segment_y ** 2
        visible = length_sq != 0

        for obstacle_xy in cls._obstacle_positions(obstacles):
            exclude_mask = np.zeros_like(start_X, dtype=bool)
            skip_obstacle = False
            for point in fixed_exclude:
                if np.linalg.norm(obstacle_xy - cls._xy(point)) < ChineseTableSpecs.BALL_RADIUS:
                    skip_obstacle = True
                    break
            if skip_obstacle:
                continue

            for grid_X, grid_Y in grid_exclude:
                exclude_mask |= (
                    (obstacle_xy[0] - grid_X) ** 2
                    + (obstacle_xy[1] - grid_Y) ** 2
                    < ChineseTableSpecs.BALL_RADIUS ** 2
                )

            with np.errstate(divide="ignore", invalid="ignore"):
                t = (
                    (obstacle_xy[0] - start_X) * segment_x
                    + (obstacle_xy[1] - start_Y) * segment_y
                ) / length_sq
            on_segment = (0 < t) & (t < 1)
            closest_x = start_X + t * segment_x
            closest_y = start_Y + t * segment_y
            blocked = (
                visible
                & ~exclude_mask
                & on_segment
                & (
                    (obstacle_xy[0] - closest_x) ** 2
                    + (obstacle_xy[1] - closest_y) ** 2
                    < cls.BALL_CLEARANCE ** 2
                )
            )
            visible &= ~blocked
        return visible

    @classmethod
    def _cue_cushion_quality(cls, cue_xy) -> float:
        x, y = cls._xy(cue_xy)
        r = ChineseTableSpecs.BALL_RADIUS
        cushion_distance = min(
            x - r,
            ChineseTableSpecs.TABLE_WIDTH - r - x,
            y - r,
            ChineseTableSpecs.TABLE_LENGTH - r - y,
        )
        return float(np.clip(cushion_distance / cls.CUSHION_SAFE_DISTANCE, 0.0, 1.0))

    @classmethod
    def _cue_cushion_quality_grid(cls, cue_X, cue_Y):
        r = ChineseTableSpecs.BALL_RADIUS
        cushion_distance = np.minimum.reduce([
            cue_X - r,
            ChineseTableSpecs.TABLE_WIDTH - r - cue_X,
            cue_Y - r,
            ChineseTableSpecs.TABLE_LENGTH - r - cue_Y,
        ])
        return np.clip(cushion_distance / cls.CUSHION_SAFE_DISTANCE, 0.0, 1.0)

    @staticmethod
    def _is_side_pocket(pocket_id) -> bool:
        return isinstance(pocket_id, str) and pocket_id in ("lc", "rc")

    @staticmethod
    def _aim_pocket_xy(pocket) -> np.ndarray:
        if isinstance(pocket, str):
            return np.asarray(ChineseTableSpecs.pocket_pos_by_id(pocket), dtype=float)
        return PlannerEvaluation._xy(pocket)

    @staticmethod
    def _obstacle_positions(obstacles):
        if obstacles is None:
            return []
        if isinstance(obstacles, dict):
            obstacles = obstacles.values()
        return [PlannerEvaluation._xy(obstacle) for obstacle in obstacles]

    @staticmethod
    def _xy(value) -> np.ndarray:
        if hasattr(value, "xyz"):
            return np.asarray(value.xyz[:2], dtype=float)
        return np.asarray(value[:2], dtype=float)



def evaluate_shot_ease(cue_ball_pos, target_ball_pos, target_pocket_id, obstacles=None) -> float:
    return PlannerEvaluation.evaluate_shot_ease(
        cue_ball_pos,
        target_ball_pos,
        target_pocket_id,
        obstacles=obstacles,
    )


def evaluate_position(cue_ball_pos, target_ball_pos, target_pocket_id, obstacles=None) -> float:
    return PlannerEvaluation.evaluate_position(
        cue_ball_pos,
        target_ball_pos,
        target_pocket_id,
        obstacles=obstacles,
    )


def evaluate_shot_ease_grid(cue_X, cue_Y, target_ball_pos, target_pocket_id, obstacles=None):
    return PlannerEvaluation.evaluate_shot_ease_grid(
        cue_X,
        cue_Y,
        target_ball_pos,
        target_pocket_id,
        obstacles=obstacles,
    )


def evaluate_position_grid(cue_X, cue_Y, target_ball_pos, target_pocket_id, obstacles=None):
    return PlannerEvaluation.evaluate_position_grid(
        cue_X,
        cue_Y,
        target_ball_pos,
        target_pocket_id,
        obstacles=obstacles,
    )
