from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cue_router.core.geometry import Geometry, HeatmapTarget, PositionTarget
from cue_router.core.request import BreakoutTarget, SearchRequest, ShotObjective
from cue_router.core.table import ChineseTableSpecs



@dataclass
class ShotOption:
    target_ball_id: str
    target_pocket_id: str
    target_region: PositionTarget | None
    score: float
    route_type: str = "direct"
    objective: ShotObjective | str = ShotObjective.POSITION
    breakout_target: BreakoutTarget | None = None
    bank_rail: Literal["left", "right", "bottom", "top"] | None = None
    option_family: str = "normal_position"
    score_breakdown: dict[str, object] = field(default_factory=dict)

    @property
    def position_target(self):
        return self.target_region

    def to_search_request(self, shot):
        return SearchRequest(
            shot=shot,
            target_ball_id=self.target_ball_id,
            target_pocket_id=self.target_pocket_id,
            position_target=self.target_region,
            route_type=self.route_type,
            bank_rail=self.bank_rail,
            objective=self.objective,
            breakout_target=self.breakout_target,
        )


class PlannerEvaluation:
    BANK_RAILS = ("left", "right", "bottom", "top")
    BANK_SHORT_RAIL_FACTOR = 1.0
    BANK_LONG_RAIL_FACTOR = 0.75
    BANK_TRAVEL_SOFTNESS = 1.6
    BANK_DEFAULT_CAP = 0.45
    BALL_CLEARANCE = 2 * ChineseTableSpecs.BALL_RADIUS
    OPTIMAL_CO_RANGE = (0.15, 0.6)
    OPTIMAL_OP_RANGE = (0.0, 1.0)
    CUSHION_SAFE_DISTANCE = 0.1
    ALPHA_FREE_DEG = 30.0
    ALPHA_HARD_CUTOFF_DEG = 85.0
    MIN_PLAYABLE_QUALITY = 0.08
    POSITION_ALPHA_FREE_DEG = 15.0
    POSITION_ALPHA_SOFTNESS_DEG = 25.0
    POSITION_OPTIMAL_CO_RANGE = (0.15, 0.6)
    POSITION_CO_CLOSE_SOFTNESS = 0.18
    POSITION_CO_FAR_SOFTNESS = 0.55
    OP_FAR_SOFTNESS = 0.5
    NEAR_CUSHION_DISTANCE = 0.1
    TARGET_CUSHION_POCKET_DISTANCE_SCALE = ChineseTableSpecs.TABLE_WIDTH
    TARGET_CUSHION_SOFTNESS = 0.14
    MID_POCKET_Y_OFFSET = 0.03
    MID_POCKET_MIN_CUSHION_DISTANCE = 0.1
    CUE_BACK_CUSHION_MIN_QUALITY = 0.65
    BACK_SNOOKER_DISTANCE = 0.15
    BACK_SNOOKER_CLEARANCE = 2.5 * ChineseTableSpecs.BALL_RADIUS
    BACK_SNOOKER_MAX_PENALTY = 0.45

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
        pocket_xy = cls._aim_pocket_xy(target_pocket_id)
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
            pot_quality *= 0.5 + 0.5 * cos_gamma

        pot_quality *= cls._target_cushion_quality(target_xy, obstacles)
        pot_quality *= cls._cue_back_cushion_quality(cue_xy, cue_to_object)
        pot_quality *= cls._back_snooker_quality(cue_xy, cue_to_object, obstacles, exclude=(cue_xy, target_xy))
        if pot_quality <= 0:
            return 0.0
        pot_quality = cls.MIN_PLAYABLE_QUALITY + (1 - cls.MIN_PLAYABLE_QUALITY) * pot_quality
        return float(np.clip(pot_quality, 0.0, 1.0))

    @classmethod
    def evaluate_bank_shot_ease(
            cls,
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            rail,
            obstacles=None,
            cap=BANK_DEFAULT_CAP,
    ) -> float:
        bank = cls.bank_geometry(cue_ball_pos, target_ball_pos, target_pocket_id, rail, obstacles)
        if bank is None:
            return 0.0

        cue_xy = cls._xy(cue_ball_pos)
        target_xy = cls._xy(target_ball_pos)
        contact_xy = bank["contact_xy"]
        object_to_contact = contact_xy - target_xy
        cue_to_object = target_xy - cue_xy
        v_co = np.linalg.norm(cue_to_object)
        v_or = np.linalg.norm(object_to_contact)
        if v_co == 0 or v_or == 0:
            return 0.0

        cos_alpha = np.dot(cue_to_object, object_to_contact) / (v_co * v_or)
        cos_alpha = max(float(cos_alpha), 0.0)
        alpha_deg = np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0)))
        if alpha_deg >= cls.ALPHA_HARD_CUTOFF_DEG:
            return 0.0

        raw_quality = (
            cls._cut_angle_quality(alpha_deg)
            * cls._cue_object_distance_quality(v_co)
            * cls._bank_travel_distance_quality(bank["travel_distance"])
            * cls._bank_rail_quality(rail)
        )
        return float(np.clip(cap * raw_quality, 0.0, cap))

    @classmethod
    def is_bank_candidate(
            cls,
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            rail,
            obstacles=None,
            cap=BANK_DEFAULT_CAP,
    ):
        return cls.evaluate_bank_shot_ease(
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            rail,
            obstacles=obstacles,
            cap=cap,
        ) > 0

    @classmethod
    def bank_geometry(cls, cue_ball_pos, target_ball_pos, target_pocket_id, rail, obstacles=None):
        if rail not in cls.BANK_RAILS:
            raise ValueError(f"unknown bank rail: {rail}")

        cue_xy = cls._xy(cue_ball_pos)
        target_xy = cls._xy(target_ball_pos)
        pocket_xy = cls._aim_pocket_xy(target_pocket_id)

        if not ChineseTableSpecs.in_table(cue_xy) or not ChineseTableSpecs.in_table(target_xy):
            return None

        mirrored_pocket = Geometry._mirror_point_across_rail(
            pocket_xy,
            rail,
            W=ChineseTableSpecs.TABLE_WIDTH,
            L=ChineseTableSpecs.TABLE_LENGTH,
            R=ChineseTableSpecs.BALL_RADIUS,
        )
        contact_xy = cls._bank_contact_point(target_xy, mirrored_pocket, rail)
        if contact_xy is None:
            return None

        if not cls.has_line_of_sight(cue_xy, target_xy, obstacles, exclude=(cue_xy, target_xy)):
            return None
        if not cls.has_line_of_sight(target_xy, contact_xy, obstacles, exclude=(cue_xy, target_xy)):
            return None
        if not cls.has_line_of_sight(contact_xy, pocket_xy, obstacles, exclude=(cue_xy, target_xy)):
            return None

        return {
            "contact_xy": contact_xy,
            "mirrored_pocket_xy": mirrored_pocket,
            "travel_distance": (
                float(np.linalg.norm(contact_xy - target_xy))
                + float(np.linalg.norm(pocket_xy - contact_xy))
            ),
        }

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

        pot_quality *= cls._target_cushion_quality(target_xy, obstacles)
        pot_quality *= cls._cue_back_cushion_quality_grid(cue_X, cue_Y, cue_to_object_x, cue_to_object_y)
        pot_quality *= cls._back_snooker_quality_grid(
            cue_X,
            cue_Y,
            cue_to_object_x,
            cue_to_object_y,
            obstacles,
            fixed_exclude=(target_xy,),
            grid_exclude=((cue_X, cue_Y),),
        )
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
        return float(0.3 + 0.7 / (1.0 + (v_op / 1.1) ** 2))

    @classmethod
    def _object_pocket_distance_quality_grid(cls, v_op):
        _, op_max = cls.OPTIMAL_OP_RANGE
        too_far = np.maximum(0.0, v_op - op_max)
        return np.exp(-(too_far / cls.OP_FAR_SOFTNESS) ** 1.5)

    @classmethod
    def _bank_contact_point(cls, target_xy, mirrored_pocket_xy, rail):
        target_xy = cls._xy(target_xy)
        mirrored_pocket_xy = cls._xy(mirrored_pocket_xy)
        direction = mirrored_pocket_xy - target_xy
        if rail in ("left", "right"):
            rail_x = (
                ChineseTableSpecs.BALL_RADIUS
                if rail == "left"
                else ChineseTableSpecs.TABLE_WIDTH - ChineseTableSpecs.BALL_RADIUS
            )
            if direction[0] == 0:
                return None
            t = (rail_x - target_xy[0]) / direction[0]
            contact = target_xy + t * direction
            if t <= 0:
                return None
            if not (ChineseTableSpecs.BALL_RADIUS < contact[1] < ChineseTableSpecs.TABLE_LENGTH - ChineseTableSpecs.BALL_RADIUS):
                return None
            return contact

        rail_y = (
            ChineseTableSpecs.BALL_RADIUS
            if rail == "bottom"
            else ChineseTableSpecs.TABLE_LENGTH - ChineseTableSpecs.BALL_RADIUS
        )
        if direction[1] == 0:
            return None
        t = (rail_y - target_xy[1]) / direction[1]
        contact = target_xy + t * direction
        if t <= 0:
            return None
        if not (ChineseTableSpecs.BALL_RADIUS < contact[0] < ChineseTableSpecs.TABLE_WIDTH - ChineseTableSpecs.BALL_RADIUS):
            return None
        return contact

    @classmethod
    def _bank_rail_quality(cls, rail):
        return cls.BANK_SHORT_RAIL_FACTOR if rail in ("bottom", "top") else cls.BANK_LONG_RAIL_FACTOR

    @classmethod
    def _bank_travel_distance_quality(cls, travel_distance):
        return float(np.exp(-((travel_distance / cls.BANK_TRAVEL_SOFTNESS) ** 2)))

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

    @classmethod
    def _target_cushion_quality(cls, target_xy, obstacles=None):
        cushion_distance = cls._nearest_cushion_distance(target_xy)
        if cushion_distance >= cls.NEAR_CUSHION_DISTANCE:
            return 1.0

        nearest_pocket_distance = cls.nearest_available_pocket_distance(target_xy, obstacles)
        if not np.isfinite(nearest_pocket_distance):
            return 0.0

        pocket_quality = np.exp(-((nearest_pocket_distance / cls.TARGET_CUSHION_POCKET_DISTANCE_SCALE) ** 2))
        cushion_excess = max(0.0, cls.NEAR_CUSHION_DISTANCE - cushion_distance)
        cushion_quality = np.exp(-((cushion_excess / cls.TARGET_CUSHION_SOFTNESS) ** 2))
        return float(np.clip(pocket_quality * cushion_quality, 0.0, 1.0))

    @classmethod
    def nearest_available_pocket_distance(cls, target_xy, obstacles=None):
        target_xy = cls._xy(target_xy)
        best_distance = float("inf")
        for pocket_id in cls.available_pocket_ids_for_target(target_xy):
            pocket_xy = cls._aim_pocket_xy(pocket_id)
            if not cls.has_line_of_sight(
                    target_xy,
                    pocket_xy,
                    obstacles,
                    exclude=(target_xy,),
            ):
                continue
            best_distance = min(best_distance, float(np.linalg.norm(pocket_xy - target_xy)))
        return best_distance

    @classmethod
    def available_pocket_ids_for_target(cls, target_xy):
        target_xy = cls._xy(target_xy)
        side_pockets_allowed = cls._side_pockets_allowed_for_target(target_xy)
        return [
            pocket_id
            for pocket_id in ChineseTableSpecs.POCKET_ID
            if side_pockets_allowed or not cls._is_side_pocket(pocket_id)
        ]

    @classmethod
    def _side_pockets_allowed_for_target(cls, target_xy):
        target_xy = cls._xy(target_xy)
        mid_y = ChineseTableSpecs.TABLE_LENGTH / 2
        side_width = ChineseTableSpecs.table_specs.side_pocket_width
        in_mid_pocket_range = abs(target_xy[1] - mid_y) <= side_width / 2
        near_mid_pocket_range = abs(target_xy[1] - mid_y) <= side_width / 2 + cls.MID_POCKET_Y_OFFSET
        return (
            (in_mid_pocket_range or near_mid_pocket_range)
            and cls._nearest_cushion_distance(target_xy) > cls.MID_POCKET_MIN_CUSHION_DISTANCE
        )

    @classmethod
    def target_cushion_pocket_severity(cls, target_xy, nearest_pocket_distance):
        cushion_distance = cls._nearest_cushion_distance(target_xy)
        if cushion_distance >= cls.NEAR_CUSHION_DISTANCE:
            return 0.0
        if not np.isfinite(nearest_pocket_distance):
            return 1.0
        pocket_pressure = 1.0 - np.exp(-((nearest_pocket_distance / cls.TARGET_CUSHION_POCKET_DISTANCE_SCALE) ** 2))
        cushion_excess = cls.NEAR_CUSHION_DISTANCE - cushion_distance
        cushion_pressure = 1.0 - np.exp(-((cushion_excess / cls.TARGET_CUSHION_SOFTNESS) ** 2))
        return float(np.clip(0.65 * pocket_pressure + 0.35 * cushion_pressure, 0.0, 1.0))

    @classmethod
    def _cue_back_cushion_quality(cls, cue_xy, cue_to_object):
        v_co = np.linalg.norm(cue_to_object)
        if v_co == 0:
            return 0.0

        back_dir = -cue_to_object / v_co
        back_cushion_distance = cls._ray_to_cushion_distance(cue_xy, back_dir)
        if back_cushion_distance >= cls.NEAR_CUSHION_DISTANCE:
            return 1.0
        t = np.clip(back_cushion_distance / cls.NEAR_CUSHION_DISTANCE, 0.0, 1.0)
        return float(cls.CUE_BACK_CUSHION_MIN_QUALITY + (1.0 - cls.CUE_BACK_CUSHION_MIN_QUALITY) * t)

    @classmethod
    def _cue_back_cushion_quality_grid(cls, cue_X, cue_Y, cue_to_object_x, cue_to_object_y):
        v_co = np.sqrt(cue_to_object_x ** 2 + cue_to_object_y ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            back_x = -cue_to_object_x / v_co
            back_y = -cue_to_object_y / v_co

        back_cushion_distance = cls._ray_to_cushion_distance_grid(cue_X, cue_Y, back_x, back_y)
        t = np.clip(back_cushion_distance / cls.NEAR_CUSHION_DISTANCE, 0.0, 1.0)
        quality = cls.CUE_BACK_CUSHION_MIN_QUALITY + (1.0 - cls.CUE_BACK_CUSHION_MIN_QUALITY) * t
        return np.where(back_cushion_distance >= cls.NEAR_CUSHION_DISTANCE, 1.0, quality)

    @classmethod
    def _back_snooker_quality(cls, cue_xy, cue_to_object, obstacles, exclude=()):
        v_co = np.linalg.norm(cue_to_object)
        if v_co == 0:
            return 0.0

        back_dir = -cue_to_object / v_co
        worst_severity = 0.0
        for obstacle_xy in cls._obstacle_positions(obstacles):
            if any(np.linalg.norm(obstacle_xy - cls._xy(point)) < ChineseTableSpecs.BALL_RADIUS for point in exclude):
                continue
            projection = float(np.dot(obstacle_xy - cue_xy, back_dir))
            if projection <= 0 or projection > cls.BACK_SNOOKER_DISTANCE:
                continue
            closest = cue_xy + projection * back_dir
            perpendicular = np.linalg.norm(obstacle_xy - closest)
            if perpendicular >= cls.BACK_SNOOKER_CLEARANCE:
                continue
            line_severity = 1.0 - perpendicular / cls.BACK_SNOOKER_CLEARANCE
            distance_severity = 1.0 - projection / cls.BACK_SNOOKER_DISTANCE
            worst_severity = max(worst_severity, line_severity * distance_severity)

        return float(1.0 - cls.BACK_SNOOKER_MAX_PENALTY * worst_severity)

    @classmethod
    def _back_snooker_quality_grid(
            cls,
            cue_X,
            cue_Y,
            cue_to_object_x,
            cue_to_object_y,
            obstacles,
            fixed_exclude=(),
            grid_exclude=(),
    ):
        v_co = np.sqrt(cue_to_object_x ** 2 + cue_to_object_y ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            back_x = -cue_to_object_x / v_co
            back_y = -cue_to_object_y / v_co

        worst_severity = np.zeros_like(cue_X, dtype=float)
        for obstacle_xy in cls._obstacle_positions(obstacles):
            exclude_mask = np.zeros_like(cue_X, dtype=bool)
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

            projection = (obstacle_xy[0] - cue_X) * back_x + (obstacle_xy[1] - cue_Y) * back_y
            closest_x = cue_X + projection * back_x
            closest_y = cue_Y + projection * back_y
            perpendicular = np.sqrt((obstacle_xy[0] - closest_x) ** 2 + (obstacle_xy[1] - closest_y) ** 2)
            line_severity = 1.0 - perpendicular / cls.BACK_SNOOKER_CLEARANCE
            distance_severity = 1.0 - projection / cls.BACK_SNOOKER_DISTANCE
            severity = np.clip(line_severity, 0.0, 1.0) * np.clip(distance_severity, 0.0, 1.0)
            blocked = (
                ~exclude_mask
                & (projection > 0)
                & (projection <= cls.BACK_SNOOKER_DISTANCE)
                & (perpendicular < cls.BACK_SNOOKER_CLEARANCE)
            )
            worst_severity = np.maximum(worst_severity, np.where(blocked, severity, 0.0))

        return 1.0 - cls.BACK_SNOOKER_MAX_PENALTY * worst_severity

    @classmethod
    def _nearest_cushion_distance(cls, xy):
        x, y = cls._xy(xy)
        r = ChineseTableSpecs.BALL_RADIUS
        return min(
            x - r,
            ChineseTableSpecs.TABLE_WIDTH - r - x,
            y - r,
            ChineseTableSpecs.TABLE_LENGTH - r - y,
        )

    @classmethod
    def _ray_to_cushion_distance(cls, xy, direction):
        x, y = cls._xy(xy)
        dx, dy = direction
        r = ChineseTableSpecs.BALL_RADIUS
        distances = []
        if dx < 0:
            distances.append((x - r) / -dx)
        elif dx > 0:
            distances.append((ChineseTableSpecs.TABLE_WIDTH - r - x) / dx)
        if dy < 0:
            distances.append((y - r) / -dy)
        elif dy > 0:
            distances.append((ChineseTableSpecs.TABLE_LENGTH - r - y) / dy)
        if len(distances) == 0:
            return float("inf")
        return float(max(0.0, min(distances)))

    @classmethod
    def _ray_to_cushion_distance_grid(cls, cue_X, cue_Y, dir_X, dir_Y):
        r = ChineseTableSpecs.BALL_RADIUS
        inf = np.full_like(cue_X, np.inf, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(dir_X < 0, (cue_X - r) / -dir_X, inf)
            right = np.where(dir_X > 0, (ChineseTableSpecs.TABLE_WIDTH - r - cue_X) / dir_X, inf)
            bottom = np.where(dir_Y < 0, (cue_Y - r) / -dir_Y, inf)
            top = np.where(dir_Y > 0, (ChineseTableSpecs.TABLE_LENGTH - r - cue_Y) / dir_Y, inf)
        distance = np.minimum.reduce([left, right, bottom, top])
        return np.maximum(distance, 0.0)

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


def evaluate_bank_shot_ease(cue_ball_pos, target_ball_pos, target_pocket_id, rail, obstacles=None) -> float:
    return PlannerEvaluation.evaluate_bank_shot_ease(
        cue_ball_pos,
        target_ball_pos,
        target_pocket_id,
        rail,
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
