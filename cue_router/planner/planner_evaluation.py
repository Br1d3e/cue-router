import numpy as np

from cue_router.core.table import ChineseTableSpecs


class PlannerEvaluation:
    BALL_CLEARANCE = 2 * ChineseTableSpecs.BALL_RADIUS
    CUSHION_SAFE_DISTANCE = 0.12

    @classmethod
    def evaluate_position(
            cls,
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            obstacles=None,
    ) -> float:
        return cls.evaluate_shot_ease(
            cue_ball_pos,
            target_ball_pos,
            target_pocket_id,
            obstacles=obstacles,
        )

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
        pocket_xy = cls._pocket_xy(target_pocket_id)

        if not ChineseTableSpecs.in_table(cue_xy) or not ChineseTableSpecs.in_table(target_xy):
            return 0.0
        if not cls._has_line_of_sight(cue_xy, target_xy, obstacles, exclude=(cue_xy, target_xy)):
            return 0.0
        if not cls._has_line_of_sight(target_xy, pocket_xy, obstacles, exclude=(cue_xy, target_xy)):
            return 0.0

        cue_to_object = target_xy - cue_xy
        object_to_pocket = pocket_xy - target_xy
        v_co = np.linalg.norm(cue_to_object)
        v_op = np.linalg.norm(object_to_pocket)
        if v_co == 0 or v_op == 0:
            return 0.0

        cos_alpha = np.dot(cue_to_object, object_to_pocket) / (v_co * v_op)
        cos_alpha = max(float(cos_alpha), 0.0)
        kappa = cos_alpha / (v_op * v_co)

        if cls._is_side_pocket(target_pocket_id):
            cos_gamma = abs(float(object_to_pocket[0] / v_op))
            kappa *= cos_gamma

        difficulty = 1 / (1 + kappa)
        pot_quality = 1 - difficulty
        return float(np.clip(pot_quality * cls._cue_cushion_quality(cue_xy), 0.0, 1.0))

    @classmethod
    def _has_line_of_sight(cls, start_xy, end_xy, obstacles, exclude=()) -> bool:
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

    @staticmethod
    def _is_side_pocket(pocket_id) -> bool:
        return isinstance(pocket_id, str) and pocket_id in ("lc", "rc")

    @staticmethod
    def _pocket_xy(pocket) -> np.ndarray:
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

