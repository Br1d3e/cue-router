from enum import Enum, auto
import pooltool as pt
import numpy as np

try:
    from .table import ChineseTableSpecs
except ImportError:
    from table import ChineseTableSpecs


class InvalidReason(Enum):
    FIRST_HIT_NOT_TARGET_BALL = auto()
    CUE_SCRATCH = auto()
    NO_CUSHION_OR_POCKET = auto()
    TARGET_NOT_IN_CORRECT_POCKET = auto()


class ShotEvaluation:
    """
    Scoring (simplified):
    score normalized to (0, 1)

    **Invalid/Foul**: set `score = 0`\n
    - first hit is not target ball
    - cb in pocket
    - no balls hit cushion or in pocket
    - target ball not reachable
    - target ball does not enter pocket:
    - target ball does not enter correct pocket

    **Score calculation**:\n
    1. if `Rect.contains(xy)` is `True`: `score = 1`
    2. else: `score = 1 / (1 + k * Rect.distance_to(xy))`
    """
    valid = True
    position_quality = 0
    target_route_quality = 0
    cue_action_ease = 0
    score = 0

    def __init__(self, shot: pt.System, target_id, pocket_id, target_rect, continuize=False):
        self.valid = True
        self.invalid_reason = None
        self.position_quality = 0
        self.target_route_quality = 0
        self.cue_action_ease = 0
        self.target_near_miss_quality = 1
        self.cue_route_quality = 0
        self.score = self.evaluate_shot(shot, target_id, pocket_id, target_rect, continuize)

    def is_shot_valid(self, shot: pt.System, target_id, pocket_id, continuize) -> bool:
        if pt.ruleset.get_id_of_first_ball_hit(shot) != target_id:
            self.invalid_reason = InvalidReason.FIRST_HIT_NOT_TARGET_BALL
            return False
        if pt.ruleset.is_ball_pocketed(shot, "cue"):
            self.invalid_reason = InvalidReason.CUE_SCRATCH
            return False
        if len(pt.ruleset.balls_that_hit_cushion(shot)) == 0 and len(pt.ruleset.get_pocketed_ball_ids(shot)) == 0:
            self.invalid_reason = InvalidReason.NO_CUSHION_OR_POCKET
            return False
        if not continuize and not pt.ruleset.is_ball_pocketed_in_pocket(shot, target_id, pocket_id):
            self.invalid_reason = InvalidReason.TARGET_NOT_IN_CORRECT_POCKET
            return False
        return True

    @staticmethod
    def events_before_target_pocket(shot: pt.System, target_id, pocket_id):
        result = []

        for e in shot.events:
            if (
                    e.event_type == pt.EventType.BALL_POCKET
                    and target_id in e.ids
                    and pocket_id in e.ids
            ):
                break
            result.append(e)

        return result

    @staticmethod
    def target_cushions_count(shot: pt.System, target_id, pocket_id):
        count = 0

        for e in ShotEvaluation.events_before_target_pocket(shot, target_id, pocket_id):
            if target_id not in e.ids:
                continue
            if (
                    e.event_type == pt.EventType.BALL_LINEAR_CUSHION or
                    e.event_type == pt.EventType.BALL_CIRCULAR_CUSHION
            ):
                count += 1

        return count

    @staticmethod
    def target_ball_collision_count(shot: pt.System, target_id, pocket_id):
        count = 0
        cue_id = shot.cue.cue_ball_id
        ignored_initial_cue_contact = False

        for e in ShotEvaluation.events_before_target_pocket(shot, target_id, pocket_id):
            if e.event_type != pt.EventType.BALL_BALL:
                continue

            if target_id not in e.ids:
                continue

            other_ids = [x for x in e.ids if x != target_id]
            if not other_ids:
                continue

            other_id = other_ids[0]

            if other_id == cue_id and not ignored_initial_cue_contact:
                ignored_initial_cue_contact = True
                continue

            count += 1

        return count

    @staticmethod
    def target_ball_closest_pocket_distance(shot: pt.System, target_id, pocket_id, continuize=False):
        ball = shot.balls[target_id]

        if continuize and len(ball.history_cts) == 0:
            pt.continuize(shot, inplace=True)

        pocket_xy = np.asarray(
            ChineseTableSpecs.pocket_pos_by_id(pocket_id),
            dtype=float
        )

        min_d = float("inf")

        for state in ball.history_cts:
            ball_xy = np.asarray(state.rvw[0, :2], dtype=float)
            d = np.linalg.norm(ball_xy - pocket_xy)
            min_d = min(min_d, d)

        return min_d

    @staticmethod
    def cue_pocket_corner_collision_count(shot: pt.System) -> int:
        cue_ball_id = shot.cue.cue_ball_id
        count = 0
        for e in shot.events:
            if cue_ball_id not in e.ids:
                continue

            if e.event_type == pt.EventType.BALL_CIRCULAR_CUSHION:
                count += 1

        return count

    @staticmethod
    def cue_ball_collision_count(shot: pt.System, target_id) -> int:
        cue_ball_id = shot.cue.cue_ball_id
        count = 0
        ignored_initial_cue_contact = False
        for e in shot.events:
            if cue_ball_id not in e.ids:
                continue

            if e.event_type != pt.EventType.BALL_BALL:
                continue

            other_ids = [x for x in e.ids if x != cue_ball_id]
            if not other_ids:
                continue

            other_id = other_ids[0]

            if other_id == target_id and not ignored_initial_cue_contact:
                ignored_initial_cue_contact = True
                continue

            count += 1

        return count

    def get_position_quality(self, shot: pt.System, target_rect, k=10) -> float:
        cb_xy = shot.balls.get(shot.cue.cue_ball_id).xyz[:2]
        if target_rect.contains(cb_xy):
            return 1.0

        return 1 / (1 + k * target_rect.distance_to(cb_xy))

    def get_target_route_quality(self, shot: pt.System, target_id, pocket_id, k_b=0.8, k_c=1.0) -> float:
        b = self.target_ball_collision_count(shot, target_id, pocket_id)
        c = self.target_cushions_count(shot, target_id, pocket_id)

        return 0.1 + 0.9 * (np.exp(-(k_b * b + k_c * c)))

    def get_cue_action_ease(self, shot: pt.System, k=0.5) -> float:
        V0 = shot.cue.V0
        a = shot.cue.a
        b = shot.cue.b
        return 0.8 + 0.2 / (1 + k * (0.5 * (V0 - 1.5) + abs(a) + abs(b)))

    def get_target_near_miss_quality(self, shot: pt.System, target_id, pocket_id, k=30, limit=0.15) -> float:
        if pt.ruleset.is_ball_pocketed_in_pocket(shot, target_id, pocket_id):
            return 1.0

        d = self.target_ball_closest_pocket_distance(shot, target_id, pocket_id, continuize=True)
        return limit * 1 / (1 + k * d)

    def get_cue_route_quality(self, shot: pt.System, target_id, k_1=0.3, k_2=0.1) -> float:
        pc = self.cue_pocket_corner_collision_count(shot)
        cbb = self.cue_ball_collision_count(shot, target_id)
        return 0.5 + 0.5 / (1 + (k_1 * pc + k_2 * (cbb ** 2)))

    def evaluate_shot(self, shot: pt.System, target_id, pocket_id, target_rect, continuize) -> float:
        if not self.is_shot_valid(shot, target_id, pocket_id, continuize):
            self.valid = False
            return 0.0
        self.valid = True

        self.position_quality = self.get_position_quality(shot, target_rect, k=10)
        self.target_route_quality = self.get_target_route_quality(shot, target_id, pocket_id)
        self.cue_action_ease = self.get_cue_action_ease(shot)
        self.cue_route_quality = self.get_cue_route_quality(shot, target_id)
        if continuize:
            self.target_near_miss_quality = self.get_target_near_miss_quality(shot, target_id, pocket_id, k=30,
                                                                              limit=0.15)
        self.score = (self.position_quality * self.target_route_quality
                      * self.cue_action_ease * self.cue_route_quality * self.target_near_miss_quality)
        return self.score

    @staticmethod
    def debug_eval(shot, target_id, pocket_id):
        print("events:")
        for e in shot.events:
            print(round(e.time, 4), e.event_type, e.ids)

        print("first hit:", pt.ruleset.get_id_of_first_ball_hit(shot))
        print("cue potted:", pt.ruleset.is_ball_pocketed(shot, "cue"))
        print("cushion balls:", pt.ruleset.balls_that_hit_cushion(shot))
        print("pocketed:", pt.ruleset.get_pocketed_ball_ids(shot))
        print(
            "target in pocket:",
            pt.ruleset.is_ball_pocketed_in_pocket(shot, target_id, pocket_id),
        )
