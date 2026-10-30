from enum import Enum, auto
from dataclasses import dataclass
import pooltool as pt
import numpy as np

from cue_router.core.table import ChineseTableSpecs
from cue_router.core.request import BreakoutTarget, SearchRequest, ShotObjective
from cue_router.executor.action import ShotAction

import time

class InvalidReason(Enum):
    FIRST_HIT_NOT_TARGET_BALL = auto()
    CUE_SCRATCH = auto()
    NO_CUSHION_OR_POCKET = auto()
    TARGET_NOT_IN_CORRECT_POCKET = auto()


@dataclass
class ShotScoreConfig:
    """
    Config notes:\n
    `position_k`: cue ball-target region relative position penalty. Larger values punishes bad positions more.\n
    `target_collison_k`: target ball collision with balls penalty. (exponential)\n
    `target_cushion_k`: target ball cushion penalty. (exponential)\n
    `cue_action_k`: cue action penalty coefficient. Larger values punishes hard actions more.\n
    `cue_route_corner_k`: cue ball pocket corner collision penalty (cubic increase so k is usually set small)\n
    `cue_route_collision_k`: cue ball collision with other balls penalty\n
    `cue_route_cushion_k`: cue ball cushion penalty. Cubic function, value set very small (0.005)\n
    `near_miss_k`: target ball distance to the closest pocket penalty factor.\n
    `near_miss_limit`: maximum `target_near_miss_quality` when a target ball misses\n
    `robustness_weight`: importance of robustness quality in final evaluation
    `noise_n_samples`: robustness noise sample volume\n
    `noise_scales`: Normal distribution scale of noise (phi, V0, a, b)\n
    `noise_max_attempts`: maximum truncated-Gaussian attempts to collect valid robustness samples\n
    """
    position_k: float = 2.0
    target_collision_k: float = 1.1
    target_cushion_k: float = 1.2
    cue_action_k: float = 1.4
    cue_route_corner_k: float = 0.3
    cue_route_collision_k: float = 0.4
    cue_route_cushion_k: float = 0.005
    near_miss_k: float = 30.0
    near_miss_limit: float = 0.15
    robustness_weight: float = 0.2
    noise_n_samples: int = None
    noise_scales: tuple[float, float, float, float] = None
    noise_max_attempts: int = 128

    def __iter__(self):
        return iter([self.position_k, self.target_collision_k, self.target_cushion_k,
                     self.cue_action_k, self.cue_route_corner_k, self.cue_route_collision_k,
                     self.cue_route_cushion_k, self.near_miss_k, self.near_miss_limit])


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

    def __init__(self, config=ShotScoreConfig()):
        self.config = config
        self.valid = True
        self.invalid_reason = None
        self.position_quality = np.nan
        self.breakout_quality = np.nan
        self.target_route_quality = np.nan
        self.cue_action_ease = np.nan
        self.target_near_miss_quality = np.nan
        self.cue_route_quality = np.nan
        self.route_robustness = np.nan
        self.score = np.nan

        self.diagnostics = {
            "pt_simulate": 0,
            "score_calc": 0,
        }

    def is_shot_valid(self, shot: pt.System, target_id, pocket_id, strict_mode) -> bool:
        if pt.ruleset.get_id_of_first_ball_hit(shot) != target_id:
            self.invalid_reason = InvalidReason.FIRST_HIT_NOT_TARGET_BALL
            return False
        if pt.ruleset.is_ball_pocketed(shot, "cue"):
            self.invalid_reason = InvalidReason.CUE_SCRATCH
            return False
        if len(pt.ruleset.balls_that_hit_cushion(shot)) == 0 and len(pt.ruleset.get_pocketed_ball_ids(shot)) == 0:
            self.invalid_reason = InvalidReason.NO_CUSHION_OR_POCKET
            return False
        if strict_mode and not pt.ruleset.is_ball_pocketed_in_pocket(shot, target_id, pocket_id):
            self.invalid_reason = InvalidReason.TARGET_NOT_IN_CORRECT_POCKET
            return False
        return True

    @staticmethod
    def _route_value(route_type):
        return route_type.value if hasattr(route_type, "value") else route_type

    @staticmethod
    def _objective_value(objective):
        return objective.value if hasattr(objective, "value") else objective

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
    def cue_pocket_cushion_count(shot: pt.System) -> int:
        cue_ball_id = shot.cue.cue_ball_id
        count = 0
        for e in shot.events:
            if cue_ball_id not in e.ids:
                continue

            if e.event_type == pt.EventType.BALL_LINEAR_CUSHION:
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

    # @classmethod
    # def generate_action_noise(cls, n_samples=16, scales=(0.25, 0.5, 0.1, 0.1), seed=42):
    #     if n_samples is None:
    #         n_samples = 16
    #     if scales is None:
    #         scales = (0.25, 0.5, 0.1, 0.1)
    #     rng = np.random.default_rng(seed)
    #     scales = np.asarray(scales, dtype=float)
    #     return rng.normal(0.0, scales, size=(n_samples, 4))

    @classmethod
    def generate_truncated_actions(
            cls,
            action: ShotAction,
            n_samples=16,
            scales=(0.25, 0.5, 0.1, 0.1),
            max_attempts=128,
            seed=42,
    ):
        if n_samples is None:
            n_samples = 16
        if scales is None:
            scales = (0.25, 0.5, 0.1, 0.1)
        if max_attempts is None:
            max_attempts = max(8 * n_samples, n_samples)

        rng = np.random.default_rng(seed)
        scales = np.asarray(scales, dtype=float)
        samples = []
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            phi, V0, a, b = action.as_array() + rng.normal(0.0, scales)
            phi = phi % 360
            if not ShotAction.is_V0_possible(V0) or not ShotAction.is_spin_possible(a, b):
                continue
            samples.append((phi, V0, a, b))

        return np.asarray(samples, dtype=float)

    @classmethod
    def apply_action_noise(cls, action: ShotAction, action_noises) -> list[np.ndarray]:
        if not isinstance(action, ShotAction):
            raise TypeError(f"action must be type of ShotAction(phi, V0, a, b), got {action!r} instead.")
        return action.as_array() + action_noises

    def get_position_quality(self, shot: pt.System, target_region, k: float = 10.0) -> float:
        if k is None:
            k = 10.0
        cb_xy = shot.balls.get(shot.cue.cue_ball_id).xyz[:2]
        return target_region.position_quality(cb_xy, position_k=k)

    def get_target_route_quality(self, shot: pt.System, target_id, pocket_id, k_b=0.8, k_c=1.0, route_type="direct") -> float:
        if k_b is None:
            k_b = 0.8
        if k_c is None:
            k_c = 1.0
        desired_c = {
            "direct": 0,
            "bank": 1,
            "one_rail_bank": 1
        }
        b = self.target_ball_collision_count(shot, target_id, pocket_id)
        c = self.target_cushions_count(shot, target_id, pocket_id)

        route_type = self._route_value(route_type)
        return 0.1 + 0.9 * (np.exp(-(k_b * b + k_c * abs(c - desired_c[route_type]))))

    def get_cue_action_ease(self, shot: pt.System, k=0.5) -> float:
        if k is None:
            k = 0.5
        V0 = shot.cue.V0
        a = shot.cue.a
        b = shot.cue.b
        return 0.85 + 0.15 / (1 + k * (0.4 * (V0 - 1.5) + abs(a) + abs(b) + 4 * V0 * abs(a) + abs(b)))

    def get_target_near_miss_quality(self, shot: pt.System, target_id, pocket_id, k=30, limit=0.15) -> float:
        if pt.ruleset.is_ball_pocketed_in_pocket(shot, target_id, pocket_id):
            return 1.0

        if k is None:
            k = 30
        if limit is None:
            limit = 0.15
        d = self.target_ball_closest_pocket_distance(shot, target_id, pocket_id, continuize=True)
        return limit * 1 / (1 + k * d)

    def get_cue_route_quality(self, shot: pt.System, target_id, k_1=0.6, k_2=0.1, k_3=0.005) -> float:
        if k_1 is None:
            k_1 = 0.6
        if k_2 is None:
            k_2 = 0.1
        if k_3 is None:
            k_3 = 0.005
        pc = self.cue_pocket_corner_collision_count(shot)
        cbb = self.cue_ball_collision_count(shot, target_id)
        cc = self.cue_pocket_cushion_count(shot)
        return 0.5 + 0.5 / (1 + (k_1 * cbb + k_2 * (pc ** 3) + k_3 * (cc ** 3)))

    @staticmethod
    def _ball_path_xy(shot: pt.System, ball_id: str):
        ball = shot.balls[ball_id]
        points = [np.asarray(ball.xyz[:2], dtype=float)]
        for state in ball.history_cts:
            points.append(np.asarray(state.rvw[0, :2], dtype=float))
        return points

    @staticmethod
    def _min_ball_path_distance(shot: pt.System, ball_id: str, xy, continuize=False):
        if continuize and len(shot.balls[ball_id].history_cts) == 0:
            pt.continuize(shot, inplace=True)
        target_xy = np.asarray(xy[:2], dtype=float)
        min_d = float("inf")
        for ball_xy in ShotEvaluation._ball_path_xy(shot, ball_id):
            min_d = min(min_d, float(np.linalg.norm(ball_xy - target_xy)))
        return min_d

    def get_breakout_path_region_quality(self, shot: pt.System, breakout_target: BreakoutTarget, k=None):
        if breakout_target.path_region is None:
            return 0.0
        if k is None:
            k = self.config.position_k
        cue_id = shot.cue.cue_ball_id
        if len(shot.balls[cue_id].history_cts) == 0:
            pt.continuize(shot, inplace=True)
        best = 0.0
        for cue_xy in self._ball_path_xy(shot, cue_id):
            best = max(best, breakout_target.path_region.position_quality(cue_xy, position_k=k))
        return float(np.clip(best, 0.0, 1.0))

    def get_breakout_near_miss_quality(self, shot: pt.System, breakout_target: BreakoutTarget, k=30, limit=0.4):
        if k is None:
            k = 30
        if limit is None:
            limit = 0.4
        cue_id = shot.cue.cue_ball_id
        dead_ball = shot.balls.get(breakout_target.ball_id)
        if dead_ball is None:
            return 0.0
        d = self._min_ball_path_distance(shot, cue_id, dead_ball.xyz[:2], continuize=True)
        contact_gap = max(0.0, d - 2 * ChineseTableSpecs.BALL_RADIUS)
        return float(limit / (1 + k * contact_gap))

    @staticmethod
    def breakout_collision_quality(shot: pt.System, target_id, breakout_target: BreakoutTarget) -> tuple[float, str | None]:
        cue_id = shot.cue.cue_ball_id
        dead_id = breakout_target.ball_id
        if dead_id not in shot.balls:
            return 0.0, None

        first_target_contact_seen = False
        chain_depth = {cue_id: 0, target_id: 0}
        max_depth = breakout_target.max_chain_collisions

        for e in shot.events:
            if e.event_type != pt.EventType.BALL_BALL:
                continue
            ids = list(e.ids)
            if len(ids) < 2:
                continue
            a_id, b_id = ids[0], ids[1]

            if not first_target_contact_seen:
                if cue_id in ids and target_id in ids:
                    first_target_contact_seen = True
                else:
                    continue

            if cue_id in ids and dead_id in ids:
                return 1.0, "direct"

            a_depth = chain_depth.get(a_id)
            b_depth = chain_depth.get(b_id)
            if a_depth is not None and b_depth is None:
                chain_depth[b_id] = a_depth + 1
            elif b_depth is not None and a_depth is None:
                chain_depth[a_id] = b_depth + 1

            dead_depth = chain_depth.get(dead_id)
            if dead_depth is not None and 1 <= dead_depth <= max_depth:
                return 1.0, "indirect"

        return 0.0, None

    def get_route_robustness(
            self,
            base_score,
            shot: pt.System,
            target_id,
            pocket_id,
            target_region,
            config: ShotScoreConfig = None,
    ):
        action = ShotAction(shot.cue.phi, shot.cue.V0, shot.cue.a, shot.cue.b)

        jittered_actions = self.generate_truncated_actions(
            action,
            n_samples=config.noise_n_samples,
            scales=config.noise_scales,
            max_attempts=config.noise_max_attempts,
        )

        base_shot = shot.copy()
        pt.simulate(base_shot, inplace=True, continuous=False)
        base_quality = (
                self.get_position_quality(base_shot, target_region, k=config.position_k)
                * self.get_cue_route_quality(
                base_shot, target_id,
                k_1=config.cue_route_collision_k,
                k_2=config.cue_route_corner_k,
                k_3=config.cue_route_cushion_k
            ) * self.get_target_route_quality(
                base_shot, target_id, pocket_id,
                k_b=config.target_collision_k,
                k_c=config.target_cushion_k
            )
        )

        qualities = []
        for ja in jittered_actions:
            new_phi, new_V0, new_a, new_b = ja
            new_shot = shot.copy()
            new_shot.cue.set_state(V0=new_V0, phi=new_phi, a=new_a, b=new_b)
            pt.simulate(new_shot, inplace=True, continuous=False)
            if (pt.ruleset.get_id_of_first_ball_hit(new_shot) != target_id or
                    pt.ruleset.is_ball_pocketed(new_shot, "cue")):
                continue
            quality = (
                    self.get_position_quality(new_shot, target_region, k=config.position_k)
                    * self.get_cue_route_quality(
                new_shot, target_id,
                k_1=config.cue_route_collision_k,
                k_2=config.cue_route_corner_k,
                k_3=config.cue_route_cushion_k
            ) * self.get_target_route_quality(
                new_shot, target_id, pocket_id,
                k_b=config.target_collision_k,
                k_c=config.target_cushion_k
            )
            )
            qualities.append(quality)

        if base_score <= 0 or base_quality <= 0 or len(qualities) == 0:
            self.route_robustness = 0.0
        else:
            self.route_robustness = max(min(np.mean(qualities) / base_quality, 1), 0)

        return self.route_robustness

    def evaluate_shot(self, shot: pt.System, target_id, pocket_id, target_region,
                      strict_mode, config, route_type="direct") -> float:
        if not isinstance(config, ShotScoreConfig):
            raise ValueError(f"config must be instance of ShotScoreConfig, got {type(config)!r} instead.")
        position_k = config.position_k
        target_collision_k = config.target_collision_k
        target_cushion_k = config.target_cushion_k
        cue_action_k = config.cue_action_k
        cue_route_corner_k = config.cue_route_corner_k
        cue_route_collision_k = config.cue_route_collision_k
        cue_route_cushion_k = config.cue_route_cushion_k
        near_miss_k = config.near_miss_k
        near_miss_limit = config.near_miss_limit

        b4 = time.perf_counter()
        shot = shot.copy()
        pt.simulate(shot, inplace=True)
        self.diagnostics["pt_simulate"] += time.perf_counter() - b4

        b4 = time.perf_counter()
        if not self.is_shot_valid(shot, target_id, pocket_id, strict_mode):
            self.valid = False
            self.score = 0.0
            return 0.0
        self.valid = True

        self.position_quality = self.get_position_quality(shot, target_region, k=position_k)
        self.target_route_quality = self.get_target_route_quality(shot, target_id, pocket_id, k_b=target_collision_k,
                                                                  k_c=target_cushion_k, route_type=route_type)
        self.cue_action_ease = self.get_cue_action_ease(shot, k=cue_action_k)
        self.cue_route_quality = self.get_cue_route_quality(shot, target_id, k_1=cue_route_collision_k,
                                                            k_2=cue_route_corner_k, k_3=cue_route_cushion_k)
        self.score = (self.position_quality * self.target_route_quality
                      * self.cue_action_ease * self.cue_route_quality)

        if not strict_mode:
            self.target_near_miss_quality = self.get_target_near_miss_quality(shot, target_id, pocket_id, k=near_miss_k,
                                                                              limit=near_miss_limit)
            self.score *= self.target_near_miss_quality
        else:
            self.target_near_miss_quality = 1

        self.score = np.clip(self.score, 0, 1)
        self.diagnostics["score_calc"] += time.perf_counter() - b4
        return self.score

    def evaluate_breakout(self, shot: pt.System, target_id, pocket_id, breakout_target: BreakoutTarget,
                          strict_mode, config, route_type="direct") -> float:
        if not isinstance(config, ShotScoreConfig):
            raise ValueError(f"config must be instance of ShotScoreConfig, got {type(config)!r} instead.")
        if not isinstance(breakout_target, BreakoutTarget):
            raise ValueError(f"breakout_target must be BreakoutTarget, got {type(breakout_target)!r} instead.")
        if shot.cue.V0 < breakout_target.min_v0:
            self.valid = False
            self.score = 0.0
            return 0.0

        b4 = time.perf_counter()
        shot = shot.copy()
        pt.simulate(shot, inplace=True)
        self.diagnostics["pt_simulate"] += time.perf_counter() - b4

        b4 = time.perf_counter()
        if not self.is_shot_valid(shot, target_id, pocket_id, strict_mode):
            self.valid = False
            self.score = 0.0
            return 0.0
        self.valid = True

        self.position_quality = np.nan
        self.target_route_quality = self.get_target_route_quality(
            shot,
            target_id,
            pocket_id,
            k_b=config.target_collision_k,
            k_c=config.target_cushion_k,
            route_type=route_type,
        )
        self.cue_action_ease = self.get_cue_action_ease(shot, k=config.cue_action_k)
        self.cue_route_quality = 1.0

        collision_quality, _ = self.breakout_collision_quality(shot, target_id, breakout_target)
        if strict_mode:
            self.target_near_miss_quality = 1.0
            self.breakout_quality = collision_quality
        else:
            self.target_near_miss_quality = self.get_target_near_miss_quality(
                shot,
                target_id,
                pocket_id,
                k=config.near_miss_k,
                limit=config.near_miss_limit,
            )
            near_miss_quality = self.get_breakout_near_miss_quality(
                shot,
                breakout_target,
                k=config.near_miss_k,
            )
            path_region_quality = self.get_breakout_path_region_quality(
                shot,
                breakout_target,
                k=config.position_k,
            )
            self.breakout_quality = max(
                collision_quality,
                near_miss_quality,
                breakout_target.path_region_weight * path_region_quality,
            )

        self.score = (
                self.target_route_quality
                * self.cue_action_ease
                * self.target_near_miss_quality
                * self.breakout_quality
        )
        self.score = np.clip(self.score, 0, 1)
        self.diagnostics["score_calc"] += time.perf_counter() - b4
        return self.score

    def evaluate_shot_norm(self, shot: pt.System, target_id, pocket_id, target_region,
                           route_type="direct", config=None):
        if config is None:
            config = self.config
        return self.evaluate_shot(shot, target_id, pocket_id, target_region, False, route_type=route_type, config=config)

    def evaluate_shot_strict(self, shot: pt.System, target_id, pocket_id, target_region,
                             route_type="direct", config=None):
        if config is None:
            config = self.config
        return self.evaluate_shot(shot, target_id, pocket_id, target_region, route_type=route_type, strict_mode=True, config=config)

    def evaluate_breakout_norm(self, shot: pt.System, target_id, pocket_id, breakout_target: BreakoutTarget,
                               route_type="direct", config=None):
        if config is None:
            config = self.config
        return self.evaluate_breakout(shot, target_id, pocket_id, breakout_target, False, route_type=route_type, config=config)

    def evaluate_breakout_strict(self, shot: pt.System, target_id, pocket_id, breakout_target: BreakoutTarget,
                                 route_type="direct", config=None):
        if config is None:
            config = self.config
        return self.evaluate_breakout(shot, target_id, pocket_id, breakout_target, True, route_type=route_type, config=config)

    def evaluate_request_norm(self, shot: pt.System, search_request: SearchRequest, config=None):
        if config is None:
            config = self.config
        if self._objective_value(search_request.objective) == ShotObjective.BREAKOUT.value:
            return self.evaluate_breakout_norm(
                shot,
                search_request.target_ball_id,
                search_request.target_pocket_id,
                search_request.breakout_target,
                route_type=search_request.route_type,
                config=config,
            )
        return self.evaluate_shot_norm(
            shot,
            search_request.target_ball_id,
            search_request.target_pocket_id,
            search_request.position_target,
            route_type=search_request.route_type,
            config=config,
        )

    def evaluate_request_strict(self, shot: pt.System, search_request: SearchRequest, config=None):
        if config is None:
            config = self.config
        if self._objective_value(search_request.objective) == ShotObjective.BREAKOUT.value:
            return self.evaluate_breakout_strict(
                shot,
                search_request.target_ball_id,
                search_request.target_pocket_id,
                search_request.breakout_target,
                route_type=search_request.route_type,
                config=config,
            )
        return self.evaluate_shot_strict(
            shot,
            search_request.target_ball_id,
            search_request.target_pocket_id,
            search_request.position_target,
            route_type=search_request.route_type,
            config=config,
        )

    def evaluate_shot_final(
            self,
            shot: pt.System,
            target_id,
            pocket_id,
            target_region,
            route_type="direct",
            config: ShotScoreConfig = None,
    ):
        if config is None:
            config = self.config
        robustness_shot = shot.copy()
        base_score = self.evaluate_shot_strict(shot, target_id, pocket_id, target_region, route_type=route_type, config=config)
        robustness = self.get_route_robustness(
            base_score,
            robustness_shot,
            target_id,
            pocket_id,
            target_region,
            config=config,
        )
        self.score = base_score * (1 - config.robustness_weight + config.robustness_weight * robustness)
        self.score = np.clip(self.score, 0, 1)
        return self.score

    def evaluate_breakout_final(
            self,
            shot: pt.System,
            target_id,
            pocket_id,
            breakout_target: BreakoutTarget,
            route_type="direct",
            config: ShotScoreConfig = None,
    ):
        if config is None:
            config = self.config
        self.score = self.evaluate_breakout_strict(
            shot,
            target_id,
            pocket_id,
            breakout_target,
            route_type=route_type,
            config=config,
        )
        return self.score

    def evaluate_request_final(self, shot: pt.System, search_request: SearchRequest, config=None):
        if config is None:
            config = self.config
        if self._objective_value(search_request.objective) == ShotObjective.BREAKOUT.value:
            return self.evaluate_breakout_final(
                shot,
                search_request.target_ball_id,
                search_request.target_pocket_id,
                search_request.breakout_target,
                route_type=search_request.route_type,
                config=config,
            )
        return self.evaluate_shot_final(
            shot,
            search_request.target_ball_id,
            search_request.target_pocket_id,
            search_request.position_target,
            route_type=search_request.route_type,
            config=config,
        )

    def print_score_breakdown(self):
        print(f"score: {self.score:.3f} \nBreakdown:")
        print(f"position_quality: {self.position_quality:.4f}")
        print(f"breakout_quality: {self.breakout_quality:.4f}")
        print(f"target_route_quality: {self.target_route_quality:.4f}")
        print(f"cue_action_ease: {self.cue_action_ease:.4f}")
        print(f"cue_route_quality: {self.cue_route_quality:.4f}")
        if self.target_near_miss_quality != float('nan'):
            print(f"target_near_miss_quality: {self.target_near_miss_quality:.4f}")
        if self.route_robustness != float('nan'):
            print(f"base_score: {self.score / (1 - self.config.robustness_weight +
                                               self.config.robustness_weight * self.route_robustness):.3f}")
            print(f"route_robustness: {self.route_robustness}")
        return {
            "score": self.score,
            "position_quality": self.position_quality,
            "breakout_quality": self.breakout_quality,
            "target_route_quality": self.target_route_quality,
            "cue_action_ease": self.cue_action_ease,
            "cue_route_quality": self.cue_route_quality,
            "target_near_miss_quality": self.target_near_miss_quality,
            "route_robustness": self.route_robustness,
        }

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
