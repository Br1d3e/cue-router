import numpy as np

from cue_router.core.geometry import Geometry
from cue_router.core.request import ShotObjective
from cue_router.executor.action import ShotAction
from cue_router.executor.shot_evaluation import ShotEvaluation, ShotScoreConfig


class GridSearch:
    SIDE_SEEDS = [
        (0.0, 0.0),  # stun
        (0.0, 0.4),  # follow
        (0.0, -0.4),  # screw
        (-0.4, 0.0),  # right
        (0.4, 0.0),  # left
        (-0.3, 0.3),  # right follow
        (0.3, 0.3),  # left follow
        (-0.3, -0.3),  # right draw
        (0.3, -0.3),  # left draw
    ]

    V0_VALUES = [1.5, 2.5, 3.5]
    PHI_OFFSETS = np.linspace(-7, 7, num=1 + 7 * 8)

    EVAL_CONFIG = ShotScoreConfig(
        position_k=2.0,
        target_collision_k=1.1,
        target_cushion_k=1.2,
        cue_action_k=1.5,
        cue_route_corner_k=0.4,
        cue_route_collision_k=0.5,
        cue_route_cushion_k=0.005,
        near_miss_k=30.0,
        near_miss_limit=0.15,
    )

    def __init__(self, search_request, eval_config:ShotScoreConfig=None):
        self.search_request = search_request
        self.scores = np.asarray([], dtype=float)
        self.policies = np.asarray([], dtype=object)
        self.eval_config = eval_config or self.EVAL_CONFIG

    @staticmethod
    def _search(search_request, side_seeds=SIDE_SEEDS,
                v0_values=V0_VALUES, phi_offsets=PHI_OFFSETS,
                eval_config=None):
        shot = search_request.shot
        target_ball_id = search_request.target_ball_id
        target_pocket_id = search_request.target_pocket_id
        route_type = search_request.route_type
        route_value = ShotEvaluation._route_value(route_type)
        objective = search_request.objective.value if hasattr(search_request.objective, "value") else search_request.objective
        min_v0 = 0.25
        if objective == ShotObjective.BREAKOUT.value and search_request.breakout_target is not None:
            min_v0 = search_request.breakout_target.min_v0
        target_ball = shot.balls.get(target_ball_id)
        cue_ball = shot.balls.get(shot.cue.cue_ball_id)

        if target_ball is None:
            raise KeyError(f"target ball not found: {target_ball_id}")
        if cue_ball is None:
            raise KeyError(f"cue ball not found: {shot.cue.cue_ball_id}")

        target_ball_pos = target_ball.xyz[:2]
        cue_ball_pos = cue_ball.xyz[:2]
        if route_value == "bank":
            phi0 = Geometry.aim_phi_bank(
                target_pocket_id,
                target_ball_pos,
                cue_ball_pos,
                rail=search_request.bank_rail,
            )
            phi_offsets = np.linspace(-12, 12, num=1 + 12 * 6)
        else:
            phi0 = Geometry.aim_phi(target_pocket_id, target_ball_pos, cue_ball_pos)
        if phi0 is None:
            print(f"shot not reachable. cue_ball at {cue_ball_pos} | target_ball at {target_ball_pos} "
                  f"| target_pocket: {target_pocket_id}")
            return np.asarray([], dtype=float), np.asarray([], dtype=object)

        scores = []
        policies = []
        shot_evaluator = ShotEvaluation(config=eval_config)
        v0_values = [V0 for V0 in v0_values if V0 >= min_v0]

        for a, b in side_seeds:
            for V0 in v0_values:
                for d_phi in phi_offsets:
                    phi = (phi0 + d_phi) % 360

                    new_shot = shot.copy()
                    new_shot.cue.set_state(V0=V0, a=a, b=b, phi=phi, theta=0.0)

                    score = shot_evaluator.evaluate_request_strict(new_shot, search_request, config=eval_config)
                    scores.append(score)
                    policies.append(ShotAction(phi, V0, a, b, route_type=route_value))

        return np.asarray(scores, dtype=float), np.asarray(policies, dtype=object)

    def search(self):
        self.scores, self.policies = self._search(
            self.search_request,
            self.SIDE_SEEDS,
            self.V0_VALUES,
            self.PHI_OFFSETS,
            self.eval_config,
        )
        return self.scores, self.policies
