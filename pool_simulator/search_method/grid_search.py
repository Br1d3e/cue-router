import numpy as np
import pooltool as pt

try:
    from ..action import ShotAction
    from ..evaluation import ShotEvaluation, ShotScoreConfig
    from ..geometry import Geometry
except ImportError:
    from action import ShotAction
    from evaluation import ShotEvaluation, ShotScoreConfig
    from geometry import Geometry


class GridSearch:
    SIDE_SEEDS = [
        (0.0, 0.0),  # stun
        (0.0, 0.5),  # follow
        (0.0, -0.5),  # screw
        (-0.5, 0.0),  # right
        (0.5, 0.0),  # left
        (-0.4, 0.4),  # right follow
        (0.4, 0.4),  # left follow
        (-0.4, -0.4),  # right draw
        (0.4, -0.4),  # left draw
    ]

    V0_VALUES = [1.5, 3.0, 4.5]
    PHI_OFFSETS = np.linspace(-8, 8, num=1 + 8 * 4)

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
        shot, target_ball_id, target_pocket_id, target_rect = search_request
        target_ball = shot.balls.get(target_ball_id)
        cue_ball = shot.balls.get(shot.cue.cue_ball_id)

        if target_ball is None:
            raise KeyError(f"target ball not found: {target_ball_id}")
        if cue_ball is None:
            raise KeyError(f"cue ball not found: {shot.cue.cue_ball_id}")

        target_ball_pos = target_ball.xyz[:2]
        cue_ball_pos = cue_ball.xyz[:2]
        phi0 = Geometry.aim_phi(target_pocket_id, target_ball_pos, cue_ball_pos)
        if phi0 is None:
            print(f"shot not reachable. cue_ball at {cue_ball_pos} | target_ball at {target_ball_pos} "
                  f"| target_pocket: {target_pocket_id}")
            return np.asarray([], dtype=float), np.asarray([], dtype=object)

        scores = []
        policies = []

        for a, b in side_seeds:
            for V0 in v0_values:
                for d_phi in phi_offsets:
                    phi = (phi0 + d_phi) % 360

                    new_shot = shot.copy()
                    new_shot.cue.set_state(V0=V0, a=a, b=b, phi=phi, theta=0.0)

                    shot_evaluator = ShotEvaluation(config=eval_config)
                    score = shot_evaluator.evaluate_shot_strict(new_shot, target_ball_id, target_pocket_id, target_rect)
                    scores.append(score)
                    policies.append(ShotAction(phi, V0, a, b))

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
