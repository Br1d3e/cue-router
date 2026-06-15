import numpy as np
import pooltool as pt

try:
    from ..action import ShotAction
    from ..evaluation import ShotEvaluation, ShotScoreConfig
except ImportError:
    from action import ShotAction
    from evaluation import ShotEvaluation, ShotScoreConfig


class PatternSearch:
    STEP = np.asarray([0.5, 0.5, 0.1, 0.1], dtype=float)
    DIRECTIONS = [
        # single-parameter probes
        (1, 0, 0, 0),
        (-1, 0, 0, 0),
        (0, 1, 0, 0),
        (0, -1, 0, 0),
        (0, 0, 1, 0),
        (0, 0, -1, 0),
        (0, 0, 0, 1),
        (0, 0, 0, -1),

        # coupled probes
        (1, 0, 1, 0),
        (-1, 0, -1, 0),
        (1, 0, -1, 0),
        (-1, 0, 1, 0),
        (2, 0, 1, 0),
        (-2, 0, -1, 0),
        (2, 0, -1, 0),
        (-2, 0, 1, 0),
        (1, 0, 0, 1),
        (-1, 0, 0, -1),
        (1, 0, 0, -1),
        (-1, 0, 0, 1),
        (0, 1, 1, 0),
        (0, -1, -1, 0),
        (0, 1, -1, 0),
        (0, -1, 1, 0),
        (0, 1, 0, 1),
        (0, -1, 0, -1),
        (0, 1, 0, -1),
        (0, -1, 0, 1),
    ]
    IMPROVEMENT_EPS = 0.02
    MAX_SEARCHES = 3

    def __init__(self, search_request, eval_config: ShotScoreConfig):
        self.shot = search_request.shot
        self.target_ball_id = search_request.target_ball_id
        self.target_pocket_id = search_request.target_pocket_id
        self.target_rect = search_request.target_region
        self.eval_config = eval_config

    def search_once(self, init_score, policy: ShotAction, directions=DIRECTIONS,
                    step=STEP, improv_eps=IMPROVEMENT_EPS,
                    eval_config=None):
        best_score = init_score
        best_policy = policy
        phi, V0, a, b = policy.as_array()

        shot_evaluator = ShotEvaluation(config=eval_config)

        for direction in directions:
            direction = np.asarray(direction)
            d_phi, d_V0, d_a, d_b = direction * step
            new_phi = phi + d_phi
            new_V0 = V0 + d_V0
            new_a = a + d_a
            new_b = b + d_b

            if not ShotAction.is_spin_possible(new_a, new_b):
                continue
            if not ShotAction.is_V0_possible(new_V0):
                continue

            new_shot = self.shot.copy()
            new_shot.cue.set_state(V0=new_V0, a=new_a, b=new_b, phi=new_phi, theta=0.0)

            new_score = shot_evaluator.evaluate_shot_norm(
                new_shot,
                self.target_ball_id,
                self.target_pocket_id,
                self.target_rect,
            )
            new_policy = ShotAction(new_phi, new_V0, new_a, new_b)

            if new_score > best_score + improv_eps:
                best_score = new_score
                best_policy = new_policy

        return best_score, best_policy

    def search_n_rounds(self, init_score, policy: ShotAction, n_rounds=MAX_SEARCHES, improv_eps=IMPROVEMENT_EPS):
        best_score = init_score
        best_policy = policy

        for _ in range(n_rounds):
            new_score, new_policy = self.search_once(best_score, best_policy, improv_eps=improv_eps, eval_config=self.eval_config)
            if new_score > best_score + improv_eps:
                best_score = new_score
                best_policy = new_policy
            else:
                break

        return best_score, best_policy

