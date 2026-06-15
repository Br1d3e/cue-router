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


class StochasticSearch:
    SIDE_SEEDS = np.asarray([
        (0.0, 0.0),  # stun
        (0.0, 0.4),  # follow
        (0.0, -0.4),  # screw
        (-0.4, 0.0),  # right
        (0.4, 0.0),  # left
        (-0.3, 0.3),  # right follow
        (0.3, 0.3),  # left follow
        (-0.3, -0.3),  # right draw
        (0.3, -0.3),  # left draw
    ], dtype=float)

    V0_VALUES = [1.5, 2.5, 3.5]

    PHI_OFFSETS = np.linspace(-7, 7, num=1 + 7 * 8)

    EVAL_CONFIG = ShotScoreConfig(
        position_k=2.0,
        target_collision_k=1.1,
        target_cushion_k=1.2,
        cue_action_k=1.4,
        cue_route_corner_k=0.3,
        cue_route_collision_k=0.4,
        cue_route_cushion_k=0.005,
        near_miss_k=30.0,
        near_miss_limit=0.15,
        robustness_weight=0.2,
        noise_n_samples=16,
        noise_scales=(0.2, 0.5, 0.1, 0.1),
        noise_max_attempts=128,
    )

    def __init__(self, search_request, eval_config: ShotScoreConfig = None, seed=42):
        self.search_request = search_request
        self.shot = search_request.shot
        self.target_ball_id = search_request.target_ball_id
        self.target_pocket_id = search_request.target_pocket_id
        self.target_rect = search_request.target_rect
        self.scores = np.asarray([], dtype=float)
        self.policies = np.asarray([], dtype=object)
        self.rng = np.random.default_rng(seed)
        self.eval_config = eval_config or self.EVAL_CONFIG
        self.evaluator = ShotEvaluation(config=self.eval_config)

    @classmethod
    def choose_spin_family(cls, size=1, seed=42):
        rng = np.random.default_rng(seed)
        side_index = rng.choice(len(cls.SIDE_SEEDS), size)
        return cls.SIDE_SEEDS[side_index]

    def search(self, max_iterations=1200):
        target_ball = self.shot.balls.get(self.target_ball_id)
        cue_ball = self.shot.balls.get(self.shot.cue.cue_ball_id)

        if target_ball is None:
            raise KeyError(f"target ball not found: {self.target_ball_id}")
        if cue_ball is None:
            raise KeyError(f"cue ball not found: {self.shot.cue.cue_ball_id}")

        target_ball_pos = target_ball.xyz[:2]
        cue_ball_pos = cue_ball.xyz[:2]
        phi0 = Geometry.aim_phi(self.target_pocket_id, target_ball_pos, cue_ball_pos)
        if phi0 is None:
            print(f"shot not reachable. cue_ball at {cue_ball_pos} | target_ball at {target_ball_pos} "
                  f"| target_pocket: {self.target_pocket_id}")
            return np.asarray([], dtype=float), np.asarray([], dtype=object)

        scores = []
        policies = []

        # family_spins = self.choose_spin_family(16)
        # v0_values = self.uniform(1, 5, 8)
        # v0_values = self.V0_VALUES

        # it_count = 0
        #
        # for spin in family_spins:
        #     a0, b0 = spin
        #     a = a0 + self.normal(0, 0.1)
        #     b = b0 + self.normal(0, 0.1)
        #
        #     phi_offsets = self.uniform(-7, 7, 10)
        #     for V0 in v0_values:
        #         for d_phi in phi_offsets:
        #             if it_count > max_iterations:
        #                 self.scores = np.asarray(scores, dtype=float)
        #                 self.policies = np.asarray(policies, dtype=object)
        #                 return self.scores, self.policies
        #             phi = phi0 + d_phi
        #             policy = ShotAction(phi, V0, a, b)
        #
        #             new_shot = self.shot.copy()
        #             new_shot.cue.set_state(V0=V0, a=a, b=b, phi=phi, theta=0.0)
        #
        #             self.evaluator.evaluate_shot_strict(
        #                 new_shot,
        #                 self.target_ball_id,
        #                 self.target_pocket_id,
        #                 self.target_rect
        #             )
        #             score = self.evaluator.score
        #
        #             scores.append(score)
        #             policies.append(policy)
        #
        #             it_count += 1

        for i in range(max_iterations):
            spin = self.rng.choice(self.SIDE_SEEDS)
            a0, b0 = spin
            a = a0 + self.rng.normal(0, 0.1)
            b = b0 + self.rng.normal(0, 0.1)
            if not ShotAction.is_spin_possible(a, b):
                continue

            phi_offset = self.rng.uniform(-8, 8)
            phi = phi0 + phi_offset

            if self.rng.random() < 0.7:
                V0 = self.rng.choice([1.5, 2.5, 3.5, 4.5])
            else:
                V0 = self.rng.uniform(1.0, 5.0)

            policy = ShotAction(phi, V0, a, b)
            new_shot = self.shot.copy()
            new_shot.cue.set_state(V0=V0, a=a, b=b, phi=phi, theta=0.0)

            self.evaluator.evaluate_shot_strict(
                new_shot,
                self.target_ball_id,
                self.target_pocket_id,
                self.target_rect
            )
            score = self.evaluator.score

            scores.append(score)
            policies.append(policy)

        self.scores = np.asarray(scores, dtype=float)
        self.policies = np.asarray(policies, dtype=object)
        return self.scores, self.policies
