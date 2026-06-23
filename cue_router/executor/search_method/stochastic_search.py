import numpy as np
from scipy.stats import qmc

from cue_router.core.geometry import Geometry
from cue_router.core.request import ShotObjective
from cue_router.executor.action import ShotAction
from cue_router.executor.shot_evaluation import ShotEvaluation, ShotScoreConfig


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
        self.target_region = search_request.position_target
        self.route_type = search_request.route_type
        self.route_value = ShotEvaluation._route_value(self.route_type)
        self.scores = np.asarray([], dtype=float)
        self.policies = np.asarray([], dtype=object)
        self.rng = np.random.default_rng(seed)
        self.eval_config = eval_config or self.EVAL_CONFIG
        self.evaluator = ShotEvaluation(config=self.eval_config)
        self.candidate_simulations = 0
        self.valid_potting_candidates = 0

    @classmethod
    def choose_spin_family(cls, size=1, seed=42, p=None):
        rng = np.random.default_rng(seed)
        side_index = rng.choice(len(cls.SIDE_SEEDS), size, p=p)
        return cls.SIDE_SEEDS[side_index]

    @staticmethod
    def spin_family_probabilities(side_seeds, search_request):
        objective = search_request.objective.value if hasattr(search_request.objective, "value") else search_request.objective
        if objective != ShotObjective.BREAKOUT.value:
            return None
        breakout_target = search_request.breakout_target
        if breakout_target is None or breakout_target.preferred_side not in ("left", "right"):
            return None
        side_seeds = np.asarray(side_seeds, dtype=float)
        side = side_seeds[:, 0]
        weights = np.ones(len(side), dtype=float)
        if breakout_target.preferred_side == "left":
            weights[side > 0] = 3.0
            weights[side < 0] = 0.5
        else:
            weights[side < 0] = 3.0
            weights[side > 0] = 0.5
        return weights / np.sum(weights)

    def search(self, max_iterations=1200, sample_mode="random"):
        self.candidate_simulations = 0
        self.valid_potting_candidates = 0
        target_ball = self.shot.balls.get(self.target_ball_id)
        cue_ball = self.shot.balls.get(self.shot.cue.cue_ball_id)

        if target_ball is None:
            raise KeyError(f"target ball not found: {self.target_ball_id}")
        if cue_ball is None:
            raise KeyError(f"cue ball not found: {self.shot.cue.cue_ball_id}")

        if sample_mode not in ["random", "LHS"]:
            raise ValueError(f"Invalid sample_mode, {sample_mode!r}.")
        target_ball_pos = target_ball.xyz[:2]
        cue_ball_pos = cue_ball.xyz[:2]
        if self.route_value == "bank":
            phi0 = Geometry.aim_phi_bank(self.target_pocket_id, target_ball_pos, cue_ball_pos)
        else:
            phi0 = Geometry.aim_phi(self.target_pocket_id, target_ball_pos, cue_ball_pos)
        if phi0 is None:
            print(f"shot not reachable. cue_ball at {cue_ball_pos} | target_ball at {target_ball_pos} "
                  f"| target_pocket: {self.target_pocket_id}")
            return np.asarray([], dtype=float), np.asarray([], dtype=object)


        scores = []
        policies = []
        if sample_mode == "random":
            samples = self.random_samples(max_iterations)
        else:
            samples = self.LHS_5d_samples(max_iterations)
        candidate_policies = self.rough_unique_policies(samples, phi0)

        for policy in candidate_policies:
            new_shot = self.shot.copy()
            new_shot.cue.set_state(V0=policy.V0, a=policy.a, b=policy.b, phi=policy.phi, theta=0.0)

            self.evaluator.evaluate_request_strict(new_shot, self.search_request, config=self.eval_config)
            self.candidate_simulations += 1
            score = self.evaluator.score
            if score > 0:
                self.valid_potting_candidates += 1

            scores.append(score)
            policies.append(policy)

        self.scores = np.asarray(scores, dtype=float)
        self.policies = np.asarray(policies, dtype=object)

        print("Stochastic search eval time:\n", self.evaluator.diagnostics)
        return self.scores, self.policies


    def random_samples(self, size):
        p = self.spin_family_probabilities(self.SIDE_SEEDS, self.search_request)
        side_indices = self.rng.choice(len(self.SIDE_SEEDS), size, p=p)
        spins = self.SIDE_SEEDS[side_indices]
        spin_noise = self.rng.normal(0, 0.1, size=(size, 2))
        a_b = spins + spin_noise
        valid_spin = ShotAction.valid_spin_mask(a_b)

        phi_offsets = self.rng.uniform(-8, 8, size=size)
        discrete_v0 = self.rng.choice([1.5, 2.5, 3.5, 4.5], size)
        random_v0 = self.rng.uniform(1.0, 5.0, size=size)
        V0 = np.where(self.rng.random(size) < 0.7, discrete_v0, random_v0)

        samples = np.column_stack((phi_offsets, V0, a_b))
        return samples[valid_spin]

    def rough_unique_policies(self, samples, phi0):
        print(f"[Stochastic search]: given samples: {len(samples)}")
        policies = []
        for phi_offset, V0, a, b in np.asarray(samples, dtype=float).reshape(-1, 4):
            policy = ShotAction(phi0 + phi_offset, V0, a, b, route_type=self.route_value)
            if any(existing.rough_equal(policy) for existing in policies):
                continue
            policies.append(policy)
        print(f"[Stochastic search]: unique policies: {len(policies)}")
        return policies

    def random_sample(self):
        samples = self.random_samples(1)
        if len(samples) == 0:
            return None
        return samples[0]

    def LHS(self):
        spin = self.rng.choice(self.SIDE_SEEDS)
        a0, b0 = spin
        sampler = qmc.LatinHypercube(d=4, optimization="random-cd", rng=self.rng)
        sample = sampler.random(n=1)
        l_bounds = [-8, 1, -0.1, -0.1]
        u_bounds = [8, 5, 0.1, 0.1]
        if "bank" in self.route_value:
            l_bounds[0] = -12
            u_bounds[0] = 12
        scaled_sample = qmc.scale(sample, l_bounds, u_bounds).flatten()

        phi_offset, V0, d_a, d_b = scaled_sample
        a = a0 + d_a
        b = b0 + d_b
        if not ShotAction.is_spin_possible(a, b):
            return None

        return phi_offset, V0, a, b

    def LHS_5d_samples(self, size):
        sampler = qmc.LatinHypercube(d=5, optimization="random-cd", rng=self.rng)
        sample = sampler.random(n=size)
        l_bounds = [-8, 1, 0, -0.1, -0.1]
        u_bounds = [8, 5, 9, 0.1, 0.1]
        if "bank" in self.route_value:
            l_bounds[0] = -12
            u_bounds[0] = 12
        scaled_sample = qmc.scale(sample, l_bounds, u_bounds)

        phi_offsets = scaled_sample[:, 0]
        V0 = scaled_sample[:, 1]
        p = self.spin_family_probabilities(self.SIDE_SEEDS, self.search_request)
        if p is None:
            spin_family_idx = np.floor(scaled_sample[:, 2]).astype(int)
            spin_family_idx = np.clip(spin_family_idx, 0, len(self.SIDE_SEEDS) - 1)
        else:
            spin_family_idx = self.rng.choice(len(self.SIDE_SEEDS), size=size, p=p)
        a_b = self.SIDE_SEEDS[spin_family_idx] + scaled_sample[:, 3:5]
        valid_spin = ShotAction.valid_spin_mask(a_b)

        samples = np.column_stack((phi_offsets, V0, a_b))
        return samples[valid_spin]

    def LHS_5d(self):
        samples = self.LHS_5d_samples(1)
        if len(samples) == 0:
            return None
        return samples[0]
