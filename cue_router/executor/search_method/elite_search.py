import numpy as np
from scipy.stats import qmc

from cue_router.core.geometry import Geometry
from cue_router.core.request import ShotObjective
from cue_router.executor.action import ShotAction
from cue_router.executor.shot_evaluation import ShotEvaluation, ShotScoreConfig


class EliteSearch:
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

    def simulate(self, policy: ShotAction | np.ndarray):
        if isinstance(policy, ShotAction):
            phi, V0, a, b = policy.as_array()
        elif isinstance(policy, np.ndarray) or isinstance(policy, list) or isinstance(policy, tuple):
            phi, V0, a, b = policy
        else:
            raise TypeError(f"policy must be ShotAction, np.ndarray, list, or tuple, got {type(policy)!r} instead.")
        new_shot = self.shot.copy()
        new_shot.cue.set_state(V0=V0, a=a, b=b, phi=phi, theta=0.0)

        self.evaluator.evaluate_request_strict(new_shot, self.search_request, config=self.eval_config)
        return self.evaluator.score

    def simulate_policies(self, policies):
        scores = []
        for policy in policies:
            scores.append(self.simulate(policy))
        return np.asarray(scores, dtype=float)

    def get_aim_phi0(self):
        target_ball = self.shot.balls.get(self.target_ball_id)
        cue_ball = self.shot.balls.get(self.shot.cue.cue_ball_id)

        if target_ball is None:
            raise KeyError(f"target ball not found: {self.target_ball_id}")
        if cue_ball is None:
            raise KeyError(f"cue ball not found: {self.shot.cue.cue_ball_id}")

        target_ball_pos = target_ball.xyz[:2]
        cue_ball_pos = cue_ball.xyz[:2]
        if self.route_value == "bank":
            return Geometry.aim_phi_bank(self.target_pocket_id, target_ball_pos, cue_ball_pos)
        return Geometry.aim_phi(self.target_pocket_id, target_ball_pos, cue_ball_pos)

    def search(self, max_iterations=1000, n_rounds=5, elite_frac=0.2):
        phi0 = self.get_aim_phi0()
        if phi0 is None:
            print(f"shot not reachable. "
                  f"| target_pocket: {self.target_pocket_id}")
            return np.asarray([], dtype=float), np.asarray([], dtype=object)

        round_budget = max(1, max_iterations // n_rounds)
        all_samples = []
        all_scores = []
        all_policies = []
        for _ in range(n_rounds):
            n_exploit = int(round_budget * elite_frac) if len(all_scores) > 0 else 0
            n_explore = round_budget - n_exploit

            explore = self.rough_unique_samples(self.LHS_5d_samples(n_explore))
            exploit = self.sample_elite(all_scores, all_samples, n_exploit)
            if exploit is None:
                round_samples = explore
            else:
                round_samples = np.concatenate((explore, exploit))
            if len(round_samples) == 0:
                continue

            round_policies = self.samples_to_policies(round_samples, phi0)
            round_scores = self.simulate_policies(round_policies)

            all_samples.extend(round_samples)
            all_scores.extend(round_scores.tolist())
            all_policies.extend(round_policies)

        self.scores = np.asarray(all_scores, dtype=float)
        self.policies = np.asarray(all_policies, dtype=object)
        return self.scores, self.policies

    def samples_to_policies(self, samples, phi0):
        policies = []
        for phi_offset, V0, a, b in np.asarray(samples, dtype=float).reshape(-1, 4):
            policies.append(ShotAction(phi0 + phi_offset, V0, a, b, route_type=self.route_value))
        return np.asarray(policies, dtype=object)


    def LHS_5d_samples(self, size):
        if size <= 0:
            return np.empty((0, 4), dtype=float)
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
        p = self.spin_family_probabilities()
        if p is None:
            spin_family_idx = np.floor(scaled_sample[:, 2]).astype(int)
            spin_family_idx = np.clip(spin_family_idx, 0, len(self.SIDE_SEEDS) - 1)
        else:
            spin_family_idx = self.rng.choice(len(self.SIDE_SEEDS), size=size, p=p)
        a_b = self.SIDE_SEEDS[spin_family_idx] + scaled_sample[:, 3:5]
        valid_spin = ShotAction.valid_spin_mask(a_b)

        samples = np.column_stack((phi_offsets, V0, a_b))
        return samples[valid_spin]

    @staticmethod
    def _valid_sample_mask(samples):
        return (
                ShotAction.valid_V0_mask(samples[:, 1])
                & ShotAction.valid_spin_mask(samples[:, 2:])
        )

    def sample_elite(self, scores, samples, size, top_k=30):
        if size <= 0 or len(scores) == 0:
            return None
        scores = np.asarray(scores, dtype=float)
        samples = np.asarray(samples, dtype=float).reshape(-1, 4)
        valid_score_mask = scores > 0
        scores = scores[valid_score_mask]
        samples = samples[valid_score_mask]
        if len(scores) == 0:
            return None

        parent_scores, parent_samples = self._top_unique(scores, samples, top_k)
        if len(parent_scores) == 0:
            return None

        accepted = []
        weights = parent_scores / np.sum(parent_scores)
        max_attempts = 8 * size
        while len(accepted) < size and max_attempts > 0:
            batch_size = min(size - len(accepted), max_attempts)
            parent_idx = self.rng.choice(len(parent_samples), size=batch_size, p=weights)
            candidates = self._sample_parent_children(parent_samples[parent_idx])
            valid_mask = self._valid_sample_mask(candidates)
            accepted.extend(candidates[valid_mask])
            max_attempts -= batch_size

        if len(accepted) == 0:
            return None
        return np.asarray(accepted[:size], dtype=float)

    @staticmethod
    def _rough_equal_sample(sample_a, sample_b):
        return (
            (-2 <= sample_a[0] - sample_b[0] <= 2)
            and (-0.5 < sample_a[1] - sample_b[1] < 0.5)
            and (-0.15 < sample_a[2] - sample_b[2] < 0.15)
            and (-0.15 < sample_a[3] - sample_b[3] < 0.15)
        )

    @staticmethod
    def rough_unique_samples(samples):
        samples = np.asarray(samples, dtype=float).reshape(-1, 4)
        keep = np.ones(len(samples), dtype=bool)
        for idx in range(len(samples)):
            if not keep[idx]:
                continue
            diff = samples[idx + 1:] - samples[idx]
            rough_equal = (
                (diff[:, 0] >= -2.0)
                & (diff[:, 0] <= 2.0)
                & (diff[:, 1] > -0.5)
                & (diff[:, 1] < 0.5)
                & (diff[:, 2] > -0.15)
                & (diff[:, 2] < 0.15)
                & (diff[:, 3] > -0.15)
                & (diff[:, 3] < 0.15)
            )
            keep[idx + 1:][rough_equal] = False
        return samples[keep]

    @staticmethod
    def _top_unique(scores, samples, top_k):
        top_idx = np.argsort(scores)[::-1][:top_k]
        parent_scores = np.array(scores, dtype=float)[top_idx]
        parent_samples = np.array(samples, dtype=float)[top_idx]
        return parent_scores, parent_samples

    def _top_rough_unique(self, scores, samples, top_k):
        parent_scores = []
        parent_samples = []
        for idx in np.argsort(scores)[::-1]:
            sample = samples[idx]
            if any(self._rough_equal_sample(sample, parent) for parent in parent_samples):
                continue
            parent_scores.append(scores[idx])
            parent_samples.append(sample)
            if len(parent_samples) == top_k:
                break
        return np.asarray(parent_scores, dtype=float), np.asarray(parent_samples, dtype=float)

    def _sample_parent_children(self, parents, scales=None):
        parents = np.asarray(parents, dtype=float).reshape(-1, 4)
        if scales is None:
            scales = np.asarray([1.0, 0.5, 0.1, 0.1], dtype=float)
        children = parents + self.rng.normal(0.0, scales, size=parents.shape)
        if "bank" in self.route_value:
            children[:, 0] = np.clip(children[:, 0], -12.0, 12.0)
        else:
            children[:, 0] = np.clip(children[:, 0], -8.0, 8.0)
        return children

    def spin_family_probabilities(self):
        objective = self.search_request.objective.value if hasattr(self.search_request.objective, "value") else self.search_request.objective
        if objective != ShotObjective.BREAKOUT.value:
            return None
        breakout_target = self.search_request.breakout_target
        if breakout_target is None or breakout_target.preferred_side not in ("left", "right"):
            return None
        side = self.SIDE_SEEDS[:, 0]
        weights = np.ones(len(side), dtype=float)
        if breakout_target.preferred_side == "left":
            weights[side > 0] = 3.0
            weights[side < 0] = 0.5
        else:
            weights[side < 0] = 3.0
            weights[side > 0] = 0.5
        return weights / np.sum(weights)
