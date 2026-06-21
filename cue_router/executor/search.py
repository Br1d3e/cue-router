from dataclasses import dataclass
import time

import numpy as np

from cue_router.core.request import SearchRequest
from cue_router.executor.search_method.grid_search import GridSearch
from cue_router.executor.search_method.pattern_search import PatternSearch
from cue_router.executor.search_method.stochastic_search import StochasticSearch
from cue_router.executor.shot_evaluation import ShotScoreConfig, ShotEvaluation

@dataclass
class SearchConfig:
    display_k: int = 5
    search_k: int = 20
    refine_k: int = 10
    precise: bool = True
    quick_mode: str = "grid"
    n_rounds: int = PatternSearch.MAX_SEARCHES
    stochastic_max_iter: int = 1200
    sample_mode: str = "random"
    seed: int = 42

    def __repr__(self):
        return (f"SearchConfig(\n"
                f"  display_k={self.display_k}, \n"
                f"  search_k={self.search_k}, \n"
                f"  refine_k={self.refine_k}, \n"
                f"  precise={self.precise}, \n"
                f"  quick_mode={self.quick_mode}, \n"
                f"  n_rounds={self.n_rounds}, \n"
                f"  stochastic_max_iter={self.stochastic_max_iter}, \n"
                f"  sample_mode={self.sample_mode}, \n"
                f"  seed={self.seed}, \n"
                f")")


class ScoresHelper:
    @staticmethod
    def _scores_np(scores):
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores, dtype=float)
        return scores

    @staticmethod
    def _policies_np(policies):
        if not isinstance(policies, np.ndarray):
            policies = np.array(policies, dtype=object)
        return policies

    @staticmethod
    def top_k_idx(scores, k=1):
        scores = ScoresHelper._scores_np(scores)
        if scores.size == 0:
            return np.asarray([], dtype=int)
        return np.argsort(scores)[::-1][:k]

    @staticmethod
    def _policy_key(policy):
        if hasattr(policy, "as_array"):
            return tuple(policy.as_array())
        return tuple(policy)

    @staticmethod
    def top_k_unique_idx(scores, policies, k=1):
        scores = ScoresHelper._scores_np(scores)
        policies = ScoresHelper._policies_np(policies)
        if scores.size == 0:
            return np.asarray([], dtype=int)

        seen = set()
        idx = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] == 0:
                continue
            key = ScoresHelper._policy_key(policies[i])
            if key in seen:
                continue
            seen.add(key)
            idx.append(i)
            if len(idx) == k:
                break

        return np.asarray(idx, dtype=int)

    @staticmethod
    def top_k_rough_unique_idx(scores, policies, k=1):
        scores = ScoresHelper._scores_np(scores)
        policies = ScoresHelper._policies_np(policies)
        if scores.size == 0:
            return np.asarray([], dtype=int)

        idx = []
        for i in np.argsort(scores)[::-1]:
            if scores[i] == 0:
                continue
            if any(policies[j].rough_equal(policies[i]) for j in idx):
                continue
            idx.append(i)
            if len(idx) == k:
                break

        return np.asarray(idx, dtype=int)

    @staticmethod
    def top_k_scores(scores, policies, k, rough_unique=False):
        scores = ScoresHelper._scores_np(scores)
        if rough_unique:
            return scores[ScoresHelper.top_k_rough_unique_idx(scores, policies, k)]
        return scores[ScoresHelper.top_k_unique_idx(scores, policies, k)]

    @staticmethod
    def top_k_policies(scores, policies, k, rough_unique=False):
        policies = ScoresHelper._policies_np(policies)
        if rough_unique:
            return policies[ScoresHelper.top_k_rough_unique_idx(scores, policies, k)]
        return policies[ScoresHelper.top_k_unique_idx(scores, policies, k)]

    @staticmethod
    def unique_policies_idx(policies):
        policies = ScoresHelper._policies_np(policies)
        seen = set()
        unique_idx = []
        for candidate_idx, policy in enumerate(policies):
            key = ScoresHelper._policy_key(policy)
            if key in seen:
                continue
            seen.add(key)
            unique_idx.append(candidate_idx)
        return np.asarray(unique_idx, dtype=int)

    @staticmethod
    def unique_scores(scores, policies):
        scores = ScoresHelper._scores_np(scores)
        unique_idx = ScoresHelper.unique_policies_idx(policies)
        return scores[unique_idx]

    @staticmethod
    def unique_policies(policies):
        policies = ScoresHelper._policies_np(policies)
        unique_idx = ScoresHelper.unique_policies_idx(policies)
        return policies[unique_idx]

    @staticmethod
    def best_score(scores):
        scores = ScoresHelper._scores_np(scores)
        if scores.size == 0:
            return None
        return scores.max()

    @staticmethod
    def best_policy(scores, policies):
        scores = ScoresHelper._scores_np(scores)
        if scores.size == 0:
            return None
        best_idx = np.argmax(scores)
        return ScoresHelper._policies_np(policies)[best_idx]


class PolicySearch:
    EVAL_CONFIG = ShotScoreConfig(
        position_k=5.0,
        target_collision_k=1.1,
        target_cushion_k=1.2,
        cue_action_k=1.4,
        cue_route_corner_k=0.3,
        cue_route_collision_k=0.8,
        cue_route_cushion_k=0.005,
        near_miss_k=30.0,
        near_miss_limit=0.15,
        robustness_weight=0.15,
        noise_n_samples=16,
        noise_scales=(0.2, 0.5, 0.08, 0.08),
        noise_max_attempts=128,
    )

    def __init__(self, search_request: SearchRequest):
        self.search_request = search_request
        self._diagnostics = None

    def quick_search(self, k=5, mode="grid", search_config=None, eval_config=None):
        if mode not in ["grid", "stochastic"]:
            raise ValueError(f"Invalid quick_search mode, {mode!r}")
        if mode == "grid":
            scores, policies = GridSearch(self.search_request, eval_config=eval_config).search()
            return self._format_result(
                ScoresHelper.top_k_scores(scores, policies, k),
                ScoresHelper.top_k_policies(scores, policies, k),
                k,
            )

        seed = search_config.seed if search_config else 42
        max_iter = search_config.stochastic_max_iter if search_config else 1200
        sample_mode = search_config.sample_mode if search_config else "random"
        stochastic_search = StochasticSearch(self.search_request, eval_config, seed=seed)
        scores, policies = stochastic_search.search(max_iter, sample_mode)
        self._add_diagnostics_count("candidate_simulations", stochastic_search.candidate_simulations)
        self._add_diagnostics_count("valid_potting_candidates", stochastic_search.valid_potting_candidates)
        return self._format_result(
            ScoresHelper.top_k_scores(scores, policies, k, rough_unique=True),
            ScoresHelper.top_k_policies(scores, policies, k, rough_unique=True),
            k,
        )

    def precise_search(self, k=5, n_rounds=PatternSearch.MAX_SEARCHES, quick_mode="grid",
                       search_config=None, eval_config=None):
        if self._diagnostics is None:
            self._reset_diagnostics()
        if quick_mode not in ["grid", "stochastic"]:
            raise ValueError(f"Invalid quick_search mode, {quick_mode!r}")
        if quick_mode == "grid":
            top_scores, top_policies = self._top_k_from_grid(k, eval_config=eval_config)
        else:
            seed = search_config.seed if search_config else 42
            max_iter = search_config.stochastic_max_iter if search_config else 1200
            sample_mode = search_config.sample_mode if search_config else "random"
            t0 = time.perf_counter()
            top_scores, top_policies = self._top_k_from_stochastic(k, eval_config=eval_config, max_iter=max_iter,
                                                                   sample_mode=sample_mode, seed=seed)
            self._diagnostics["global_stochastic_time"] += time.perf_counter() - t0
        pattern_search = PatternSearch(self.search_request, eval_config)

        refined_scores = []
        refined_policies = []
        t0 = time.perf_counter()
        for score, policy in zip(top_scores, top_policies):
            refined_score, refined_policy = pattern_search.search_n_rounds(score, policy, n_rounds=n_rounds)
            self._add_diagnostics_count("candidate_simulations", pattern_search.candidate_simulations)
            self._add_diagnostics_count("valid_potting_candidates", pattern_search.valid_potting_candidates)
            refined_scores.append(refined_score)
            refined_policies.append(refined_policy)
        self._diagnostics["pattern_search_time"] += time.perf_counter() - t0

        return self._format_result(
            ScoresHelper.top_k_scores(refined_scores, refined_policies, k),
            ScoresHelper.top_k_policies(refined_scores, refined_policies, k),
            k,
        )


    def search(self, search_config: SearchConfig, eval_config=None):
        display_k = search_config.display_k
        search_k = search_config.search_k
        refine_k = search_config.refine_k
        precise = search_config.precise
        n_rounds = search_config.n_rounds
        quick_mode = search_config.quick_mode

        if eval_config is None:
            eval_config = self.EVAL_CONFIG
        self._reset_diagnostics()
        if precise:
            search_results = self.precise_search(k=refine_k, n_rounds=n_rounds, quick_mode=quick_mode,
                                                 search_config=search_config, eval_config=eval_config)
        else:
            t0 = time.perf_counter()
            search_results = self.quick_search(k=search_k, mode=quick_mode,
                                               search_config=search_config, eval_config=eval_config)
            if quick_mode == "stochastic":
                self._diagnostics["global_stochastic_time"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        search_results = self.top_k_rerank_by_robustness(
            search_results,
            display_k,
            eval_config=eval_config,
        )
        self._diagnostics["robustness_rerank_time"] += time.perf_counter() - t0
        self._print_diagnostics()
        return search_results

    def _top_k_from_grid(self, k, eval_config=None):
        scores, policies = GridSearch(self.search_request, eval_config=eval_config).search()
        return ScoresHelper.top_k_scores(scores, policies, k), ScoresHelper.top_k_policies(scores, policies, k)

    def _top_k_from_stochastic(self, k, max_iter=None, sample_mode=None, seed=42, eval_config=None,):
        stochastic_search = StochasticSearch(self.search_request, eval_config=eval_config, seed=seed)
        scores, policies = stochastic_search.search(max_iter, sample_mode)
        self._add_diagnostics_count("candidate_simulations", stochastic_search.candidate_simulations)
        self._add_diagnostics_count("valid_potting_candidates", stochastic_search.valid_potting_candidates)
        return (ScoresHelper.top_k_scores(scores, policies, k, rough_unique=True),
                ScoresHelper.top_k_policies(scores, policies, k, rough_unique=True))

    def top_k_rerank_by_robustness(self, search_results, k=5, eval_config=None):
        if eval_config is None:
            eval_config = self.EVAL_CONFIG
        shot = self.search_request.shot
        final_scores = []
        final_policies = []
        _, policies = self._unpack_search_results(search_results)
        for i, policy in enumerate(policies):
            evaluator = ShotEvaluation(config=eval_config)
            top_shot = shot.copy()
            top_shot.cue.set_state(
                V0=policy.V0,
                phi=policy.phi,
                a=policy.a,
                b=policy.b
            )
            new_score = evaluator.evaluate_shot_final(
                top_shot,
                self.search_request.target_ball_id,
                self.search_request.target_pocket_id,
                self.search_request.target_region,
                config=eval_config
            )
            final_scores.append(new_score)
            final_policies.append(policy)

        return self._format_result(
            ScoresHelper.top_k_scores(final_scores, final_policies, k),
            ScoresHelper.top_k_policies(final_scores, final_policies, k),
            k,
        )

    def _reset_diagnostics(self):
        self._diagnostics = {
            "global_stochastic_time": 0.0,
            "pattern_search_time": 0.0,
            "robustness_rerank_time": 0.0,
            "candidate_simulations": 0,
            "valid_potting_candidates": 0,
        }

    def _add_diagnostics_count(self, key, value):
        if self._diagnostics is not None:
            self._diagnostics[key] += int(value)

    def _print_diagnostics(self):
        if self._diagnostics is None:
            return
        print("PolicySearch diagnostics:")
        print(f"  global stochastic time: {self._diagnostics['global_stochastic_time']:.4f}s")
        print(f"  pattern search time: {self._diagnostics['pattern_search_time']:.4f}s")
        print(f"  robustness rerank time: {self._diagnostics['robustness_rerank_time']:.4f}s")
        print(f"  candidate simulations: {self._diagnostics['candidate_simulations']}")
        print(f"  valid potting candidates: {self._diagnostics['valid_potting_candidates']}")


    @staticmethod
    def _unpack_search_results(search_results):
        scores, policies = search_results
        if isinstance(policies, np.ndarray):
            return ScoresHelper._scores_np(scores), ScoresHelper._policies_np(policies)
        return np.asarray([scores], dtype=float), np.asarray([policies], dtype=object)

    @staticmethod
    def _format_result(scores, policies, k):
        if k == 1:
            if len(scores) == 0:
                return None, None
            return scores[0], policies[0]
        return scores, policies
