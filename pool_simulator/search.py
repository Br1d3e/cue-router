from dataclasses import dataclass

import numpy as np
import pooltool as pt

try:
    from .geometry import Rect
    from .search_method.grid_search import GridSearch
    from .search_method.pattern_search import PatternSearch
except ImportError:
    from geometry import Rect
    from search_method.grid_search import GridSearch
    from search_method.pattern_search import PatternSearch


@dataclass
class SearchRequest:
    shot: pt.System
    target_ball_id: str
    target_pocket_id: str
    target_rect: Rect

    def __repr__(self):
        return (f"SearchRequest(shot: {self.shot}, "
                f"target_ball_id: {self.target_ball_id}, "
                f"target_pocket_id: {self.target_pocket_id}, "
                f"target_rect: {self.target_rect})")

    def __iter__(self):
        return iter((self.shot, self.target_ball_id, self.target_pocket_id, self.target_rect))


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
            key = ScoresHelper._policy_key(policies[i])
            if key in seen:
                continue
            seen.add(key)
            idx.append(i)
            if len(idx) == k:
                break

        return np.asarray(idx, dtype=int)

    @staticmethod
    def top_k_scores(scores, policies, k):
        scores = ScoresHelper._scores_np(scores)
        return scores[ScoresHelper.top_k_unique_idx(scores, policies, k)]

    @staticmethod
    def top_k_policies(scores, policies, k):
        policies = ScoresHelper._policies_np(policies)
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
    def __init__(self, search_request: SearchRequest):
        self.search_request = search_request

    def quick_search(self, k=5):
        scores, policies = GridSearch(self.search_request).search()
        return self._format_result(
            ScoresHelper.top_k_scores(scores, policies, k),
            ScoresHelper.top_k_policies(scores, policies, k),
            k,
        )

    def precise_search(self, k=5, n_rounds=PatternSearch.MAX_SEARCHES):
        top_scores, top_policies = self._top_k_from_grid(k)
        pattern_search = PatternSearch(self.search_request)

        refined_scores = []
        refined_policies = []
        for score, policy in zip(top_scores, top_policies):
            refined_score, refined_policy = pattern_search.search_n_rounds(score, policy, n_rounds=n_rounds)
            refined_scores.append(refined_score)
            refined_policies.append(refined_policy)

        return self._format_result(
            ScoresHelper.top_k_scores(refined_scores, refined_policies, k),
            ScoresHelper.top_k_policies(refined_scores, refined_policies, k),
            k,
        )

    def search(self, k=5, precise=True, n_rounds=PatternSearch.MAX_SEARCHES):
        if precise:
            return self.precise_search(k=k, n_rounds=n_rounds)
        return self.quick_search(k=k)

    def _top_k_from_grid(self, k):
        scores, policies = GridSearch(self.search_request).search()
        return ScoresHelper.top_k_scores(scores, policies, k), ScoresHelper.top_k_policies(scores, policies, k)


    @staticmethod
    def _format_result(scores, policies, k):
        if k == 1:
            if len(scores) == 0:
                return None, None
            return scores[0], policies[0]
        return scores, policies
