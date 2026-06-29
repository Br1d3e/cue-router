from dataclasses import dataclass

import numpy as np
import pooltool as pt

from cue_router.core.table import ChineseTableSpecs
from cue_router.executor.search import PolicySearch, SearchConfig
from cue_router.executor.action import ShotAction
from cue_router.planner.planner_evaluation import ShotOption
from cue_router.planner.planner_search import PlannerConfig, PlannerContext, PlannerSearch
from cue_router.planner.position_value_map import ContinuationOption
from cue_router.ui.layout import PlannerLayout


@dataclass
class TwoPlyConfig:
    planner_top_k: int | None = None
    min_execution_score: float = 0.05
    plan_weight: float = 1.2
    execution_weight: float = 0.8
    two_ply_weight: float = 0.30
    fallback_to_all_next_shots: bool = True
    breakout_two_ply_value: float = 1.0


@dataclass
class TwoPlyPlan:
    option: ShotOption
    policy: ShotAction
    simulated_shot: pt.System
    planner_score: float
    execution_score: float
    two_ply_value: float
    combined_score: float
    next_context: PlannerContext
    two_ply_candidates: tuple[ContinuationOption, ...] = ()
    two_ply_excluded: bool = False


class TwoPlySearch:
    QUICK_SEARCH_CONFIG = SearchConfig(
        display_k=3,
        search_k=8,
        refine_k=5,
        mode="elite",
        quick_mode="stochastic",
        sample_mode="LHS",
        n_rounds=1,
        stochastic_max_iter=300,
        elite_max_iter=200,
        elite_frac=0.3,
    )

    def __init__(
            self,
            planner_config: PlannerConfig | None = None,
            two_ply_config: TwoPlyConfig | None = None,
            search_config: SearchConfig | None = None,
            eval_config=None,
    ):
        self.planner_config = planner_config or PlannerConfig()
        self.two_ply_config = two_ply_config or TwoPlyConfig()
        self.search_config = search_config or self.QUICK_SEARCH_CONFIG
        self.eval_config = eval_config
        self.planner = PlannerSearch(self.planner_config)

    def search(self, shot, layout, legal_targets, context=None):
        planner_top_k = self.two_ply_config.planner_top_k or self.planner_config.display_k
        options = self.planner.search(
            layout,
            legal_targets,
            display_k=planner_top_k,
            context=context,
        )

        results = []
        for option in options:
            result = self._evaluate_option(shot, option, legal_targets)
            if result is not None:
                results.append(result)

        return sorted(results, key=lambda result: result.combined_score, reverse=True)

    def _evaluate_option(self, shot, option, legal_targets):
        search_request = option.to_search_request(shot.copy())
        search_config = self.search_config or SearchConfig(display_k=1)
        searcher = PolicySearch(search_request)
        scores, policies = searcher.search(search_config=search_config, eval_config=self.eval_config)
        execution_score, policy = self._best_result(scores, policies)
        if policy is None or execution_score <= self.two_ply_config.min_execution_score:
            return None

        simulated_shot = search_request.shot.copy()
        simulated_shot.cue.set_state(V0=policy.V0, phi=policy.phi, a=policy.a, b=policy.b)
        pt.simulate(simulated_shot, inplace=True)

        two_ply_value, two_ply_candidates, two_ply_excluded = self._two_ply_value(
            simulated_shot,
            option,
            legal_targets,
        )
        combined_score = self._combined_score(option.score, execution_score, two_ply_value)

        return TwoPlyPlan(
            option=option,
            policy=policy,
            simulated_shot=simulated_shot,
            planner_score=float(option.score),
            execution_score=float(execution_score),
            two_ply_value=float(two_ply_value),
            combined_score=float(combined_score),
            next_context=PlannerContext.from_option(option),
            two_ply_candidates=two_ply_candidates,
            two_ply_excluded=two_ply_excluded,
        )

    @staticmethod
    def _best_result(scores, policies):
        scores = np.atleast_1d(np.asarray([] if scores is None else scores, dtype=float))
        policies = np.atleast_1d(np.asarray([] if policies is None else policies, dtype=object))
        if scores.size == 0 or policies.size == 0:
            return 0.0, None
        best_idx = int(np.argmax(scores))
        return float(scores[best_idx]), policies[best_idx]

    def _two_ply_value(self, simulated_shot, option, legal_targets):
        """
        :param simulated_shot:
        :param option:
        :param legal_targets:
        :return: two_ply_value, two_ply_candidates, two_ply_excluded
        """
        if option.option_family != "normal_position":
            return self.two_ply_config.breakout_two_ply_value, (), True

        layout = PlannerLayout.from_shot(simulated_shot)
        legal_after_current = self._legal_targets_after_current(layout, legal_targets)
        if len(legal_after_current) == 0:
            return 1.0, (), False

        intended_next_shots = tuple(option.score_breakdown.get("intended_next_shots", ()) or ())
        candidates = tuple(
            candidate
            for candidate in intended_next_shots
            if candidate.target_ball_id in legal_after_current
            and candidate.target_ball_id in layout.balls_info
        )
        if len(candidates) == 0 and self.two_ply_config.fallback_to_all_next_shots:
            candidates = tuple(
                ContinuationOption(ball_id, pocket_id, route_type="direct")
                for ball_id in legal_after_current
                if ball_id in layout.balls_info
                for pocket_id in ChineseTableSpecs.POCKET_ID
            )
        if len(candidates) == 0:
            return 0.0, (), False

        values = [
            self._next_shot_future_position_value(layout, legal_after_current, candidate)
            for candidate in candidates
        ]
        return float(max(values, default=0.0)), candidates, False

    def _next_shot_future_position_value(self, layout, legal_targets, candidate):
        balls = {
            str(ball_id): np.asarray(pos, dtype=float)
            for ball_id, pos in layout.balls_info.items()
        }
        if candidate.target_ball_id not in balls:
            return 0.0

        future_balls = {
            ball_id: xy
            for ball_id, xy in balls.items()
            if ball_id != candidate.target_ball_id
        }
        future_legal_targets = self.planner._future_legal_targets(
            balls,
            legal_targets,
            candidate.target_ball_id,
        )
        value_map = self.planner._build_position_value_map(
            future_balls,
            future_legal_targets,
            balls[candidate.target_ball_id],
        )
        _, future_value, _ = value_map.select_best_target_heatmap(
            threshold=self.planner_config.target_region_threshold,
            top_k=self.planner_config.target_heatmap_k,
            top_fraction=self.planner_config.top_heatmap_percent,
        )
        return float(future_value)

    @staticmethod
    def _legal_targets_after_current(layout, legal_targets):
        remaining = [ball_id for ball_id in legal_targets if ball_id in layout.balls_info]
        if len(remaining) == 0 and "8" in layout.balls_info:
            return ["8"]
        return remaining

    def _combined_score(self, planner_score, execution_score, two_ply_value):
        eps = 1e-9
        plan = max(eps, float(np.clip(planner_score, 0.0, 1.0)))
        execution = max(eps, float(np.clip(execution_score, 0.0, 1.0)))
        future = max(eps, float(np.clip(two_ply_value, 0.0, 1.0)))
        return (
            plan ** self.two_ply_config.plan_weight
            * execution ** self.two_ply_config.execution_weight
            * future ** self.two_ply_config.two_ply_weight
        )
