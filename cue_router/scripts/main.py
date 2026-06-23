import pooltool as pt
import time

from cue_router.core.table import ChineseTableSpecs
from cue_router.executor.search import PolicySearch, SearchConfig
from cue_router.core.geometry import Rect
from cue_router.ui.executor_selector import select_executor_layout
from cue_router.ui.planner_selector import select_planner_layout
from cue_router.ui.layout import ExecutorLayout, PlannerLayout
from cue_router.ui.table_view import show_heatmap_target
from cue_router.executor.shot_evaluation import ShotEvaluation
from cue_router.core.request import BreakoutTarget, SearchRequest

from cue_router.planner.planner_search import PlannerSearch, PlannerConfig

chinese_table = ChineseTableSpecs.chinese_table
MODE = "breakout"
VISUAL_SELECTOR = "manual"

PLANNER_CONFIG = PlannerConfig(
    display_k=5
)
LEGAL_TARGET_BALLS = [str(i) for i in range(1, 8)]
TOP_K = 5

SEARCH_CONFIG = SearchConfig(
    display_k=TOP_K,
    search_k=10,
    refine_k=5,
    mode="precise",
    quick_mode="stochastic",
    n_rounds=2,
    stochastic_max_iter=600,
    elite_max_iter=500,
    elite_frac=0.3,
    sample_mode="LHS",
    seed=42,
)


def manual_layout():
    return ExecutorLayout(
        cue_ball_pos=(0.6226075122482306, 1.3573298856831795),
        target_ball_pos=(0.9010506260206856, 0.6190337506804582),
        target_ball_id="1",
        other_balls_info={'2': (0.43697876973326055, 0.6295808383233537), '3': (0.512917800762112, 1.9184349482852479)},
        target_pocket_id="rb",
        target_region=Rect(0.966442569406641, 0.9924006532389773, 1.1731654872074029, 1.258187261839957)
    )


def manual_layout_2():
    return PlannerLayout(
        cue_ball_pos=(0.6247169297768098, 1.2434213391399027),
        balls_info={'1': (0.9622237343494826, 0.3933260751224834),
                    '2': (0.6141698421339137, 0.08535111594991918),
                    '3': (0.32728905824714194, 0.783568317909636),
                    '4': (0.8926129559063689, 1.1801388132825263),
                    '5': (0.8082362547632007, 1.8424959172563966),
                    '7': (0.4285410996189438, 1.960623298856832),
                    '8': (0.25345944474686966, 1.536630375612412)}
    )


def manual_layout_3():
    return ExecutorLayout(
        cue_ball_pos=(0.6901088731627651, 2.4500081654872075),
        target_ball_pos=(1.0634757757212845, 2.414148067501361),
        target_ball_id="1",
        other_balls_info={"3": (1.2132444202504078, 1.7707757212847035)},
        target_pocket_id="rt",
        target_region=None,
        route_type="direct",
    )


def build_search_request(layout: ExecutorLayout):
    cue = pt.Cue(cue_ball_id="cue")
    shot = pt.System(table=chinese_table, balls=layout.get_balls(), cue=cue)
    shot.set_ballset(ChineseTableSpecs.ball_set)
    return SearchRequest(
        shot=shot,
        target_ball_id=layout.target_ball_id,
        target_pocket_id=layout.target_pocket_id,
        position_target=layout.target_region,
        route_type=layout.route_type,
    )


def build_breakout_search_request(layout: ExecutorLayout, breakout_ball_id: str):
    cue = pt.Cue(cue_ball_id="cue")
    shot = pt.System(table=chinese_table, balls=layout.get_balls(), cue=cue)
    shot.set_ballset(ChineseTableSpecs.ball_set)
    return SearchRequest(
        shot=shot,
        target_ball_id=layout.target_ball_id,
        target_pocket_id=layout.target_pocket_id,
        route_type=layout.route_type,
        objective="breakout",
        breakout_target=BreakoutTarget(breakout_ball_id),
    )


def route_search(layout):
    search_request = build_search_request(layout)
    searcher = PolicySearch(search_request)

    import time
    b4 = time.time()
    top_k_scores, top_k_policies = searcher.search(
        search_config=SEARCH_CONFIG,
        eval_config=None,
    )
    if TOP_K == 1:
        top_k_scores = [top_k_scores]
        top_k_policies = [top_k_policies]
    print(f"search time: {time.time() - b4}")

    for i in range(len(top_k_policies)):
        print(f"===Route #{i + 1}===")

        top_score = top_k_scores[i]
        top_policy = top_k_policies[i]

        print(top_policy)

        top_shot = search_request.shot.copy()
        top_shot.cue.set_state(V0=top_policy.V0, phi=top_policy.phi, a=top_policy.a, b=top_policy.b)

        shot_evaluator = ShotEvaluation(config=searcher.EVAL_CONFIG)
        score = shot_evaluator.evaluate_shot_final(
            top_shot,
            search_request.target_ball_id,
            search_request.target_pocket_id,
            search_request.position_target,
            route_type=search_request.route_type
        )
        # if shot_evaluator.position_quality <= 0.4:
        #     print(f"Skipping route: position quality={shot_evaluator.position_quality} is too low")
        #     continue
        shot_evaluator.print_score_breakdown()

        pt.simulate(top_shot, inplace=True)
        pt.show(top_shot)


def breakout_search(layout):
    search_request = build_breakout_search_request(layout, breakout_ball_id="3")
    searcher = PolicySearch(search_request)

    b4 = time.time()
    top_k_scores, top_k_policies = searcher.search(
        search_config=SEARCH_CONFIG,
        eval_config=None,
    )
    if TOP_K == 1:
        top_k_scores = [top_k_scores]
        top_k_policies = [top_k_policies]
    print(f"breakout search time: {time.time() - b4}")

    for i in range(len(top_k_policies)):
        print(f"===Breakout Route #{i + 1}===")

        top_score = top_k_scores[i]
        top_policy = top_k_policies[i]

        print("score:", top_score)
        print(top_policy)

        top_shot = search_request.shot.copy()
        top_shot.cue.set_state(V0=top_policy.V0, phi=top_policy.phi, a=top_policy.a, b=top_policy.b)

        shot_evaluator = ShotEvaluation(config=searcher.EVAL_CONFIG)
        shot_evaluator.evaluate_request_final(top_shot, search_request)
        shot_evaluator.print_score_breakdown()

        pt.simulate(top_shot, inplace=True)
        pt.show(top_shot)


def planner_1_ball(shot):
    planner_layout = PlannerLayout.from_shot(shot)
    b4 = time.time()
    planner = PlannerSearch(PLANNER_CONFIG)
    plans = planner.search(planner_layout, LEGAL_TARGET_BALLS)
    print(f"plan search time used: {time.time() - b4:.4f}")

    for i, plan in enumerate(plans):
        print(f"==== Plan #{i + 1} ====")
        print(plan.target_ball_id, plan.target_pocket_id)
        print("score:", plan.score)
        print(plan.score_breakdown)

        show_heatmap_target(plan.position_target, layout=planner_layout)

        search_request = plan.to_search_request(shot.copy())
        searcher = PolicySearch(search_request)

        top_k_scores, top_k_policies = searcher.search(
            search_config=SEARCH_CONFIG,
            eval_config=None,
        )
        if TOP_K == 1:
            top_k_scores = [top_k_scores]
            top_k_policies = [top_k_policies]

        for i in range(len(top_k_policies)):
            print(f"===Route #{i + 1}===")

            top_score = top_k_scores[i]
            top_policy = top_k_policies[i]

            print(top_policy)

            top_shot = search_request.shot.copy()
            top_shot.cue.set_state(V0=top_policy.V0, phi=top_policy.phi, a=top_policy.a, b=top_policy.b)

            shot_evaluator = ShotEvaluation(config=searcher.EVAL_CONFIG)
            score = shot_evaluator.evaluate_shot_final(
                top_shot,
                search_request.target_ball_id,
                search_request.target_pocket_id,
                search_request.position_target,
                route_type=search_request.route_type
            )
            shot_evaluator.print_score_breakdown()

            pt.simulate(top_shot, inplace=True)
            pt.show(top_shot)


# Executes best plan + best route policy every time until clearance
def planner_full(shot, show_heatmap=False):
    shot = shot.copy()
    planner = PlannerSearch(PLANNER_CONFIG)
    legal_balls = LEGAL_TARGET_BALLS
    planner_layout = PlannerLayout.from_shot(shot)
    while len(legal_balls) > 0:
        b4 = time.time()
        plans = planner.search(planner_layout, legal_balls, display_k=5)
        print(f"plan search time used, {time.time() - b4:.4f}")
        if len(plans) == 0:
            break
        plan_idx = 0
        plan = plans[plan_idx]

        target_ball = plan.target_ball_id
        target_pocket = plan.target_pocket_id

        print(target_ball, target_pocket)
        print("plan score:", plan.score)
        print(plan.score_breakdown)

        if show_heatmap:
            show_heatmap_target(plan.position_target, layout=planner_layout, target_region_thres=0.0)

        searcher = PolicySearch(plan.to_search_request(shot.copy()))
        b4 = time.time()
        top_k_scores, top_k_policies = searcher.search(
            search_config=SEARCH_CONFIG,
            eval_config=None,
        )
        print(f"route search time used, {time.time() - b4:.4f}")

        # fallback
        while len(top_k_scores) <= 0 or top_k_scores[0] <= 0.2:
            plan_idx += 1
            if plan_idx >= len(plans):
                print(f"unable to find an executable plan")
                return
            plan = plans[plan_idx]
            target_ball = plan.target_ball_id
            target_pocket = plan.target_pocket_id
            print(target_ball, target_pocket)
            print("plan score:", plan.score)
            print(plan.score_breakdown)
            if show_heatmap:
                show_heatmap_target(plan.position_target, layout=planner_layout, target_region_thres=0.0)

            searcher = PolicySearch(plan.to_search_request(shot.copy()))
            top_k_scores, top_k_policies = searcher.search(
                search_config=SEARCH_CONFIG,
                eval_config=None,
            )

        best_score, best_policy = top_k_scores[0], top_k_policies[0]
        if best_score is None:
            print("shot impossible")
            return

        print("policy score:", best_score)
        print(best_policy)

        best_shot = shot.copy()
        best_shot.cue.set_state(V0=best_policy.V0, phi=best_policy.phi, a=best_policy.a, b=best_policy.b)
        pt.simulate(best_shot, inplace=True)
        pt.show(best_shot)

        shot = best_shot
        planner_layout = PlannerLayout.from_shot(shot)
        legal_balls = [ball_id for ball_id in legal_balls if ball_id in planner_layout.balls_info]
        if len(legal_balls) == 0 and "8" in planner_layout.balls_info:
            legal_balls.append("8")


def main():
    if VISUAL_SELECTOR == "executor":
        layout = select_executor_layout()
    elif VISUAL_SELECTOR == "planner":
        layout = select_planner_layout()
    elif MODE == "breakout":
        layout = manual_layout_3()
    else:
        layout = manual_layout() if MODE == "executor" else manual_layout_2()
    print("Pool layout: \n", layout)
    if layout is None:
        raise RuntimeError("No pool layout selected.")

    cue = pt.Cue(cue_ball_id="cue")
    shot = pt.System(table=chinese_table, balls=layout.get_balls(), cue=cue)
    shot.set_ballset(ChineseTableSpecs.ball_set)

    if MODE == "executor":
        route_search(layout)
    elif MODE == "breakout":
        breakout_search(layout)
    elif MODE == "planner":
        # planner_1_ball(layout, shot)
        planner_full(shot, True)


if __name__ == "__main__":
    main()
