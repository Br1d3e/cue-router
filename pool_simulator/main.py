import pooltool as pt
import numpy as np

from table import ChineseTableSpecs
from search import PolicySearch, SearchRequest
from geometry import Rect
from visual import PoolLayout, select_pool_layout


chinese_table = ChineseTableSpecs.chinese_table
USE_VISUAL_SELECTOR = True
USE_QUICK_SEARCH = True
TOP_K = 5
SEARCH_ROUNDS = 2


def manual_layout():
    x1, y1 = 0.2, 0.2
    x2, y2 = 0.3, 0.3
    return PoolLayout(
        cue_ball_pos=(0.5, 1.15),
        target_ball_pos=(1, 1.27),
        target_ball_id="1",
        other_balls_info=None,
        target_pocket_id="rc",
        target_rect=Rect(x1, y1, x2, y2),
    )


def build_search_request(layout: PoolLayout):
    cue = pt.Cue(cue_ball_id="cue")
    shot = pt.System(table=chinese_table, balls=layout.get_balls(), cue=cue)
    shot.set_ballset(ChineseTableSpecs.ball_set)
    return SearchRequest(
        shot=shot,
        target_ball_id=layout.target_ball_id,
        target_pocket_id=layout.target_pocket_id,
        target_rect=layout.target_rect,
    )


def main():
    layout = select_pool_layout() if USE_VISUAL_SELECTOR else manual_layout()
    if layout is None:
        raise RuntimeError("No pool layout selected.")

    search_request = build_search_request(layout)
    searcher = PolicySearch(search_request)

    import time
    b4 = time.time()
    top_k_scores, top_k_policies = searcher.search(
        k=TOP_K,
        precise=not USE_QUICK_SEARCH,
        n_rounds=SEARCH_ROUNDS,
    )
    if TOP_K == 1:
        top_k_scores = [top_k_scores]
        top_k_policies = [top_k_policies]
    print(f"search time: {time.time() - b4}")

    for i in range(len(top_k_policies)):
        print(f"===Route #{i + 1}===")

        top_score = top_k_scores[i]
        top_policy = top_k_policies[i]

        print("score:", round(top_score, 3))
        print(top_policy)

        top_shot = search_request.shot.copy()
        top_shot.cue.set_state(V0=top_policy.V0, phi=top_policy.phi, a=top_policy.a, b=top_policy.b)

        pt.simulate(top_shot, inplace=True)
        pt.show(top_shot)


if __name__ == "__main__":
    main()
