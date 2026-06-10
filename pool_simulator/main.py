import pooltool as pt
import numpy as np

from table import ChineseTableSpecs
from search import PolicySearch, SearchRequest
from geometry import Rect


chinese_table = ChineseTableSpecs.chinese_table

x1, y1 = 0.2, 0.2
x2, y2 = 0.3, 0.3
target_rect = Rect(x1, y1, x2, y2)

balls = {
    "cue": pt.Ball.create("cue", xy=(0.5, 1.15)),
    "1": pt.Ball.create("1", xy=(1, 1.27)),
    # "8": pt.Ball.create("8", xy=(x1, y1)),
    # "9": pt.Ball.create("9", xy=(x2, y2)),
}
cue = pt.Cue(cue_ball_id="cue")

shot = pt.System(table=chinese_table, balls=balls, cue=cue)
shot.set_ballset(ChineseTableSpecs.ball_set)

search_request = SearchRequest(shot=shot, target_ball_id="1", target_pocket_id='rc', target_rect=target_rect)
searcher = PolicySearch(search_request)

K = 5
import time
b4 = time.time()
top_k_scores, top_k_policies = searcher.search(k=K, n_rounds=2)
print(f"search time: {time.time() - b4}")

for i in range(len(top_k_policies)):
    print(f"===Route #{i+1}===")

    top_score = top_k_scores[i]
    top_policy = top_k_policies[i]

    print("score:", round(top_score, 3))
    print(top_policy)

    top_shot = shot.copy()
    top_shot.cue.set_state(V0=top_policy.V0, phi=top_policy.phi, a=top_policy.a, b=top_policy.b)

    pt.simulate(top_shot, inplace=True)
    pt.show(top_shot)