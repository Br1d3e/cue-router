import pooltool as pt
from pooltool.physics.resolve.stick_ball.squirt import get_squirt_angle
import numpy as np

from cue_router.core.table import ChineseTableSpecs
from cue_router.core.geometry import Geometry
from cue_router.planner.planner_evaluation import ShotOption
from cue_router.executor.shot_evaluation import ShotEvaluation


class LandingSimulator:
    V_WINDOWS = [1.5, 2.5, 3.5]
    A_WINDOWS = [-0.4, 0, 0.4]
    B_WINDOWS = [-0.4, 0, 0.4]

    @staticmethod
    def _build_shot(balls: dict[str, tuple[float, float]]) -> pt.System:
        balls = {
            ball_id: pt.Ball.create(ball_id, xy=pos)
            for ball_id, pos in balls.items()
        }
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=ChineseTableSpecs.chinese_table, balls=balls, cue=cue)
        shot.set_ballset(ChineseTableSpecs.ball_set)
        return shot

    @classmethod
    def simulate(cls, option: ShotOption, balls: dict[str, tuple[float, float]]):
        initial_shot = cls._build_shot(balls)
        target_pocket_id = option.target_pocket_id
        target_ball_id = option.target_ball_id
        base_phi = Geometry.aim_phi(
            pocket=target_pocket_id,
            target_pos=balls[target_ball_id],
            cue_pos=balls["cue"]
        )

        if base_phi is None:
            return np.empty((0, 2), dtype=float)

        evaluator = ShotEvaluation()
        landings = []
        for V0 in cls.V_WINDOWS:
            for a in cls.A_WINDOWS:
                for b in cls.B_WINDOWS:
                    shot = initial_shot.copy()
                    squirt_offset = get_squirt_angle(
                        shot.balls["cue"].params.m,
                        shot.cue.specs.end_mass,
                        a,
                        throttle=1.0
                    )
                    phi = (base_phi - np.degrees(squirt_offset)) % 360
                    shot.cue.set_state(
                        V0=V0,
                        phi=phi,
                        a=a,
                        b=b
                    )
                    pt.simulate(shot, inplace=True)

                    if not evaluator.is_shot_valid(shot, target_ball_id, target_pocket_id, strict_mode=True):
                        continue

                    land_pos = shot.balls["cue"].xyz[:2]
                    landings.append(land_pos)

        return np.asarray(landings, dtype=float).reshape(-1, 2)
