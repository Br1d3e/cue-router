from dataclasses import dataclass

import pooltool as pt

from cue_router.core.geometry import PositionTarget


@dataclass
class ExecutorLayout:
    cue_ball_pos: tuple[float, float]
    target_ball_pos: tuple[float, float]
    target_ball_id: str
    other_balls_info: None | dict[str, tuple[float, float]]
    target_pocket_id: str
    target_region: PositionTarget
    route_type: str = "direct"

    def get_balls(self):
        balls = {
            "cue": pt.Ball.create("cue", xy=self.cue_ball_pos),
            self.target_ball_id: pt.Ball.create(self.target_ball_id, xy=self.target_ball_pos),
        }
        if self.other_balls_info:
            for ball_id, pos in self.other_balls_info.items():
                balls[ball_id] = pt.Ball.create(ball_id, xy=pos)
        return balls


@dataclass
class PlannerLayout:
    cue_ball_pos: tuple[float, float]
    balls_info: dict[str, tuple[float, float]]

    def get_balls(self):
        balls = {
            "cue": pt.Ball.create("cue", xy=self.cue_ball_pos),
        }
        for ball_id, pos in self.balls_info.items():
            balls[ball_id] = pt.Ball.create(ball_id, xy=pos)
        return balls

    @classmethod
    def from_shot(cls, shot: pt.System):
        cue_ball_id = shot.cue.cue_ball_id
        cue_ball = shot.balls[cue_ball_id]
        balls_info = {}

        for ball_id, ball in shot.balls.items():
            if ball_id == cue_ball_id:
                continue
            if pt.ruleset.is_ball_pocketed(shot, ball_id):
                continue
            balls_info[ball_id] = tuple(float(v) for v in ball.xyz[:2])

        return cls(
            cue_ball_pos=tuple(float(v) for v in cue_ball.xyz[:2]),
            balls_info=balls_info,
        )
