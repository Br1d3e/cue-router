from dataclasses import dataclass
from enum import Enum
from typing import Literal
import pooltool as pt

from cue_router.core.geometry import PositionTarget


class RouteType(Enum):
    DIRECT = "direct"
    BANK = "bank"


class ShotObjective(Enum):
    POSITION = "position"
    BREAKOUT = "breakout"


@dataclass
class BreakoutTarget:
    ball_id: str
    max_chain_collisions: int = 3
    path_region: PositionTarget | None = None
    path_region_weight: float = 0.25
    preferred_side: Literal["left", "right", "auto"] | None = None


@dataclass
class SearchRequest:
    shot: pt.System
    target_ball_id: str
    target_pocket_id: str

    route_type: RouteType = RouteType.DIRECT
    objective: ShotObjective = ShotObjective.POSITION

    position_target: PositionTarget | None = None
    breakout_target: BreakoutTarget | None = None

    def __repr__(self):
        return (
            f"SearchRequest(shot: {self.shot}, "
            f"target_ball_id: {self.target_ball_id}, "
            f"target_pocket_id: {self.target_pocket_id}, "
            f"target_region: {self.position_target}, "
            f"breakout_target: {self.breakout_target}, "
            f"route_type: {self.route_type}, "
            f"objective: {self.objective})"
        )

    def __iter__(self):
        return iter((self.shot, self.target_ball_id, self.target_pocket_id, self.position_target, self.route_type))
