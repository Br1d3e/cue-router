from dataclasses import dataclass
import pooltool as pt

from cue_router.core.geometry import PositionTarget

@dataclass
class SearchRequest:
    shot: pt.System
    target_ball_id: str
    target_pocket_id: str
    target_region: PositionTarget
    route_type: str = "direct"

    def __repr__(self):
        return (
            f"SearchRequest(shot: {self.shot}, "
            f"target_ball_id: {self.target_ball_id}, "
            f"target_pocket_id: {self.target_pocket_id}, "
            f"target_region: {self.target_region}. "
            f"route_type: {self.route_type})"
        )

    def __iter__(self):
        return iter((self.shot, self.target_ball_id, self.target_pocket_id, self.target_region, self.route_type))
