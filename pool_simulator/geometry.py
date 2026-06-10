from dataclasses import dataclass, field
import numpy as np

try:
    from .table import ChineseTableSpecs
except ImportError:
    from table import ChineseTableSpecs


@dataclass(order=True)
class Rect:
    x1: float
    y1: float
    x2: float
    y2: float

    xmin: float = field(init=False, compare=False)
    xmax: float = field(init=False, compare=False)
    ymin: float = field(init=False, compare=False)
    ymax: float = field(init=False, compare=False)

    def __post_init__(self):
        self.xmin = min(self.x1, self.x2)
        self.ymin = min(self.y1, self.y2)
        self.xmax = max(self.x1, self.x2)
        self.ymax = max(self.y1, self.y2)

    def __repr__(self):
        return f"Rect(({self.xmin}, {self.ymin}), ({self.xmax}, {self.ymax}))"

    def __eq__(self, other):
        if not isinstance(other, Rect):
            if isinstance(other, dict) and ["xmin", "xmax", "ymin", "ymax"] in other.keys():
                return (
                        (other.get("xmin"), other.get("xmax"), other.get("ymin"), other.get("ymax"))
                        == (self.xmin, self.xmax, self.ymin, self.ymax)
                )
            return False
        return (self.xmin, self.xmax, self.ymin, self.ymax) == (other.xmin, other.xmax, other.ymin, other.ymax)

    def contains(self, xy) -> bool:
        x, y = xy
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def distance_to(self, xy) -> float:
        x, y = xy
        dx = max(self.xmin - x, 0, x - self.xmax)
        dy = max(self.ymin - y, 0, y - self.ymax)
        return (dx * dx + dy * dy) ** 0.5

    def distance_to_center(self, xy) -> float:
        x, y = xy

        x_mid = (self.xmin + self.xmax) / 2
        y_mid = (self.ymin + self.ymax) / 2
        dx = abs(x_mid - x)
        dy = abs(y_mid - y)

        return (dx * dx + dy * dy) ** 0.5


class Geometry:
    BALL_RADIUS = ChineseTableSpecs.BALL_RADIUS

    @staticmethod
    def is_ghost_reachable(cue_pos, obj_pos, ghost_pos, R=BALL_RADIUS, eps=1e-5):
        cue_pos = np.array(cue_pos, dtype=float)
        obj_pos = np.array(obj_pos, dtype=float)
        ghost_pos = np.array(ghost_pos, dtype=float)

        v = ghost_pos - cue_pos
        L2 = np.dot(v, v)
        if L2 == 0:
            return False

        # Projection of object center onto cue->ghost segment
        t = np.dot(obj_pos - cue_pos, v) / L2

        if t <= 0 or t >= 1:
            return True

        closest = cue_pos + t * v
        d = np.linalg.norm(obj_pos - closest)

        # If line passes inside the 2R collision circle before ghost,
        # cue ball will collide before reaching intended ghost point.
        return d >= 2 * R - eps

    @staticmethod
    def aim_phi(pocket_id: str, target_pos, cue_pos):
        pocket_pos = np.array(ChineseTableSpecs.pocket_pos_by_id(pocket_id))
        target_pos = np.array(target_pos[:2])
        cue_pos = np.array(cue_pos[:2])

        vec_target = pocket_pos - target_pos
        vec_target_mag = np.linalg.norm(vec_target)

        extrapolate_vec = -vec_target / vec_target_mag * 2 * Geometry.BALL_RADIUS
        aim_pos = target_pos + extrapolate_vec

        reachable = Geometry.is_ghost_reachable(cue_pos, target_pos, aim_pos)
        if not reachable:
            return None

        vec_aim = aim_pos - cue_pos
        phi = np.degrees(np.atan2(vec_aim[1], vec_aim[0])) % 360

        return float(phi)
