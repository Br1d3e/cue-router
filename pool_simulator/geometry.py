from dataclasses import dataclass, field
import numpy as np

try:
    from .table import ChineseTableSpecs
except ImportError:
    from table import ChineseTableSpecs

class PositionTarget:
    def position_quality(self, xy, position_k):
        ...

@dataclass(order=True)
class Rect(PositionTarget):
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

    def position_quality(self, xy, position_k):
        if self.contains(xy):
            return 1.0
        return 1 / (1 + position_k * self.distance_to(xy))


@dataclass(order=True)
class LineSegment(PositionTarget):
    x1: float
    y1: float
    x2: float
    y2: float

    dx: float = field(init=False, compare=False)
    dy: float = field(init=False, compare=False)
    length_sq: float = field(init=False, compare=False)

    def __post_init__(self):
        self.dx = self.x2 - self.x1
        self.dy = self.y2 - self.y1
        self.length_sq = self.dx * self.dx + self.dy * self.dy
        self.x = (self.x1, self.x2)
        self.y = (self.y1, self.y2)

    def __repr__(self):
        return f"LineSegment(({self.x1}, {self.y1}), ({self.x2}, {self.y2}))"

    def distance_to(self, xy):
        x, y = xy
        if self.length_sq == 0:
            return ((x - self.x1) ** 2 + (y - self.y1) ** 2) ** 0.5

        t = ((x - self.x1) * self.dx + (y - self.y1) * self.dy) / self.length_sq
        t = min(max(t, 0.0), 1.0)
        closest_x = self.x1 + t * self.dx
        closest_y = self.y1 + t * self.dy
        return ((x - closest_x) ** 2 + (y - closest_y) ** 2) ** 0.5

    def contains(self, xy):
        return self.distance_to(xy) <= 0.05

    def position_quality(self, xy, position_k):
        distance = self.distance_to(xy)
        if self.contains(xy):
            return 1.0
        return 1 / (1 + position_k * distance)


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
    def aim_phi(pocket, target_pos, cue_pos):
        if isinstance(pocket, str):
            pocket_pos = np.array(ChineseTableSpecs.pocket_pos_by_id(pocket))
        else:
            pocket_pos = np.array(pocket[:2])
        pocket_pos = np.array(pocket_pos[:2])
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

    @staticmethod
    def _mirror_point_across_rail(p, rail, W, L, R=0.0):
        x, y = p

        if rail == "left":
            c = R
            return np.asarray((2 * c - x, y), dtype=float)

        if rail == "right":
            c = W - R
            return np.asarray((2 * c - x, y), dtype=float)

        if rail == "bottom":
            c = R
            return np.asarray((x, 2 * c - y), dtype=float)

        if rail == "top":
            c = L - R
            return np.asarray((x, 2 * c - y), dtype=float)

        raise ValueError(f"unknown rail: {rail}")

    @staticmethod
    def aim_phi_bank(pocket, target_pos, cue_pos):
        if isinstance(pocket, str):
            pocket_pos = np.array(ChineseTableSpecs.pocket_pos_by_id(pocket))
        else:
            pocket_pos = np.asarray(pocket[:2])
        table_length = ChineseTableSpecs.TABLE_LENGTH
        table_width = ChineseTableSpecs.TABLE_WIDTH
        R = ChineseTableSpecs.BALL_RADIUS

        rails = ["left", "right", "bottom", "top"]
        mirrored_pocket_pos = None
        for rail in rails:
            pos = Geometry._mirror_point_across_rail(pocket_pos, rail, W=table_width, L=table_length, R=R)
            if Geometry.aim_phi(pos, target_pos, cue_pos) is None:
                continue
            mirrored_pocket_pos = pos

        if mirrored_pocket_pos is None:
            return None
        return Geometry.aim_phi(mirrored_pocket_pos, target_pos, cue_pos)


