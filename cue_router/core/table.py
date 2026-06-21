import pooltool as pt
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class ChineseTableSpecs:
    TABLE_LENGTH: ClassVar[float] = 2.540
    TABLE_WIDTH: ClassVar[float] = 1.270

    BALL_RADIUS: ClassVar[float] = 0.028575

    POCKET_POS: ClassVar[list[tuple[float, float]]] = [
        (0 + BALL_RADIUS, 0 + BALL_RADIUS),
        (0 + BALL_RADIUS, TABLE_LENGTH / 2),
        (0 + BALL_RADIUS, TABLE_LENGTH - BALL_RADIUS),
        (TABLE_WIDTH - BALL_RADIUS, 0 + BALL_RADIUS),
        (TABLE_WIDTH - BALL_RADIUS, TABLE_LENGTH / 2),
        (TABLE_WIDTH - BALL_RADIUS, TABLE_LENGTH - BALL_RADIUS),
    ]
    POCKET_ID: ClassVar[list[str]] = ["lb", "lc", "lt", "rb", "rc", "rt"]

    table_specs: ClassVar[pt.objects.PocketTableSpecs] = pt.objects.PocketTableSpecs(
        l=TABLE_LENGTH,
        w=TABLE_WIDTH,
        cushion_width=0.14,
        cushion_height=0.042,

        corner_pocket_width=0.084,
        side_pocket_width=0.087,

        corner_jaw_radius=0.1016,
        corner_pocket_radius=0.07,
        side_jaw_radius=0.0635,
        side_pocket_radius=0.07,

        height=0.825,
    )
    chinese_table: ClassVar[pt.Table] = pt.Table.from_table_specs(table_specs)

    ball_set: ClassVar[pt.objects.ball.sets.BallSet] = pt.objects.get_ballset("pooltool_pocket")

    def __repr__(self):
        return f"ChineseTableSpecs({self.table_specs})"

    @classmethod
    def in_table(cls, xy):
        x, y = xy
        return (0 + cls.BALL_RADIUS < x < cls.TABLE_WIDTH - cls.BALL_RADIUS and
                0 + cls.BALL_RADIUS < y < cls.TABLE_LENGTH - cls.BALL_RADIUS)

    @classmethod
    def in_table_mask(cls, X, Y):
        return (
            (cls.BALL_RADIUS < X)
            & (X < cls.TABLE_WIDTH - cls.BALL_RADIUS)
            & (cls.BALL_RADIUS < Y)
            & (Y < cls.TABLE_LENGTH - cls.BALL_RADIUS)
        )

    @classmethod
    def pocket_id_to_idx(cls, pocket_id):
        pocket_id_to_idx = dict(zip(cls.POCKET_ID, list(range(len(cls.POCKET_ID)))))
        return pocket_id_to_idx[pocket_id]

    @classmethod
    def pocket_idx_to_id(cls, pocket_idx):
        return cls.POCKET_ID[pocket_idx]

    @classmethod
    def pocket_pos_by_id(cls, pocket_id):
        return cls.POCKET_POS[cls.POCKET_ID.index(pocket_id)]

