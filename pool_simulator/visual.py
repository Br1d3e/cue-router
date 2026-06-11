from dataclasses import dataclass
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np
import pooltool as pt

try:
    from .geometry import Rect
    from .table import ChineseTableSpecs
except ImportError:
    from geometry import Rect
    from table import ChineseTableSpecs


@dataclass
class PoolLayout:
    cue_ball_pos: tuple[float, float]
    target_ball_pos: tuple[float, float]
    target_ball_id: str
    other_balls_info: None | dict[str, tuple[float, float]]
    target_pocket_id: str
    target_rect: Rect

    def get_balls(self):
        balls = {
            "cue": pt.Ball.create("cue", xy=self.cue_ball_pos),
            self.target_ball_id: pt.Ball.create(self.target_ball_id, xy=self.target_ball_pos),
        }
        if self.other_balls_info:
            for ball_id, pos in self.other_balls_info.items():
                balls[ball_id] = pt.Ball.create(ball_id, xy=pos)
        return balls


ZOOM_FACTOR = 4
TABLE_ORIGIN = np.asarray((0.5, 0.5), dtype=float)
HELP_AREA_HEIGHT = 1.6
HELP_WRAP_WIDTH = 82
BALL_KEY_MAP = {
    **{str(i): str(i) for i in range(1, 10)},
    "q": "10",
    "w": "11",
    "e": "12",
    "r": "13",
    "t": "14",
    "y": "15",
}
BALL_COLORS = {
    "cue": ("white", "white"),
    "1": ("yellow", "goldenrod"),
    "2": ("blue", "navy"),
    "3": ("red", "darkred"),
    "4": ("green", "darkgreen"),
    "5": ("orange", "darkorange"),
    "6": ("pink", "deeppink"),
    "7": ("purple", "indigo"),
    "8": ("black", "black"),
    "9": ("white", "goldenrod"),
    "10": ("white", "navy"),
    "11": ("white", "darkred"),
    "12": ("white", "darkgreen"),
    "13": ("white", "darkorange"),
    "14": ("white", "deeppink"),
    "15": ("white", "indigo"),
}
MODE_SEQUENCE = ["cue_ball", "target_ball", "pocket", "other_balls", "target_region"]


def disable_conflicting_matplotlib_keys():
    for keymap_name in ("keymap.quit", "keymap.quit_all"):
        plt.rcParams[keymap_name] = [
            key for key in plt.rcParams[keymap_name]
            if key.lower() != "q"
        ]


def table_plot_size():
    return (
        ChineseTableSpecs.TABLE_LENGTH * ZOOM_FACTOR,
        ChineseTableSpecs.TABLE_WIDTH * ZOOM_FACTOR,
    )


def table_to_plot(pos):
    x, y = pos
    return np.asarray((y * ZOOM_FACTOR, x * ZOOM_FACTOR), dtype=float) + TABLE_ORIGIN


def plot_to_table(pos):
    x_plot, y_plot = pos
    return (
        float((y_plot - TABLE_ORIGIN[1]) / ZOOM_FACTOR),
        float((x_plot - TABLE_ORIGIN[0]) / ZOOM_FACTOR),
    )


def in_table_plot(pos):
    x, y = plot_to_table(pos)
    return ChineseTableSpecs.in_table((x, y))


def create_ball(ax, pos, color_set, radius=None):
    if radius is None:
        radius = ChineseTableSpecs.BALL_RADIUS * ZOOM_FACTOR
    facecolor, edgecolor = color_set
    ball_circle = Circle(pos, radius, facecolor=facecolor, edgecolor=edgecolor, alpha=1.0, linewidth=1)
    ax.add_patch(ball_circle)
    return ball_circle


def create_pocket(ax, pos, theta1, theta2, pocket_id, radius):
    sector = Wedge(pos, radius, theta1, theta2, facecolor="grey", edgecolor="black", alpha=0.85)
    sector.set_picker(True)
    sector.set_gid(pocket_id)
    ax.add_patch(sector)
    return sector


def create_table_axes():
    disable_conflicting_matplotlib_keys()
    fig_length, fig_width = table_plot_size()
    fig, ax = plt.subplots(figsize=(fig_length, fig_width + HELP_AREA_HEIGHT))
    ax.set_xlim(0, fig_length + 1)
    ax.set_ylim(-HELP_AREA_HEIGHT, fig_width + 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.text(fig_length / 2 - 0.5, fig_width + 1, "CueRouter Selector")

    table_rect = Rectangle(
        TABLE_ORIGIN,
        fig_length,
        fig_width,
        linewidth=1,
        edgecolor="black",
        facecolor="g",
        alpha=0.95,
    )
    ax.add_patch(table_rect)

    return fig, ax


def create_pockets(ax):
    pocket_pos = np.asarray(ChineseTableSpecs.POCKET_POS, dtype=float)[:, ::-1]
    fig_pocket_pos = pocket_pos * ZOOM_FACTOR + 0.4
    fig_radius = 0.1 * ZOOM_FACTOR

    pocket_specs = {
        "lb": (fig_pocket_pos[0], 0, 90),
        "lc": (fig_pocket_pos[1], 0, 180),
        "lt": (fig_pocket_pos[2] + np.asarray((0.225, 0.0)), 90, 180),
        "rb": (fig_pocket_pos[3] + np.asarray((0.0, 0.225)), -90, 0),
        "rc": (fig_pocket_pos[4] + np.asarray((0.0, 0.225)), 180, 0),
        "rt": (fig_pocket_pos[5] + 0.225, 180, 270),
    }

    return {
        pocket_id: create_pocket(ax, pos.copy(), theta1, theta2, pocket_id, fig_radius)
        for pocket_id, (pos, theta1, theta2) in pocket_specs.items()
    }


class SearchRequestSelector:
    def __init__(self):
        self.fig, self.ax = create_table_axes()
        self.pocket_patches = create_pockets(self.ax)
        self.mode_sequence = MODE_SEQUENCE.copy()
        self.mode_idx = 0
        self.current_mode = self.mode_sequence[self.mode_idx]
        self.mode_history = []

        self.cue_ball_pos = None
        self.target_ball_pos = None
        self.target_ball_id = None
        self.target_pocket_id = None
        self.other_balls_info = {}
        self.target_region_points = []
        self.target_rect = None
        self.layout = None

        self.pending_ball_pos = None
        self.pending_ball_role = None
        self.artists_by_mode = {mode: [] for mode in self.mode_sequence}

        _, fig_width = table_plot_size()
        self.mode_text = self.ax.text(0.5, -0.25, "", va="top", ha="left", fontsize=10)
        self.help_text = self.ax.text(0.5, -0.55, "", va="top", ha="left", fontsize=9, linespacing=1.25)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("pick_event", self.on_pick)
        self.update_text()
        self.ax.set_ylim(-HELP_AREA_HEIGHT, fig_width + 2)

    def select(self):
        plt.show()
        return self.layout

    def update_text(self, extra=""):
        self.mode_text.set_text(f"Selection mode: {self.current_mode}")
        help_by_mode = {
            "cue_ball": "Click on the table to place the cue ball. Press backspace/u to undo.",
            "target_ball": "Click target ball position, then press 1-9 or QWERTY for 10-15.",
            "pocket": "Click one grey pocket wedge.",
            "other_balls": "Optional: click a ball position, then press 1-9 or QWERTY. Press esc/enter to continue.",
            "target_region": "Click two opposite corners of the target rectangle.",
        }
        suffix = f" {extra}" if extra else ""
        self.help_text.set_text(textwrap.fill(help_by_mode[self.current_mode] + suffix, HELP_WRAP_WIDTH))
        self.fig.canvas.draw_idle()

    def advance_mode(self):
        self.mode_history.append(self.current_mode)
        self.mode_idx = min(self.mode_idx + 1, len(self.mode_sequence) - 1)
        self.current_mode = self.mode_sequence[self.mode_idx]
        self.pending_ball_pos = None
        self.pending_ball_role = None
        self.update_text()

    def undo(self):
        if self.pending_ball_pos is not None:
            self.pending_ball_pos = None
            self.pending_ball_role = None
            self.update_text("Pending ball placement cleared.")
            return

        self.clear_mode_artists(self.current_mode)
        if self.current_mode == "cue_ball":
            self.cue_ball_pos = None
        elif self.current_mode == "target_ball":
            self.target_ball_pos = None
            self.target_ball_id = None
        elif self.current_mode == "pocket":
            self.target_pocket_id = None
            for patch in self.pocket_patches.values():
                patch.set_facecolor("grey")
        elif self.current_mode == "other_balls":
            self.other_balls_info.clear()
        elif self.current_mode == "target_region":
            self.target_region_points.clear()
            self.target_rect = None
            self.layout = None

        if self.mode_history:
            previous_mode = self.mode_history.pop()
            self.mode_idx = self.mode_sequence.index(previous_mode)
            self.current_mode = previous_mode

        self.update_text("Undone.")

    def clear_mode_artists(self, mode):
        for artist in self.artists_by_mode[mode]:
            artist.remove()
        self.artists_by_mode[mode] = []

    def add_mode_artist(self, artist):
        self.artists_by_mode[self.current_mode].append(artist)

    def on_click(self, event):
        if self.layout is not None:
            self.update_text("Selection complete. Close this window to continue, or undo to revise.")
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        pos_plot = (event.xdata, event.ydata)

        if self.current_mode == "pocket":
            return
        if not in_table_plot(pos_plot):
            self.update_text("Click inside the green table area.")
            return

        pos_table = plot_to_table(pos_plot)
        if self.current_mode == "cue_ball":
            self.cue_ball_pos = pos_table
            self.clear_mode_artists("cue_ball")
            self.add_mode_artist(create_ball(self.ax, pos_plot, BALL_COLORS["cue"]))
            self.advance_mode()
        elif self.current_mode == "target_ball":
            self.pending_ball_pos = pos_table
            self.pending_ball_role = "target"
            self.update_text("Now press target ball id: 1-9 or QWERTY for 10-15.")
        elif self.current_mode == "other_balls":
            self.pending_ball_pos = pos_table
            self.pending_ball_role = "other"
            self.update_text("Now press other ball id, or esc/enter to finish optional balls.")
        elif self.current_mode == "target_region":
            self.target_region_points.append(pos_table)
            marker = Circle(pos_plot, 0.035, facecolor="red", edgecolor="black")
            self.ax.add_patch(marker)
            self.add_mode_artist(marker)
            if len(self.target_region_points) == 2:
                self.finish_target_region()
            else:
                self.update_text("Click the opposite rectangle corner.")

    def on_pick(self, event):
        if self.current_mode != "pocket":
            return
        pocket_id = event.artist.get_gid()
        if pocket_id not in ChineseTableSpecs.POCKET_ID:
            return
        self.target_pocket_id = pocket_id
        for item_id, patch in self.pocket_patches.items():
            patch.set_facecolor("gold" if item_id == pocket_id else "grey")
        self.advance_mode()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in {"backspace", "u"}:
            self.undo()
            return

        if self.current_mode == "other_balls" and key in {"escape", "esc", "enter", "return"}:
            self.advance_mode()
            return

        if self.current_mode not in {"target_ball", "other_balls"}:
            return
        if self.pending_ball_pos is None:
            self.update_text("Click a ball position before pressing a ball id.")
            return

        ball_id = BALL_KEY_MAP.get(key)
        if ball_id is None:
            self.update_text("Unsupported key. Use 1-9 or QWERTY for 10-15.")
            return
        if ball_id == "cue":
            return
        if self.current_mode == "target_ball":
            self.target_ball_id = ball_id
            self.target_ball_pos = self.pending_ball_pos
            self.clear_mode_artists("target_ball")
            self.add_mode_artist(create_ball(self.ax, table_to_plot(self.target_ball_pos), BALL_COLORS[ball_id]))
            self.pending_ball_pos = None
            self.pending_ball_role = None
            self.advance_mode()
        else:
            if ball_id == self.target_ball_id:
                self.update_text("That id is already used by the target ball.")
                return
            if ball_id in self.other_balls_info:
                self.update_text("That id is already used by another ball.")
                return
            self.other_balls_info[ball_id] = self.pending_ball_pos
            self.add_mode_artist(create_ball(self.ax, table_to_plot(self.pending_ball_pos), BALL_COLORS[ball_id]))
            self.pending_ball_pos = None
            self.pending_ball_role = None
            self.update_text("Added. Click another ball or press esc/enter to continue.")

    def finish_target_region(self):
        (x1, y1), (x2, y2) = self.target_region_points
        self.target_rect = Rect(x1, y1, x2, y2)

        p1 = table_to_plot((self.target_rect.xmin, self.target_rect.ymin))
        p2 = table_to_plot((self.target_rect.xmax, self.target_rect.ymax))
        xy = (min(p1[0], p2[0]), min(p1[1], p2[1]))
        width = abs(p2[0] - p1[0])
        height = abs(p2[1] - p1[1])
        rect_patch = Rectangle(xy, width, height, facecolor="none", edgecolor="red", linewidth=1.5)
        self.ax.add_patch(rect_patch)
        self.add_mode_artist(rect_patch)

        self.layout = PoolLayout(
            cue_ball_pos=self.cue_ball_pos,
            target_ball_pos=self.target_ball_pos,
            target_ball_id=self.target_ball_id,
            other_balls_info=self.other_balls_info or None,
            target_pocket_id=self.target_pocket_id,
            target_rect=self.target_rect,
        )
        self.update_text("Selection complete. Close this window to continue.")


def select_pool_layout():
    return SearchRequestSelector().select()
