import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from cue_router.core.geometry import LineSegment, Rect
from cue_router.core.table import ChineseTableSpecs
from cue_router.ui.layout import ExecutorLayout
from cue_router.ui.table_view import (
    BALL_COLORS,
    HELP_AREA_HEIGHT,
    create_ball,
    create_pockets,
    create_table_axes,
    in_table_plot,
    plot_to_table,
    table_plot_size,
    table_to_plot,
)


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
REGION_KEY_MAP = {
    "r": Rect,
    "l": LineSegment,
}
ROUTE_TYPE_DEFAULT = "direct"
ROUTE_TYPE_KEY_MAP = {
    "d": "direct",
    "b": "bank",
}
MODE_SEQUENCE = ["cue_ball", "target_ball", "pocket", "route_type", "other_balls", "target_region"]


class ExecutorLayoutSelector:
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
        self.route_type = ROUTE_TYPE_DEFAULT
        self.other_balls_info = {}
        self.target_region_points = []
        self.region_mode = None
        self.target_region = None
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
            "route_type": "Choose route type: D for direct, B for bank. Press esc/enter to use direct.",
            "other_balls": "Optional: click a ball position, then press 1-9 or QWERTY. Press esc/enter to continue.",
            "target_region": "Choose a target region mode: rectangle (R) or line (L). Then click two opposite corners of the target region.",
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
        elif self.current_mode == "route_type":
            self.route_type = ROUTE_TYPE_DEFAULT
        elif self.current_mode == "other_balls":
            self.other_balls_info.clear()
        elif self.current_mode == "target_region":
            self.target_region_points.clear()
            self.target_region = None
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
            if not self.region_mode:
                self.update_text("Select a target region mode first. R for rect, L for line.")
                return
            self.target_region_points.append(pos_table)
            marker = Circle(pos_plot, 0.035, facecolor="red", edgecolor="black")
            self.ax.add_patch(marker)
            self.add_mode_artist(marker)
            if len(self.target_region_points) == 2:
                self.finish_target_region()
            else:
                self.update_text("Click the opposite rectangle/line corner.")

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

        if self.current_mode == "route_type":
            if key in {"escape", "esc", "enter", "return"}:
                self.route_type = ROUTE_TYPE_DEFAULT
                self.advance_mode()
                return
            route_type = ROUTE_TYPE_KEY_MAP.get(key)
            if route_type is None:
                self.update_text("Unsupported key. Use D for direct, B for bank, or esc/enter for direct.")
                return
            self.route_type = route_type
            self.advance_mode()
            return

        if self.current_mode == "target_region":
            region_cls = REGION_KEY_MAP.get(key)
            if region_cls is None:
                self.update_text("Unsupported key. Use R for rectangle or L for line.")
                return
            self.region_mode = region_cls
            self.target_region_points.clear()
            self.update_text(f"Region mode: {region_cls.__name__}. Click two points.")
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
        if self.current_mode == "target_ball":
            self.target_ball_id = ball_id
            self.target_ball_pos = self.pending_ball_pos
            self.clear_mode_artists("target_ball")
            self.add_mode_artist(create_ball(self.ax, table_to_plot(self.target_ball_pos), BALL_COLORS[ball_id]))
            self.pending_ball_pos = None
            self.pending_ball_role = None
            self.advance_mode()
        elif self.current_mode == "other_balls":
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

        if self.region_mode == Rect:
            self.target_region = Rect(x1, y1, x2, y2)

            p1 = table_to_plot((self.target_region.xmin, self.target_region.ymin))
            p2 = table_to_plot((self.target_region.xmax, self.target_region.ymax))
            xy = (min(p1[0], p2[0]), min(p1[1], p2[1]))
            width = abs(p2[0] - p1[0])
            height = abs(p2[1] - p1[1])
            rect_patch = Rectangle(xy, width, height, facecolor="none", edgecolor="red", linewidth=1.5)
            self.ax.add_patch(rect_patch)
            self.add_mode_artist(rect_patch)
        elif self.region_mode == LineSegment:
            self.target_region = LineSegment(x1, y1, x2, y2)
            p1 = table_to_plot((x1, y1))
            p2 = table_to_plot((x2, y2))
            x = (p1[0], p2[0])
            y = (p1[1], p2[1])
            line = Line2D(x, y, color="red", linewidth=2)
            self.ax.add_line(line)
            self.add_mode_artist(line)

        self.layout = ExecutorLayout(
            cue_ball_pos=self.cue_ball_pos,
            target_ball_pos=self.target_ball_pos,
            target_ball_id=self.target_ball_id,
            other_balls_info=self.other_balls_info or None,
            target_pocket_id=self.target_pocket_id,
            target_region=self.target_region,
            route_type=self.route_type,
        )
        self.update_text("Selection complete. Close this window to continue.")


def select_executor_layout():
    return ExecutorLayoutSelector().select()
