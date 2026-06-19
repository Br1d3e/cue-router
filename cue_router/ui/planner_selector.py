import textwrap

import matplotlib.pyplot as plt

from cue_router.ui.layout import PlannerLayout
from cue_router.ui.table_view import (
    BALL_COLORS,
    HELP_AREA_HEIGHT,
    create_ball,
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
MODE_SEQUENCE = ["cue_ball", "object_balls"]


class PlannerLayoutSelector:
    def __init__(self):
        self.fig, self.ax = create_table_axes()
        self.mode_idx = 0
        self.current_mode = MODE_SEQUENCE[self.mode_idx]
        self.cue_ball_pos = None
        self.balls_info = {}
        self.pending_ball_pos = None
        self.layout = None
        self.artists_by_mode = {mode: [] for mode in MODE_SEQUENCE}

        _, fig_width = table_plot_size()
        self.mode_text = self.ax.text(0.5, -0.25, "", va="top", ha="left", fontsize=10)
        self.help_text = self.ax.text(0.5, -0.55, "", va="top", ha="left", fontsize=9, linespacing=1.25)

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_text()
        self.ax.set_ylim(-HELP_AREA_HEIGHT, fig_width + 2)

    def select(self):
        plt.show()
        return self.layout

    def update_text(self, extra=""):
        self.mode_text.set_text(f"Selection mode: {self.current_mode}")
        help_by_mode = {
            "cue_ball": "Click on the table to place the cue ball. Press backspace/u to undo.",
            "object_balls": "Click an object ball position, then press 1-9 or QWERTY for 10-15. Press esc/enter when done.",
        }
        suffix = f" {extra}" if extra else ""
        self.help_text.set_text(textwrap.fill(help_by_mode[self.current_mode] + suffix, HELP_WRAP_WIDTH))
        self.fig.canvas.draw_idle()

    def advance_mode(self):
        self.mode_idx = min(self.mode_idx + 1, len(MODE_SEQUENCE) - 1)
        self.current_mode = MODE_SEQUENCE[self.mode_idx]
        self.pending_ball_pos = None
        self.update_text()

    def undo(self):
        if self.pending_ball_pos is not None:
            self.pending_ball_pos = None
            self.update_text("Pending ball placement cleared.")
            return

        if self.current_mode == "cue_ball":
            self.clear_mode_artists("cue_ball")
            self.cue_ball_pos = None
        elif self.current_mode == "object_balls":
            self.clear_mode_artists("object_balls")
            self.balls_info.clear()
            self.layout = None
            self.current_mode = "cue_ball"

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
        if not in_table_plot(pos_plot):
            self.update_text("Click inside the table area.")
            return

        pos_table = plot_to_table(pos_plot)
        if self.current_mode == "cue_ball":
            self.cue_ball_pos = pos_table
            self.clear_mode_artists("cue_ball")
            self.add_mode_artist(create_ball(self.ax, pos_plot, BALL_COLORS["cue"]))
            self.advance_mode()
        elif self.current_mode == "object_balls":
            self.pending_ball_pos = pos_table
            self.update_text("Now press object ball id: 1-9 or QWERTY for 10-15.")

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in {"backspace", "u"}:
            self.undo()
            return

        if self.current_mode == "object_balls" and key in {"escape", "esc", "enter", "return"}:
            self.finish_layout()
            return

        if self.current_mode != "object_balls":
            return
        if self.pending_ball_pos is None:
            self.update_text("Click a ball position before pressing a ball id.")
            return

        ball_id = BALL_KEY_MAP.get(key)
        if ball_id is None:
            self.update_text("Unsupported key. Use 1-9 or QWERTY for 10-15.")
            return
        if ball_id in self.balls_info:
            self.update_text("That id is already used by another ball.")
            return

        self.balls_info[ball_id] = self.pending_ball_pos
        self.add_mode_artist(create_ball(self.ax, table_to_plot(self.pending_ball_pos), BALL_COLORS[ball_id]))
        self.pending_ball_pos = None
        self.update_text("Added. Click another ball or press esc/enter to finish.")

    def finish_layout(self):
        if not self.balls_info:
            self.update_text("Add at least one object ball before finishing.")
            return
        self.layout = PlannerLayout(
            cue_ball_pos=self.cue_ball_pos,
            balls_info=self.balls_info.copy(),
        )
        self.update_text("Selection complete. Close this window to continue.")


def select_planner_layout():
    return PlannerLayoutSelector().select()
