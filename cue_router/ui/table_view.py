import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np

from cue_router.core.table import ChineseTableSpecs


ZOOM_FACTOR = 4
TABLE_ORIGIN = np.asarray((0.5, 0.5), dtype=float)
HELP_AREA_HEIGHT = 1.6
BALL_COLORS = {
    "cue": ("white", "black"),
    "1": ("yellow", "goldenrod"),
    "2": ("blue", "navy"),
    "3": ("red", "darkred"),
    "4": ("pink", "deeppink"),
    "5": ("orange", "darkorange"),
    "6": ("green", "darkgreen"),
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


def disable_conflicting_matplotlib_keys():
    for keymap_name in ("keymap.quit", "keymap.quit_all"):
        plt.rcParams[keymap_name] = [
            key for key in plt.rcParams[keymap_name]
            if key.lower() != "q"
        ]

    for keymap_name in "keymap.xscale", "keymap.yscale":
        plt.rcParams[keymap_name] = [
            key for key in plt.rcParams[keymap_name]
            if key.lower() != "l"
        ]


def table_plot_size():
    return (
        ChineseTableSpecs.TABLE_LENGTH * ZOOM_FACTOR,
        ChineseTableSpecs.TABLE_WIDTH * ZOOM_FACTOR,
    )


def table_to_plot(pos):
    x, y = pos
    return np.asarray(((ChineseTableSpecs.TABLE_LENGTH - y) * ZOOM_FACTOR, x * ZOOM_FACTOR), dtype=float) + TABLE_ORIGIN


def plot_to_table(pos):
    x_plot, y_plot = pos
    return (
        float((y_plot - TABLE_ORIGIN[1]) / ZOOM_FACTOR),
        float(ChineseTableSpecs.TABLE_LENGTH - ((x_plot - TABLE_ORIGIN[0]) / ZOOM_FACTOR)),
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


def create_table_axes(table_color="g"):
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
        facecolor=table_color,
        alpha=0.95,
    )
    ax.add_patch(table_rect)

    return fig, ax


def create_pockets(ax):
    pocket_pos = np.asarray(ChineseTableSpecs.POCKET_POS, dtype=float)[:, ::-1]
    fig_pocket_pos = pocket_pos * ZOOM_FACTOR + 0.4
    fig_radius = 0.1 * ZOOM_FACTOR

    pocket_specs = {
        "lt": (fig_pocket_pos[0], 0, 90),
        "lc": (fig_pocket_pos[1], 0, 180),
        "lb": (fig_pocket_pos[2] + np.asarray((0.225, 0.0)), 90, 180),
        "rt": (fig_pocket_pos[3] + np.asarray((0.0, 0.225)), -90, 0),
        "rc": (fig_pocket_pos[4] + np.asarray((0.0, 0.225)), 180, 0),
        "rb": (fig_pocket_pos[5] + 0.225, 180, 270),
    }

    return {
        pocket_id: create_pocket(ax, pos.copy(), theta1, theta2, pocket_id, fig_radius)
        for pocket_id, (pos, theta1, theta2) in pocket_specs.items()
    }


def add_heatmap_target(ax, heatmap, alpha=0.8, cmap="viridis", colorbar=True, vmin=0.0):
    scores = np.ma.masked_where(heatmap.scores <= 0, heatmap.scores)
    image = scores.T[:, ::-1]
    extent = (
        TABLE_ORIGIN[0],
        TABLE_ORIGIN[0] + ChineseTableSpecs.TABLE_LENGTH * ZOOM_FACTOR,
        TABLE_ORIGIN[1],
        TABLE_ORIGIN[1] + ChineseTableSpecs.TABLE_WIDTH * ZOOM_FACTOR,
    )
    artist = ax.imshow(
        image,
        extent=extent,
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=1.0,
        interpolation="bilinear",
    )
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, fraction=0.035, pad=0.02, label="position quality")
    return artist


def show_heatmap_target(heatmap, layout=None, show=True, title="CueRouter Heatmap Target", target_region_thres=0.4):
    fig, ax = create_table_axes(table_color="none")
    create_pockets(ax)
    add_heatmap_target(ax, heatmap, vmin=target_region_thres)
    ax.set_title(title)

    if layout is not None:
        create_ball(ax, table_to_plot(layout.cue_ball_pos), BALL_COLORS["cue"])
        if hasattr(layout, "target_ball_pos"):
            create_ball(ax, table_to_plot(layout.target_ball_pos), BALL_COLORS[layout.target_ball_id])
            other_balls_info = layout.other_balls_info
        else:
            other_balls_info = layout.balls_info
        if other_balls_info:
            for ball_id, pos in other_balls_info.items():
                create_ball(ax, table_to_plot(pos), BALL_COLORS.get(ball_id, ("white", "black")))
    if show:
        plt.show()
    return fig, ax
