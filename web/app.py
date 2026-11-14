import asyncio
import base64
import inspect
import json
from copy import deepcopy
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
from nicegui import run, ui, app
import numpy as np
from pathlib import Path
from PIL import Image
from enum import Enum, auto
from dataclasses import dataclass
import pooltool as pt
from pooltool.ani.constants import model_dir
import subprocess

from cue_router.executor.action import ShotAction
from cue_router.core.request import SearchRequest, RouteType
from cue_router.planner.planner_evaluation import ShotOption
from cue_router.planner.planner_search import PlannerConfig, PlannerSearch
from cue_router.planner.two_ply_search import TwoPlyConfig, TwoPlyPlan, TwoPlySearch
from cue_router.ui.layout import ExecutorLayout, PlannerLayout
from cue_router.executor.search import SearchConfig, PolicySearch
from cue_router.executor.shot_evaluation import ShotEvaluation
from cue_router.ui.table_view import show_heatmap_target

from table_vision.vision import TableVision, TableLayout, Detection
from cue_router.core.table import ChineseTableSpecs
from cue_router.core.geometry import HeatmapTarget, LineSegment, Rect, PositionTarget
from web.renderer import render_frames

MODEL_PATH = Path(__file__).parent.parent / "table_vision" / "models" / "yolo_3.pt"

W = ChineseTableSpecs.TABLE_WIDTH
L = ChineseTableSpecs.TABLE_LENGTH
R = ChineseTableSpecs.BALL_RADIUS

M = 20
PLAY_WIDTH = 800
PLAY_HEIGHT = 400
R_POCKET = 12
R_BALL = 11

POCKETS_SCREEN = {
    "lb": (M, M),
    "lc": (M + PLAY_WIDTH / 2, M),
    "lt": (M + PLAY_WIDTH, M),
    "rb": (M, PLAY_HEIGHT + M),
    "rc": (M + PLAY_WIDTH / 2, M + PLAY_HEIGHT),
    "rt": (M + PLAY_WIDTH, M + PLAY_HEIGHT),
}

BALL_COLORS = {
    "cue": ["white", "white"],
    "1": ["#F1CC47", "#F1CC47"],
    "2": ["#306EB3", "#306EB3"],
    "3": ["red", "red"],
    "4": ["#EF9497", "#EF9497"],
    "5": ["#EA6D2F", "#EA6D2F"],
    "6": ["#3F8C3E", "#3F8C3E"],
    "7": ["#C27730", "#C27730"],
    "8": ["black", "black"],
    "9": ["#F1CC47", "white"],
    "10": ["#306EB3", "white"],
    "11": ["red", "white"],
    "12": ["#EF9497", "white"],
    "13": ["#EA6D2F", "white"],
    "14": ["#3F8C3E", "white"],
    "15": ["#C27730", "white"],
}

EXECUTOR_SELECTION_SEQUENCE = ["target_ball", "pocket", "target_region"]

app.add_static_files(
    "/pooltool-english",
    str(model_dir / "hud" / "english"),
)

class Stage(Enum):
    UPLOAD = auto()
    MANUAL_POCKETS = auto()
    REVIEW = auto()
    SELECT = auto()
    EXECUTE = auto()
    DONE = auto()

    def __repr__(self):
        return self.name


@dataclass
class ExecutorSelection:
    target_ball_id: str = None
    target_pocket_id: str = None
    target_region_type: type[PositionTarget] | None = None
    pending_corner: tuple[int, int] = None
    target_region: PositionTarget = None
    route_type: str = "direct"

    def is_complete(self):
        return all([self.target_ball_id, self.target_pocket_id, self.target_region, self.route_type])


@dataclass
class CachedShot:
    id: int
    score: float = 0.0
    policy: ShotAction = None
    score_breakdown: dict[str, float] = None
    output_dir: Path | str = None
    video_path: Path | str = None

@dataclass
class CachedPlan:
    id: int
    plan: ShotOption
    output_dir: Path | str = None
    image_path: Path | str = None
    video_path: Path | str = None

SEARCH_CONFIG = SearchConfig(
    display_k=5,
    search_k=10,
    refine_k=5,
    mode="elite",
    quick_mode="stochastic",
    n_rounds=3,
    stochastic_max_iter=800,
    elite_max_iter=600,
    elite_frac=0.4,
    sample_mode="LHS",
    seed=42,
)

PLANNER_CONFIG = PlannerConfig(
    display_k=5,
)

TWO_PLY_CONFIG = TwoPlyConfig(
    planner_top_k=5,
)

TWO_PLY = False


def to_screen(x, y):
    return M + y / L * PLAY_WIDTH, M + x / W * PLAY_HEIGHT


def to_table(screen_x, screen_y):
    return (screen_y - M) / PLAY_HEIGHT * W, (screen_x - M) / PLAY_WIDTH * L


def in_table(x, y, allow_margin=False):
    if allow_margin:
        return 0 <= x <= PLAY_WIDTH + 2 * M and 0 <= y <= PLAY_HEIGHT + 2 * M
    return M <= x <= M + PLAY_WIDTH and M <= y <= M + PLAY_HEIGHT


def place_review_ball(layout, ball_id, position, source=None):
    """Commit an Add/Move only when its exact identity and position are valid."""
    if ball_id not in BALL_COLORS:
        raise ValueError("Choose a ball identity.")
    x, y = position
    if not (R < x < W - R and R < y < L - R):
        raise ValueError("Place the ball inside the cushions.")
    for field in ("balls_info", "unresolved_balls_info"):
        for other_id, other_pos in getattr(layout, field).items():
            if source == (field, other_id):
                continue
            if other_id == ball_id:
                raise ValueError(f"Ball {ball_id} already exists.")
            if np.linalg.norm(np.asarray(position) - other_pos) < 2 * R:
                raise ValueError(f"Position overlaps ball {other_id}.")
    field = source[0] if source else "balls_info"
    getattr(layout, field)[ball_id] = (float(x), float(y))
    layout.ball_ids = set(layout.balls_info)
    layout.unused_ball_ids = set(BALL_COLORS) - layout.ball_ids


def render_svg(layout: TableLayout):
    parts = [
        f'<rect width="{2 * M + PLAY_WIDTH}" height="{2 * M + PLAY_HEIGHT}" rx="15" fill="#674630"/>',
        f'<rect x="{M}" y="{M}" width="{PLAY_WIDTH}" height="{PLAY_HEIGHT}" fill="#176650"/>',
    ]

    for px, py in POCKETS_SCREEN.values():
        parts.append(
            f'<circle cx="{px}" cy="{py}" r="{R_POCKET}" fill="2C2C2C" opacity="0.8"/>'
        )

    if layout is None:
        return "".join(parts)

    for ball_id, (x, y) in layout.balls_info.items():
        sx, sy = to_screen(x, y)
        stroke, fill = BALL_COLORS[ball_id]
        parts.append(
            f'<circle cx="{sx}" cy="{sy}" r="{R_BALL}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )

    for ball_id, (x, y) in layout.unresolved_balls_info.items():
        sx, sy = to_screen(x, y)
        stroke, fill = BALL_COLORS[ball_id]
        parts.append(
            f'<circle cx="{sx}" cy="{sy}" r={R_BALL} '
            f'fill="{fill}" stroke="black" opacity="0.3" stroke-width="2" stroke-dasharray="2"/>'
        )

    return "".join(parts)


def in_circle(x, y, target_pos, r):
    return abs(target_pos[0] - x) <= r and abs(target_pos[1] - y) <= r


def resolve_screen_pos(x, y, balls_screen: dict[str, tuple[int, int]],
                       unresolved_balls_screen: dict[str, tuple[int, int]]):
    for pocket_id, pos in POCKETS_SCREEN.items():
        if in_circle(x, y, pos, R_POCKET):
            return {
                "type": "pocket",
                "id": pocket_id,
            }

    if balls_screen is not None:
        for ball_id, pos in balls_screen.items():
            if in_circle(x, y, pos, R_BALL):
                return {
                    "type": "ball",
                    "id": ball_id,
                }

    if unresolved_balls_screen is not None:
        for ball_id, pos in unresolved_balls_screen.items():
            if in_circle(x, y, pos, R_BALL):
                return {
                    "type": "ball_unresolved",
                    "id": ball_id,
                }

    return {
        "type": "table"
    }


def render_rect(img, x1, y1, x2, y2):
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    rect = (
        f'<rect x="{x1}" y="{y1}" '
        f'width="{x2 - x1}" height="{y2 - y1}" '
        f'fill="none" stroke="red" stroke-width="6" />'
    )

    img.content += rect
    return img


def redraw_regions(content, regions, pending_corner=None, region_type: type[PositionTarget] = Rect):
    """Draw rectangle or line annotations using image/screen coordinates."""
    parts = []
    for x1, y1, x2, y2 in regions:
        if region_type is LineSegment:
            shape = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        else:
            shape = (
                f'<rect x="{min(x1, x2)}" y="{min(y1, y2)}" '
                f'width="{abs(x2 - x1)}" height="{abs(y2 - y1)}" '
            )
        parts.append(shape + 'fill="none" stroke="red" stroke-width="3" pointer-events="none" />')
    if pending_corner is not None:
        x, y = pending_corner
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="yellow" pointer-events="none" />')
    content.set_content("".join(parts))


def render_circle_highlight(content, x, y, r):
    highlight = f'<circle cx="{x}" cy="{y}" r="{r + 2}" fill="purple" opacity="0.7" stroke="black" stroke_width="4" stroke-dasharray="3" />'
    content.set_content(content.content + highlight)


def build_search_request(layout: ExecutorLayout):
    cue = pt.Cue(cue_ball_id="cue")
    chinese_table = ChineseTableSpecs.chinese_table
    shot = pt.System(table=chinese_table, balls=layout.get_balls(), cue=cue)
    shot.set_ballset(ChineseTableSpecs.ball_set)
    mapping = {
        "direct": RouteType.DIRECT,
        "bank": RouteType.BANK,
    }
    return SearchRequest(
        shot=shot,
        target_ball_id=layout.target_ball_id,
        target_pocket_id=layout.target_pocket_id,
        position_target=layout.target_region,
        route_type=mapping[layout.route_type],
    )


def executor_search(search_request: SearchRequest):
    searcher = PolicySearch(search_request)
    return searcher.search(search_config=SEARCH_CONFIG)

def render_video(output_dir):
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", str(output_dir / "frame_%05d.png"),
            "-c:v", "h264_videotoolbox",
            "-b:v", "5M",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_dir / "shot.mp4"),
        ],
        check=True,
    )
    return output_dir / "shot.mp4"

def render_plan_video(shot: pt.System, output_dir: Path):
    render_frames(shot, output_dir=output_dir)
    return render_video(output_dir)

def create_shot(layout: PlannerLayout | ExecutorLayout):
    cue = pt.Cue(cue_ball_id="cue")
    chinese_table = ChineseTableSpecs.chinese_table
    shot = pt.System(table=chinese_table, balls=layout.get_balls(), cue=cue)
    shot.set_ballset(ChineseTableSpecs.ball_set)
    return shot

def two_ply_planner_search(layout: PlannerLayout, legal_target_balls: list[str]):
    two_ply_planner = TwoPlySearch(PLANNER_CONFIG, TWO_PLY_CONFIG)
    shot = create_shot(layout)
    return two_ply_planner.search(
        shot,
        layout,
        legal_target_balls
    )

@ui.page("/")
def page():
    ui.page_title("CueRouter Demo")
    ui.label("CueRouter Demo")
    ui.add_body_html('<script>' + Path(__file__).with_name('position_editor.js').read_text() + '</script>')
    ui.add_css("""
        .position-viewport { position: relative; height: min(46dvh, 420px);
            min-height: 200px; width: 100%; touch-action: none; overflow: hidden;
            user-select: none; background: #202020; }
        .position-viewport svg { width: 100%; height: 100%; display: block; }
        .position-crosshair { position: absolute; left: 50%; top: 50%;
            transform: translate(-50%, -50%); width: 32px; height: 32px;
            pointer-events: none; filter: drop-shadow(0 0 2px black); }
        .position-crosshair::before, .position-crosshair::after {
            content: ''; position: absolute; background: white; }
        .position-crosshair::before { left: 15px; width: 2px; height: 32px; }
        .position-crosshair::after { top: 15px; height: 2px; width: 32px; }
        .position-panel { width: min(100%, 560px); max-height: 95dvh; overflow-y: auto;
            padding-bottom: max(16px, env(safe-area-inset-bottom)); }
        .review-tools .q-btn, .position-panel .q-btn { min-width: 48px; min-height: 48px; }
    """)

    state = {
        "captured_image": None,
        "layout": None,
        "mode": "Executor",
        "balls_screen": {},
        "unresolved_balls_screen": {},
        "stage": Stage.UPLOAD,
        "selection": None,
        "pocket_rects": {},
        "pending_corner": None,
        "selected_ball": None,
        "finish_manual_label": None,
        "cached_shots": {},
        "legal_target_balls": [str(i) for i in range(1, 8)],
        "cached_plans": {},
        "is_two_ply": TWO_PLY,
        "review_background": None,
    }

    def reset():
        state["review_background"] = None
        for name in ("captured_image", "layout", "mode", "selection", "pending_corner", "selected_ball", "finish_manual_label"):
            state[name] = None
        for name in ("balls_screen", "unresolved_balls_screen", "pocket_rects", "cached_shots", "cached_plans"):
            state[name] = {}
        state["stage"] = Stage.UPLOAD
        state_msg.set_text(f"Stage: {state["stage"]}")
        caption_msg.set_text("")
        table.set_content("")
        render_controls.refresh()

    async def next_stage(stage: Stage, next_state_func=None):
        stage = deepcopy(stage)
        if stage in [Stage.UPLOAD, Stage.MANUAL_POCKETS]:
            stage = Stage.REVIEW
        elif stage == Stage.REVIEW:
            if "cue" not in state["balls_screen"].keys():
                ui.notify("[⚠ Warning] No cue ball selected. Designate a cue ball to continue.")
                return
            state["unresolved_balls_screen"] = None
            # for ball_id, pos in state["layout"].unresolved_balls_info.items():
            #     state["layout"].balls_info["ball_id"] = pos
            state["layout"].unresolved_balls_info.clear()
            table.set_content(render_svg(state["layout"]))
            stage = Stage.SELECT
        elif stage == Stage.SELECT:
            ball_selector.set_content("")
            if state["mode"] == "Executor":
                state["executor_layout"] = build_executor_layout()
                if state["executor_layout"] is None:
                    return stage
            stage = Stage.EXECUTE
        elif stage == Stage.EXECUTE:
            stage = Stage.DONE

        state["stage"] = stage
        state_msg.set_text(f"Stage: {stage}")

        if callable(next_state_func):
            result = next_state_func()
            await render_controls.refresh()
            if inspect.isawaitable(result):
                await result
        else:
            await render_controls.refresh()

        return stage

    async def prev_stage(stage: Stage):
        if stage in [Stage.MANUAL_POCKETS, Stage.REVIEW]:
            stage = stage.UPLOAD
            ball_selector.set_content("")
            table.set_content("")
        elif stage == Stage.SELECT:
            state["mode"] = "Executor"
            stage = stage.REVIEW
        elif stage in [Stage.EXECUTE, Stage.DONE]:
            stage = Stage.SELECT
        state["stage"] = stage
        state_msg.set_text(f"Stage: {stage}")

        await render_controls.refresh()

        return stage

    def next_selection(selection):
        if selection == "target_ball":
            selection = "pocket"
            caption_msg.set_text("Select a target pocket.")
        elif selection == "pocket":
            selection = "target_region"
            caption_msg.set_text("Choose a target region type first. Then click the opposite corners of the region.")

        state["selection"] = selection
        render_controls.refresh()

        return selection

    def prev_selection(selection):
        if selection == "pocket":
            selection = "target_ball"
            caption_msg.set_text(f"Select a target ball to continue.")
        elif selection == "target_region":
            selection = "pocket"
            caption_msg.set_text("Select a target pocket.")

        state["selection"] = selection
        render_controls.refresh()
        return selection

    def cache_review_background(vision):
        # Match the existing table SVG orientation: screen x follows table y.
        transform = np.asarray([[0, PLAY_WIDTH / L, M], [PLAY_HEIGHT / W, 0, M], [0, 0, 1]])
        warped = cv2.warpPerspective(vision.image_state.image, transform @ vision.H,
                                     (PLAY_WIDTH + 2 * M, PLAY_HEIGHT + 2 * M))
        _, encoded = cv2.imencode(".jpg", warped)
        state["review_background"] = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")

    async def on_upload(e):
        image_bytes = await e.file.read()
        image = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        vision = TableVision(image, model_path=MODEL_PATH)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        state["captured_image"] = Image.fromarray(image)
        captured_image.set_source(state["captured_image"])
        state["stage"] = Stage.UPLOAD

        try:
            state["layout"] = vision.predict_layout(preview=False)
            cache_review_background(vision)
            cache_balls()
            state["stage"] = await next_stage(state["stage"])
            table.set_content(render_svg(state["layout"]))
        except RuntimeError as e:
            if str(e) == "NOT_ENOUGH_POCKETS":
                ui.notify("Automatic pocket calibration requires ≥ 5 pockets. Label the pockets or retake an image.")
            elif str(e) == "POCKET_IDENTITIES_NOT_FOUND":
                ui.notify(
                    "Could not determine pocket identities for homography calibration. Correct the pocket labels or retake an image.")
            image_state = vision.image_state
            state["stage"] = Stage.MANUAL_POCKETS
            state_msg.set_text(f"Stage: {state["stage"]}")
            for i, detection in enumerate(image_state.pockets):
                x1, y1, x2, y2 = detection.xyxy
                render_rect(captured_image, x1, y1, x2, y2)
                state["pocket_rects"][i] = detection.xyxy

            async def on_finish():
                detections = []
                for i, new_pos in state["pocket_rects"].items():
                    detections.append(Detection(
                        id=i,
                        class_id=11,
                        class_name="pocket",
                        xyxy=new_pos,
                        confidence=1,
                    ))
                new_image_state = deepcopy(image_state)
                new_image_state.pockets = detections
                try:
                    state["layout"] = vision.predict_layout(image_state=new_image_state)
                    cache_review_background(vision)
                    state["stage"] = await next_stage(state["stage"])
                    ui.notify("Layout generation successful.")
                    table.set_content(render_svg(state["layout"]))
                    cache_balls()
                    return
                except RuntimeError as e:
                    ui.notify("Retry")
                    state["stage"] = Stage.MANUAL_POCKETS

            if state["stage"] != Stage.MANUAL_POCKETS:
                return

            state["finish_manual_label"] = on_finish
            await render_controls.refresh()

    ui.upload(
        on_upload=on_upload,
        auto_upload=True,
    ).props("accept=image/*")

    async def open_position_editor(mode, ball_id, source=None):
        layout = state["layout"]
        position = getattr(layout, source[0])[ball_id] if source else (W / 2, L / 2)
        if state["review_background"]:
            scene = f'<image href="{state["review_background"]}" width="840" height="440"/>'
            for field in ("balls_info", "unresolved_balls_info"):
                for identity, xy in getattr(layout, field).items():
                    sx, sy = to_screen(*xy)
                    scene += (f'<circle cx="{sx}" cy="{sy}" r="13" fill="none" stroke="white"/>'
                              f'<text x="{sx + 15}" y="{sy}" fill="white" stroke="black" '
                              f'stroke-width="0.4" font-size="15">{identity}</text>')
        else:
            scene = render_svg(layout)
        # The editor must outlive the identity picker that opens it.
        with ui.context.client.content, ui.dialog().props("persistent") as dialog, ui.column().classes("bg-white rounded p-4 position-panel gap-3"):
            ui.label(f'{mode} {"cue ball" if ball_id == "cue" else "ball " + ball_id}').classes("text-lg")
            viewport = ui.html(
                f'<div class="position-viewport"><svg xmlns="http://www.w3.org/2000/svg">{scene}</svg>'
                '<div class="position-crosshair" aria-hidden="true"></div></div>',
                sanitize=False,
            ).classes("w-full")
            host = f"document.getElementById('c{viewport.id}').positionEditor"
            with ui.row().classes("w-full justify-center"):
                ui.button(icon="remove", on_click=lambda: ui.run_javascript(f"{host}.zoom(0.8)")).props('aria-label="Zoom out"')
                ui.button(icon="add", on_click=lambda: ui.run_javascript(f"{host}.zoom(1.25)")).props('aria-label="Zoom in"')
            async def close():
                await ui.run_javascript(f"{host}?.dispose()")
                dialog.close()
            async def confirm():
                point = await ui.run_javascript(f"return {host}.point()")
                try:
                    place_review_ball(layout, ball_id, to_table(*point), source)
                except ValueError as error:
                    ui.notify(str(error), type="warning")
                    return
                cache_balls()
                table.set_content(render_svg(layout))
                ball_selector.set_content("")
                await close()
            with ui.row().classes("w-full justify-end"):
                ui.button("Cancel", on_click=close).props("flat")
                confirm_button = ui.button("Confirm position", on_click=confirm)
                confirm_button.disable()
        dialog.on("hide", dialog.delete)
        dialog.open()
        await ui.run_javascript(
            "await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));"
            f"window.initCuePositionEditor('c{viewport.id}', {json.dumps(to_screen(*position))});"
        )
        confirm_button.enable()

    def choose_position_ball(mode):
        layout = state["layout"]
        with ui.dialog() as picker, ui.column().classes("bg-white rounded p-4 position-panel"):
            ui.label(f"{mode} ball").classes("text-lg")
            with ui.grid(columns=4).classes("w-full gap-2"):
                choices = [(identity, None) for identity in BALL_COLORS] if mode == "Add" else [
                    (identity, (field, identity))
                    for field in ("balls_info", "unresolved_balls_info")
                    for identity in getattr(layout, field)
                ]
                for identity, source in choices:
                    async def choose(identity=identity, source=source):
                        picker.close()
                        await open_position_editor(mode, identity, source)
                    button = ui.button("Cue" if identity == "cue" else identity, on_click=choose)
                    if mode == "Add" and (identity in layout.balls_info or identity in layout.unresolved_balls_info):
                        button.disable()
            ui.button("Cancel", on_click=picker.close).props("flat")
        picker.on("hide", picker.delete)
        picker.open()

    def review_ball(res):
        state["selected_ball"] = (res, None)
        source = ("balls_info" if res["type"] == "ball" else "unresolved_balls_info", res["id"])
        with ui.dialog() as picker, ui.column().classes("bg-white rounded p-4 position-panel"):
            ui.label("Cue ball" if res["id"] == "cue" else f'Ball {res["id"]}').classes("text-lg")
            with ui.grid(columns=4).classes("w-full gap-2"):
                for identity in BALL_COLORS:
                    def assign(identity=identity):
                        assign_ball("select-" + identity)
                        picker.close()
                    ui.button("Cue" if identity == "cue" else identity, on_click=assign)
            async def move():
                picker.close()
                await open_position_editor("Move", res["id"], source)
            with ui.row():
                ui.button("Move", icon="open_with", on_click=move)
                ui.button(icon="delete", on_click=lambda: (assign_ball("delete-ball"), picker.close())).props('aria-label="Delete ball"')
                ui.button("Cancel", on_click=picker.close).props("flat")
        picker.on("hide", picker.delete)
        picker.open()

    def on_table_click(e):
        ball_selector.set_content("")
        state["selected_ball"] = None
        if state["stage"] not in [Stage.REVIEW, Stage.SELECT]:
            return
        x, y = e.image_x, e.image_y
        res = resolve_screen_pos(x, y, balls_screen=state["balls_screen"],
                                 unresolved_balls_screen=state["unresolved_balls_screen"])

        if state["stage"] == Stage.REVIEW:
            if res["type"] not in ("ball", "ball_unresolved"):
                return
            review_ball(res)
        elif state["stage"] == Stage.SELECT:
            if not in_table(x, y, allow_margin=True):
                return
            if state["selection"] == "target_ball":
                if res["type"] == "ball":
                    if res["id"] == "cue":
                        caption_msg.set_text("Cannot select the cue ball. Choose a colored ball.")
                        return
                    screen_x, screen_y = state["balls_screen"][res["id"]]
                    if state["executor_selection"].target_ball_id == res["id"]:
                        caption_msg.set_text(f"Confirmed selecting target ball #{res["id"]}")
                        next_selection(state["selection"])
                        render_circle_highlight(ball_selector, screen_x, screen_y, R_BALL)
                    else:
                        ball_selector.set_content("")
                        state["executor_selection"].target_ball_id = res["id"]
                        caption_msg.set_text(f"Selected target ball #{res['id']}. Click it again to confirm.")
                        render_circle_highlight(ball_selector, screen_x, screen_y, R_BALL)
                else:
                    state["executor_selection"].target_ball_id = None
                    ball_selector.set_content("")
                    caption_msg.set_text(f"Select a target ball to continue.")
            elif state["selection"] == "pocket":
                if res["type"] == "pocket":
                    screen_x, screen_y = POCKETS_SCREEN[res["id"]]
                    if state["executor_selection"].target_pocket_id == res["id"]:
                        caption_msg.set_text(f"Confirmed selecting pocket {res['id']}.")
                        next_selection(state["selection"])
                        render_circle_highlight(ball_selector, screen_x, screen_y, R_POCKET)
                    else:
                        ball_selector.set_content("")
                        state["executor_selection"].target_pocket_id = res["id"]
                        caption_msg.set_text(f"Selected pocket {res['id']}. Click it again to confirm.")
                        render_circle_highlight(ball_selector, screen_x, screen_y, R_POCKET)
                else:
                    state["executor_selection"].target_pocket_id = None
                    ball_selector.set_content("")
                    caption_msg.set_text(f"Select a target pocket to continue.")
            elif state["selection"] == "target_region":
                selection = state["executor_selection"]
                if selection.target_region_type is None:
                    caption_msg.set_text(
                        "Choose a position target type: Line or Rectangle. Then click two endpoints or opposite corners.")
                    return
                if selection.pending_corner is None:
                    selection.pending_corner = x, y
                    selection.target_region = None
                    redraw_regions(ball_selector, [], selection.pending_corner, selection.target_region_type)
                else:
                    x1, y1 = selection.pending_corner
                    x2, y2 = x, y
                    tx1, ty1 = to_table(x1, y1)
                    tx2, ty2 = to_table(x2, y2)
                    selection.target_region = selection.target_region_type(tx1, ty1, tx2, ty2)
                    selection.pending_corner = None
                    redraw_regions(ball_selector, [(x1, y1, x2, y2)], region_type=selection.target_region_type)
                    caption_msg.set_text("Selected target region. Click Next to execute.")
                render_controls.refresh()

    def assign_ball(element_id):
        if state["stage"] != Stage.REVIEW or state["selected_ball"] is None:
            return
        if element_id != "delete-ball" and not element_id.startswith("select-"):
            return
        res, pos = state["selected_ball"]
        layout = state["layout"]
        source = None
        if res["type"] == "ball":
            source = layout.balls_info
        elif res["type"] == "ball_unresolved":
            source = layout.unresolved_balls_info
        if source is not None:
            pos = source[res["id"]]

        if element_id == "delete-ball":
            if source is None:
                return
            del source[res["id"]]
        else:
            ball_id = element_id.removeprefix("select-")
            if ball_id not in BALL_COLORS:
                return
            if ball_id in layout.balls_info and not (res["type"] == "ball" and res["id"] == ball_id):
                ui.notify(f"Ball {ball_id} already exists. Choose another ball.")
                return
            if source is not None:
                del source[res["id"]]
            layout.balls_info[ball_id] = pos

        layout.ball_ids = set(layout.balls_info)
        layout.unused_ball_ids = set(BALL_COLORS) - layout.ball_ids
        cache_balls()
        table.set_content(render_svg(layout))
        ball_selector.set_content("")
        state["selected_ball"] = None

    def on_image_click(e):
        if state["stage"] != Stage.MANUAL_POCKETS:
            return
        x, y = e.image_x, e.image_y

        for rect_id, (x1, y1, x2, y2) in list(state["pocket_rects"].items()):
            if x1 <= x <= x2 and y1 <= y <= y2:
                del state["pocket_rects"][rect_id]
                state["pending_corner"] = None
                redraw_regions(captured_image, state["pocket_rects"].values(), state["pending_corner"])
                return

        if state["pending_corner"] is None:
            state["pending_corner"] = (x, y)
        else:
            x1, y1 = state["pending_corner"]

            state["pocket_rects"][max(state["pocket_rects"], default=-1) + 1] = (
                min(x1, x), min(y1, y),
                max(x1, x), max(y1, y)
            )

            state["pending_corner"] = None

        redraw_regions(captured_image, state["pocket_rects"].values(), state["pending_corner"])

    def cache_balls():
        state["balls_screen"].clear()
        state["unresolved_balls_screen"].clear()
        if state["layout"] is not None:
            for ball_id, (x, y) in state["layout"].balls_info.items():
                state["balls_screen"][ball_id] = to_screen(x, y)

            for ball_id, (x, y) in state["layout"].unresolved_balls_info.items():
                state["unresolved_balls_screen"][ball_id] = to_screen(x, y)

    def update_state_value(field, value):
        state[field] = value

    def update_executor_selection(field, value):
        if state.get("executor_selection") is None:
            return
        setattr(state["executor_selection"], field, value)
        if field == "target_region_type":
            state["executor_selection"].pending_corner = None
            state["executor_selection"].target_region = None
            ball_selector.set_content("")
            render_controls.refresh()

    def select():
        if state["stage"] != Stage.SELECT:
            return
        layout = state["layout"]
        if layout is None:
            ui.notify(f"No layout selected. Choose a layout first.")
            return

        if state["mode"] == "Planner":
            planner_layout = layout.to_planner_layout()
            ui.notify("Auto-generated planner layout.")
            state["planner_layout"] = planner_layout
            caption_msg.set_text("Pick a color (solids / stripes).")
        elif state["mode"] == "Executor":
            state["executor_selection"] = ExecutorSelection()
            state["selection"] = EXECUTOR_SELECTION_SEQUENCE[0]
            caption_msg.set_text("Executor selection started. Pick a target ball first.")

    def build_executor_layout() -> ExecutorLayout | None:
        if not state["layout"]:
            ui.notify(f"No layout selected. Create a table layout first.")
            return None
        if not state["executor_selection"].is_complete():
            ui.notify(f"Failed to build executor layout. Recreate a complete selection.")
            return None
        target_ball_id = state["executor_selection"].target_ball_id
        return ExecutorLayout(
            cue_ball_pos=state["layout"].balls_info["cue"],
            target_ball_pos=state["layout"].balls_info[target_ball_id],
            target_ball_id=target_ball_id,
            target_pocket_id=state["executor_selection"].target_pocket_id,
            target_region=state["executor_selection"].target_region,
            other_balls_info={
                ball_id: pos
                for ball_id, pos in state["layout"].balls_info.items()
                if ball_id not in [target_ball_id, "cue"]
            },
            route_type=state["executor_selection"].route_type,
        )

    async def execute():
        if state["stage"] != Stage.EXECUTE:
            return
        caption_msg.set_visibility(False)
        if state["mode"] == "Planner":
            if state["planner_layout"] is None:
                ui.notify("No planner layout selected. Create a layout first.")
                return
            if state["is_two_ply"]:
                caption_msg.set_visibility(True)
                caption_msg.set_text("Two-ply searching...")
                plans = await run.cpu_bound(
                    two_ply_planner_search,
                    state["planner_layout"], state["legal_target_balls"]
                )
            else:
                planner = PlannerSearch(PLANNER_CONFIG)
                caption_msg.set_visibility(True)
                caption_msg.set_text("Searching...")
                # plans = planner.search(state["planner_layout"], state["legal_target_balls"])
                plans = await run.cpu_bound(
                    planner.search,
                    state["planner_layout"], state["legal_target_balls"],
                )

            caption_msg.set_text("Search complete.")
            state["plans"] = plans
            state["cached_plans"].clear()

            await next_stage(state["stage"])
        elif state["mode"] == "Executor":
            if state["executor_layout"] is None:
                ui.notify("No executor layout selected. Select a layout first.")
                return
            caption_msg.set_visibility(True)
            caption_msg.set_text("Searching…")
            search_request = build_search_request(state["executor_layout"])
            state["search_request"] = search_request
            scores, policies = await run.cpu_bound(
                executor_search,
                search_request,
            )

            caption_msg.set_text("Search complete.")

            state["scores"] = scores
            state["policies"] = policies

            await next_stage(state["stage"])

    def draw_cue_impact_diagram(content, label, a: float, b: float):
        x = 50 * (1 - a)
        y = 50 * (1 - b)

        content.set_content(f"""
            <div style="
                width: 160px;
                height: 160px;
                position: relative;
            ">
                <img
                    src="/pooltool-english/circle.png"
                    style="width: 100%; height: 100%;"
                >

                <img
                    src="/pooltool-english/crosshairs.png"
                    style="
                        position: absolute;
                        width: 14%;
                        left: {x}%;
                        top: {y}%;
                        transform: translate(-50%, -50%);
                    "
                >
            </div>
        """)

        # ui.label(f"Impact: ({a:+.3f}, {b:+.3f})")
        label.set_text(f"Impact: ({a:+.3f}, {b:+.3f})")

    def draw_cue_speed_diagram(content, label, V0: float, max_V0: float=6.0):
        width = 4
        parts = [
            f'<rect width="{500 + 2 * width}" height="{100 + 2 * width}" fill="grey" stroke="black" stroke-width="{width}" />',
            f'<rect x="4" y="4" width="{500 * V0 / max_V0}" height="100" fill="red" />'
        ]
        # ui.interactive_image(
        #     size=(508, 108),
        #     content="\n".join(parts),
        # )
        content.set_content("\n".join(parts))
        # ui.label(f"Cue speed: {V0:.1f} m/s")
        label.set_text(f"Cue speed {V0:.1f} m/s")

    state_msg = ui.label(f"Stage: {state["stage"]}")
    caption_msg = ui.label("")

    table = ui.interactive_image(
        size=(840, 440),
        # content=render_svg(state["layout"]),
        on_mouse=on_table_click,
    ).classes("w-full max-w-2xl")
    ball_selector = table.add_layer()

    @ui.refreshable
    def render_controls():
        global captured_image
        global table
        global ball_selector
        if state["stage"] == Stage.UPLOAD:
            captured_image = ui.interactive_image(
                on_mouse=on_image_click,
            ).classes("max-w-2xl")
        elif state["stage"] == Stage.MANUAL_POCKETS:
            captured_image = ui.interactive_image(
                source=state["captured_image"],
                on_mouse=on_image_click,
            ).classes("max-w-2xl")
            redraw_regions(captured_image, state["pocket_rects"].values(), state["pending_corner"])
            ui.button("Finish manual label", on_click=state["finish_manual_label"])
            ui.button("Back", on_click=lambda: prev_stage(state["stage"]))
        elif state["stage"] == Stage.REVIEW:
            with ui.row().classes("review-tools"):
                ui.button("Add", icon="add", on_click=lambda: choose_position_ball("Add"))
                ui.button("Move", icon="open_with", on_click=lambda: choose_position_ball("Move"))
            captured_image = ui.interactive_image(
                source=state["captured_image"],
                on_mouse=on_image_click,
            ).classes("max-w-2xl")
            ui.label("Choose a CueRouter mode:")
            ui.toggle(["Executor", "Planner"], value="Executor",
                      on_change=lambda e: update_state_value("mode", e.value))
            ui.button("Next", on_click=lambda f: next_stage(state["stage"], next_state_func=select))
            ui.button("Back", on_click=lambda: prev_stage(state["stage"]))
        elif state["stage"] == Stage.SELECT and state["mode"] == "Executor":
            ui.toggle(
                {"direct": "Direct", "bank": "Bank"}, value=state["executor_selection"].route_type,
                on_change=lambda e: update_executor_selection("route_type", e.value)
            )
            ui.button("Prev selection", on_click=lambda: prev_selection(state["selection"]))
            if state["selection"] == "target_region":
                ui.button("Line Segment", on_click=lambda: update_executor_selection("target_region_type", LineSegment))
                ui.button("Rectangle", on_click=lambda: update_executor_selection("target_region_type", Rect))
                if state["executor_selection"].is_complete():
                    ui.button("Next", on_click=lambda: next_stage(state["stage"], next_state_func=execute))
            ui.button("Back", on_click=lambda: prev_stage(state["stage"]))
        elif state["stage"] == Stage.SELECT and state["mode"] == "Planner":
            ball_color_map = {
                "Solids": [str(i) for i in range(1, 8)],
                "Stripes": [str(i) for i in range(9, 16)],
            }
            ui.toggle(
                ["Solids", "Stripes"], value="Solids",
                on_change=lambda e: update_state_value("legal_target_balls", ball_color_map[e.value])
            )
            ui.button("Next", on_click=lambda: next_stage(state["stage"], next_state_func=execute))
            ui.button("Back", on_click=lambda: prev_stage(state["stage"]))
        elif state["stage"] == Stage.DONE:
            ui.button("Back", on_click=lambda: prev_stage(state["stage"]))
            if state["mode"] == "Executor":
                if state["scores"] is None or state["policies"] is None:
                    ui.notify("Failed to fetch search results.")
                    return
                if len(state["scores"]) == 0:
                    ui.label("Failed to find a possible shot.")
                    return
                ui.label(f"Click to view a shot policy (Ranked by score, #1~#{len(state["scores"])}).")
                ui.toggle([f"#{i + 1}" for i in range(len(state["scores"]))], value="#1",
                          on_change=lambda e: render_shot_video(e.value))
                score_breakdown_desc = ui.label().classes("whitespace-pre-line")
                preview = ui.video(src=None, autoplay=True, loop=True, controls=True).classes("w-full max-w-2xl")
                cue_impact_diagram = ui.html()
                impact_label = ui.label()
                cue_speed_diagram = ui.interactive_image(size=(508, 108))
                speed_label = ui.label()

                async def render_shot_video(idx: int | str):
                    ui.notify(f"Preparing video {idx}.")
                    await asyncio.sleep(0.05)
                    if isinstance(idx, str):
                        idx = int(idx[1:])
                    if idx < 1 or idx > len(state["scores"]):
                        raise ValueError(f"Invalid shot index: {idx}. len(state['scores']) = {len(state['scores'])}")

                    if idx in state["cached_shots"].keys():
                        cached_shot = state["cached_shots"][idx]
                    else:
                        score = state["scores"][idx - 1]
                        policy = state["policies"][idx - 1]

                        phi, V0, a, b = policy.as_array()
                        shot = create_shot(state["executor_layout"])
                        shot.cue.set_state(V0=V0, phi=phi, a=a, b=b)

                        shot_evaluator = ShotEvaluation()
                        search_request = state["search_request"]
                        shot_evaluator.evaluate_shot_final(
                            shot,
                            search_request.target_ball_id,
                            search_request.target_pocket_id,
                            search_request.position_target,
                            route_type=search_request.route_type
                        )
                        score_breakdown = shot_evaluator.print_score_breakdown()

                        pt.simulate(shot, inplace=True)

                        output_dir = render_frames(shot)
                        video_path = await run.io_bound(
                            render_video,
                            output_dir,
                        )
                        ui.notify(f"Rendered video {idx}.")
                        cached_shot = CachedShot(
                            id=idx,
                            score=score,
                            policy=policy,
                            score_breakdown=score_breakdown,
                            output_dir=output_dir,
                            video_path=video_path,
                        )
                        state["cached_shots"][idx] = cached_shot

                    score_breakdown_desc.set_text("\n".join(
                        f"{name}: {value:.3f}"
                        for name, value in cached_shot.score_breakdown.items()
                    ))
                    preview.set_source(cached_shot.video_path)
                    phi, V0, a, b = cached_shot.policy.as_array()
                    draw_cue_impact_diagram(cue_impact_diagram, impact_label, a, b)
                    draw_cue_speed_diagram(cue_speed_diagram, speed_label, V0)

                ui.timer(0.1, lambda: render_shot_video(1), once=True)
            else:
                if state.get("plans") is None:
                    ui.notify("Unable to search for a plan.")
                    return
                if len(state["plans"]) == 0:
                    ui.notify("Unable to find a plan.")
                    return
                ui.label(f"Click to view a shot option (Ranked by score, #1~#{len(state["plans"])}).")
                plan_toggle = ui.toggle([f"#{i + 1}" for i in range(len(state["plans"]))], value="#1",
                                        on_change=lambda e: display_shot_plan(e.value))

                plan_label = ui.label()
                score_label = ui.label()
                score_breakdown_desc = ui.label().classes("whitespace-pre-line")
                heatmap_label = ui.label()
                position_map = ui.image().classes("w-full max-w-2xl")
                preview = ui.video(src=None, autoplay=True, loop=True, controls=True).classes("w-full max-w-2xl")
                cue_impact_diagram = ui.html()
                impact_label = ui.label()
                cue_speed_diagram = ui.interactive_image(size=(508, 108))
                speed_label = ui.label()

                async def display_shot_plan(idx: int | str):
                    if isinstance(idx, str):
                        idx = int(idx[1:])
                    if idx < 1 or idx > len(state["plans"]):
                        raise ValueError(f"Invalid shot index: {idx}. len(state['plans']) = {len(state['plans'])}")
                    if not plan_toggle.enabled:
                        return
                    plan_toggle.disable()
                    try:
                        ui.notify(f"Preparing plan #{idx}.")
                        result = state["plans"][idx - 1]
                        two_ply = isinstance(result, TwoPlyPlan)
                        plan = result.option if two_ply else result
                        plan_label.set_text(
                            f"Target ball: {plan.target_ball_id}. Target pocket: {plan.target_pocket_id} "
                            f"({plan.route_type}). Objective: {getattr(plan.objective, 'value', plan.objective)}"
                        )
                        score_label.set_text(
                            f"Combined: {result.combined_score:.3f} | Planner: {result.planner_score:.3f} | "
                            f"Execution: {result.execution_score:.3f} | Two-ply: {result.two_ply_value:.3f}"
                            if two_ply else f"Score: {plan.score:.3f}"
                        )
                        score_breakdown_desc.set_text("\n".join(
                            f"{name}: {value}" for name, value in plan.score_breakdown.items()
                        ))
                        for element in (preview, cue_impact_diagram, impact_label, cue_speed_diagram, speed_label):
                            element.set_visibility(False)

                        cached_plan = state["cached_plans"].get(idx)
                        if cached_plan is None:
                            output_dir = Path(__file__).parent / "frames" / f"{datetime.now():%Y%m%d_%H%M%S_%f}-plan-{idx}"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            image_path = None
                            if isinstance(plan.position_target, HeatmapTarget):
                                fig, ax = show_heatmap_target(plan.position_target, layout=state["planner_layout"], show=False, target_region_thres=0.0)
                                ax.invert_xaxis()
                                ax.invert_yaxis()
                                image_path = output_dir / "heatmap.png"
                                try:
                                    fig.savefig(image_path)
                                finally:
                                    plt.close(fig)
                            cached_plan = CachedPlan(id=idx, plan=plan, output_dir=output_dir, image_path=image_path)
                            state["cached_plans"][idx] = cached_plan

                        position_map.set_visibility(cached_plan.image_path is not None)
                        heatmap_label.set_text("Position heatmap" if cached_plan.image_path else "No position heatmap for this option.")
                        if cached_plan.image_path is not None:
                            position_map.set_source(cached_plan.image_path)

                        if cached_plan.video_path is None:
                            if two_ply:
                                shot = result.simulated_shot.copy()
                                phi, V0, a, b = result.policy.as_array()
                            else:
                                shot = create_shot(state["planner_layout"])
                                ui.notify(f"Searching execution policy for shot #{idx}...")
                                _, policies = await run.cpu_bound(
                                    executor_search,
                                    result.to_search_request(shot),
                                )
                                if len(policies) == 0:
                                    ui.notify("No executable policies found.")
                                    return
                                best_policy = policies[0]
                                phi, V0, a, b = best_policy.as_array()
                                shot.cue.set_state(V0=V0, phi=phi, a=a, b=b)
                                shot = await run.cpu_bound(
                                    pt.simulate,
                                    shot
                                )

                            cached_plan.video_path = await run.cpu_bound(
                                render_plan_video, shot, cached_plan.output_dir,
                            )
                        preview.set_source(cached_plan.video_path)
                        draw_cue_impact_diagram(cue_impact_diagram, impact_label, a, b)
                        draw_cue_speed_diagram(cue_speed_diagram, speed_label, V0)
                        for element in (preview, cue_impact_diagram, impact_label, cue_speed_diagram, speed_label):
                            element.set_visibility(True)
                        ui.notify(f"Displayed plan #{idx}.")
                    finally:
                        plan_toggle.enable()

                ui.timer(0.05, lambda: display_shot_plan(plan_toggle.value), once=True)

            ui.button("Run again", on_click=reset)

    render_controls()


ui.run(on_air=False)
