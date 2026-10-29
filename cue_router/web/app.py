from copy import deepcopy

import cv2
from nicegui import run, ui
import numpy as np
from pathlib import Path
from PIL import Image
from enum import Enum, auto
from dataclasses import dataclass

from nicegui.events import ValueChangeEventArguments

from cue_router.planner.planner_search import PlannerSearch
from cue_router.ui.layout import ExecutorLayout, PlannerLayout

from table_vision.vision import TableVision, TableLayout, Detection
from cue_router.core.table import ChineseTableSpecs
from cue_router.core.geometry import LineSegment, Rect, PositionTarget

MODEL_PATH = Path(__file__).parent.parent.parent / "table_vision" / "models" / "yolo_3.pt"

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

class Stage(Enum):
    UPLOAD = auto()
    MANUAL_POCKETS = auto()
    REVIEW = auto()
    SELECT = auto()
    EXECUTE = auto()

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

def to_screen(x, y):
    return M + y / L * PLAY_WIDTH, M + x / W * PLAY_HEIGHT

def in_table(x, y, allow_margin=False):
    if allow_margin:
        return 0 <= x <= PLAY_WIDTH + 2 * M and 0 <= y <= PLAY_HEIGHT + 2 * M
    return M <= x <= M + PLAY_WIDTH and M <= y <= M + PLAY_HEIGHT

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
        f'width="{x2-x1}" height="{y2-y1}" '
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

def render_ball_selector(content, x, y, del_button=False):
    cell = 44
    padding = 8
    width = 8 * cell + 2 * padding + (cell if del_button else 0)
    height = 2 * cell + 2 * padding
    left = max(0, min(x - width / 2, PLAY_WIDTH + 2 * M - width))
    top = max(0, min(y - height / 2, PLAY_HEIGHT + 2 * M - height))
    parts = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" '
        f'rx="10" fill="#f8fafc" stroke="#64748b" pointer-events="all"/>',
    ]
    for i, (ball_id, (stroke, fill)) in enumerate(BALL_COLORS.items()):
        cx = left + padding + (i % 8 + 0.5) * cell
        cy = top + padding + (i // 8 + 0.5) * cell
        label = "" if ball_id == "cue" else ball_id
        parts.extend([
            f'<circle cx="{cx}" cy="{cy}" r="17" fill="{fill}" '
            f'stroke="{stroke if ball_id != "cue" else "#94a3b8"}" stroke-width="4"/>',
            f'<circle cx="{cx}" cy="{cy}" r="10" fill="white"/>',
            f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="13" font-family="sans-serif" fill="black">{label}</text>',
            f'<rect id="select-{ball_id}" x="{cx - cell / 2}" y="{cy - cell / 2}" '
            f'width="{cell}" height="{cell}" fill="transparent" pointer-events="all" '
            f'cursor="pointer"><title>{"Cue ball" if ball_id == "cue" else "Ball " + ball_id}</title></rect>',
        ])
    if del_button:
        cx = left + padding + 8.5 * cell
        cy = top + height / 2
        parts.extend([
            f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
            f'font-size="12" font-family="sans-serif" fill="#b91c1c">Delete</text>',
            f'<rect id="delete-ball" x="{cx - cell / 2}" y="{top + padding}" '
            f'width="{cell}" height="{2 * cell}" fill="transparent" '
            f'pointer-events="all" cursor="pointer"><title>Delete ball</title></rect>',
        ])
    content.set_content("".join(parts))

def render_circle_highlight(content, x, y, r):
    highlight = f'<circle cx="{x}" cy="{y}" r="{r+2}" fill="purple" opacity="0.7" stroke="black" stroke_width="4" stroke-dasharray="3" />'
    content.set_content(content.content + highlight)

@ui.page("/")
def page():
        ui.page_title("CueRouter Demo")
        ui.label("CueRouter Demo")

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
        }

        def next_stage(stage: Stage, next_state_func=None):
            stage = deepcopy(stage)
            if stage in [Stage.UPLOAD, Stage.MANUAL_POCKETS]:
                stage = Stage.REVIEW
            elif stage == Stage.REVIEW:
                if "cue" not in state["balls_screen"].keys():
                    ui.notify("[⚠ Warning] No cue ball selected.")
                state["unresolved_balls_screen"] = None
                state["layout"].unresolved_balls_info.clear()
                stage = Stage.SELECT
            elif stage == Stage.SELECT:
                stage = Stage.EXECUTE

            state["stage"] = stage
            state_msg.set_text(f"Stage: {stage}")
            render_controls.refresh()

            if callable(next_state_func):
                next_state_func()
            return stage

        def next_selection(selection):
            # i = EXECUTOR_SELECTION_SEQUENCE.index(selection)
            # if i == len(EXECUTOR_SELECTION_SEQUENCE) - 1:
            #     selection = None
            # else:
            #     selection = EXECUTOR_SELECTION_SEQUENCE[i + 1]

            if selection == "target_ball":
                selection = "pocket"
                selection_msg.set_text("Select a target pocket.")
            elif selection == "pocket":
                selection = "target_region"
                selection_msg.set_text("Choose a target region type first. Then click the opposite corners of the region.")
                ui.button("Line Segment",
                          on_click=lambda f: update_executor_selection("target_region_type", LineSegment))
                ui.button("Rectangle", on_click=lambda f: update_executor_selection("target_region_type", Rect))
            # elif selection == "target_region":
            #     ...

            state["selection"] = selection

            return selection

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
                cache_balls()
                state["stage"] = next_stage(state["stage"])
                table.set_content(render_svg(state["layout"]))
            except RuntimeError as e:
                if str(e) == "NOT_ENOUGH_POCKETS":
                    ui.notify("Automatic pocket calibration requires ≥ 5 pockets. Label the pockets or retake an image.")
                elif str(e) == "POCKET_IDENTITIES_NOT_FOUND":
                    ui.notify("Could not determine pocket identities for homography calibration. Correct the pocket labels or retake an image.")
                image_state = vision.image_state
                state["stage"] = Stage.MANUAL_POCKETS
                state_msg.set_text(f"Stage: {state["stage"]}")
                for i, detection in enumerate(image_state.pockets):
                    x1, y1, x2, y2 = detection.xyxy
                    render_rect(captured_image, x1, y1, x2, y2)
                    state["pocket_rects"][i] = detection.xyxy

                def on_finish():
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
                        state["stage"] = next_stage(state["stage"])
                        ui.notify("Layout generation successful.")
                        table.set_content(render_svg(state["layout"]))
                        cache_balls()
                        return
                    except RuntimeError as e:
                        ui.notify("Retry")
                        state["stage"] = Stage.MANUAL_POCKETS

                if state["stage"] != Stage.MANUAL_POCKETS:
                    return

                ui.button("Finish manual label", on_click=on_finish)

        ui.upload(
            on_upload=on_upload,
            auto_upload=True,
        ).props("accept=image/*")


        def on_table_click(e):
            ball_selector.set_content("")
            state["selected_ball"] = None
            if state["stage"] not in [Stage.REVIEW, Stage.SELECT]:
                return
            x, y = e.image_x, e.image_y
            res = resolve_screen_pos(x, y, balls_screen=state["balls_screen"], unresolved_balls_screen=state["unresolved_balls_screen"])

            if state["stage"] == Stage.REVIEW:
                if res["type"] == "pocket" or not in_table(x, y):
                    return
                state["selected_ball"] = (res, ((y - M) / PLAY_HEIGHT * W, (x - M) / PLAY_WIDTH * L))
                render_ball_selector(ball_selector, x, y + R_BALL + 60, del_button=res["type"] != "table")
            elif state["stage"] == Stage.SELECT:
                if not in_table(x, y, allow_margin=True):
                    return
                if state["selection"] == "target_ball":
                    if res["type"] == "ball":
                        if res["id"] == "cue":
                            selection_msg.set_text("Cannot select the cue ball. Choose a colored ball.")
                            return
                        screen_x, screen_y = state["balls_screen"][res["id"]]
                        if state["executor_selection"].target_ball_id == res["id"]:
                            selection_msg.set_text(f"Confirmed selecting target ball #{res["id"]}")
                            next_selection(state["selection"])
                            render_circle_highlight(ball_selector, screen_x, screen_y, R_BALL)
                        else:
                            ball_selector.set_content("")
                            state["executor_selection"].target_ball_id = res["id"]
                            selection_msg.set_text(f"Selected target ball #{res['id']}. Click it again to confirm.")
                            render_circle_highlight(ball_selector, screen_x, screen_y, R_BALL)
                    else:
                        state["executor_selection"].target_ball_id = None
                        ball_selector.set_content("")
                        selection_msg.set_text(f"Select a target ball to continue.")
                elif state["selection"] == "pocket":
                    if res["type"] == "pocket":
                        screen_x, screen_y = POCKETS_SCREEN[res["id"]]
                        if state["executor_selection"].target_pocket_id == res["id"]:
                            selection_msg.set_text(f"Confirmed selecting pocket {res['id']}.")
                            next_selection(state["selection"])
                            render_circle_highlight(ball_selector, screen_x, screen_y, R_POCKET)
                        else:
                            ball_selector.set_content("")
                            state["executor_selection"].target_pocket_id = res["id"]
                            selection_msg.set_text(f"Selected pocket {res['id']}. Click it again to confirm.")
                            render_circle_highlight(ball_selector, screen_x, screen_y, R_POCKET)
                    else:
                        state["executor_selection"].target_pocket_id = None
                        ball_selector.set_content("")
                        selection_msg.set_text(f"Select a target pocket to continue.")
                elif state["selection"] == "target_region":
                    selection = state["executor_selection"]
                    if selection.target_region_type is None:
                        selection_msg.set_text("Choose a position target type: Line or Rectangle. Then click two endpoints or opposite corners.")
                        return
                    if selection.pending_corner is None:
                        selection.pending_corner = x, y
                        selection.target_region = None
                        redraw_regions(ball_selector, [], selection.pending_corner, selection.target_region_type)
                    else:
                        x1, y1 = selection.pending_corner
                        x2, y2 = x, y
                        selection.target_region = selection.target_region_type(x1, y1, x2, y2)
                        selection.pending_corner = None
                        redraw_regions(ball_selector, [(x1, y1, x2, y2)], region_type=selection.target_region_type)
                        selection_msg.set_text("Selected target region. Click Next to execute.")
                        state["executor_layout"] = build_executor_layout(state["executor_selection"])
                        ui.button("Next", on_click=lambda f: next_stage(state["stage"]))

        def on_ball_select(e):
            if state["stage"] != Stage.REVIEW or state["selected_ball"] is None:
                return
            element_id = e.args["element_id"]
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
            elif state["mode"] == "Executor":
                state["executor_selection"] = ExecutorSelection()
                state["selection"] = EXECUTOR_SELECTION_SEQUENCE[0]
                selection_msg.set_text("Executor selection started. Pick a target ball first.")

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
                route_type=state["executor_selection"].route_type,
            )

        def execute():
            if state["stage"] != Stage.EXECUTE:
                return
            if state["mode"] == "Planner":
                ...
            elif state["mode"] == "Executor":
                ...

        state_msg = ui.label(f"Stage: {state["stage"]}")
        selection_msg = ui.label("")

        # captured_image = ui.interactive_image(
        #     source=state["captured_image"],
        #     on_mouse=on_image_click,
        # ).classes("max-w-2xl")

        table = ui.interactive_image(
            size=(840, 440),
            on_mouse=on_table_click,
        ).classes("w-full max-w-2xl")
        ball_selector = table.add_layer()
        ball_selector.on("svg:pointerup", on_ball_select)

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
                ui.button("Next", on_click=lambda f: next_stage(state["stage"]))
            elif state["stage"] == Stage.REVIEW:
                captured_image = ui.interactive_image(
                    source=state["captured_image"],
                    on_mouse=on_image_click,
                ).classes("max-w-2xl")
                ui.button("Next", on_click=lambda f: next_stage(state["stage"], next_state_func=select))
                ui.toggle(["Executor", "Planner"], value="Executor", on_change=lambda e: update_state_value("mode", e.value))
            elif state["stage"] == Stage.SELECT:
                # def set_executor_route_type(value):
                #     state["executor_selection"]["route_type"] = value
                ui.toggle(
                    ["Direct", "Bank"], value="Direct",
                    on_change=lambda e: update_executor_selection("route_type", e.value)
                )
            else:
                ...

        render_controls()

ui.run()
