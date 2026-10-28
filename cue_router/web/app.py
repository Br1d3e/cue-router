from copy import deepcopy

import cv2
from nicegui import run, ui
import numpy as np
from pathlib import Path
from PIL import Image

from table_vision.vision import TableVision, TableLayout, Detection
from cue_router.core.table import ChineseTableSpecs

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

SELECTION_SEQUENCE = ["cue_ball", "target_ball", "pocket", "route_type", "other_balls", "target_region"]

def to_screen(x, y):
    return M + y / L * PLAY_WIDTH, M + x / W * PLAY_HEIGHT

def render_svg(layout: TableLayout):
    parts = [
        f'<rect width="{2 * M + PLAY_WIDTH}" height="{2 * M + PLAY_HEIGHT}" rx="15" fill="#674630"/>',
        f'<rect x="{M}" y="{M}" width="{PLAY_WIDTH}" height="{PLAY_HEIGHT}" fill="#176650"/>',
    ]

    for px, py in POCKETS_SCREEN.values():
        parts.append(
            f'<circle cx="{px}" cy="{py}" r="{R_POCKET}" fill="#101515"/>'
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

def render_ball_selector(content, x, y, del_button=False):
    """Replace the selector layer with a panel centered at image coordinates (x, y)."""
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
        "stage": "upload",
        "pocket_rects": {},
        "pending_corner": None,
        "selected_ball": None,
    }

    def next_stage(stage):
        if stage in ["upload", "manual_pockets"]:
            stage = "check_table"
        elif stage == "check_table":
            stage = SELECTION_SEQUENCE[0]
        elif stage in SELECTION_SEQUENCE:
            i = SELECTION_SEQUENCE.index(stage)
            if i == len(SELECTION_SEQUENCE):
                stage = "done"
            else:
                stage = SELECTION_SEQUENCE[i+1]

        state_msg.set_text("Stage: " + stage)
        # if selection_stage not in SELECTION_SEQUENCE:
        #     raise ValueError(f"Invalid selection stage: {selection_stage}")
        # i = SELECTION_SEQUENCE.index(selection_stage)
        # if i == len(SELECTION_SEQUENCE) - 1:
        #     ...
        # selection_stage = SELECTION_SEQUENCE[i]
        return stage

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
            state["stage"] = "manual_pockets"
            state_msg.set_text("Stage: Manually label pockets")
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
                    # state_msg.set_text("Check Table Layout")
                    table.set_content(render_svg(state["layout"]))
                    cache_balls()
                    return
                except RuntimeError as e:
                    ui.notify("Retry")
                    state["stage"] = "manual_pockets"

            if state["stage"] != "manual_pockets":
                return

            next_button = ui.button("Finish manual label", on_click=on_finish)

    ui.upload(
        on_upload=on_upload,
        auto_upload=True,
    ).props("accept=image/*")

    def on_table_click(e):
        ball_selector.set_content("")
        state["selected_ball"] = None
        if state["stage"] != "check_table":
            return
        x, y = e.image_x, e.image_y
        res = resolve_screen_pos(x, y, balls_screen=state["balls_screen"], unresolved_balls_screen=state["unresolved_balls_screen"])
        if res["type"] == "pocket" or not (M <= x <= M + PLAY_WIDTH and M <= y <= M + PLAY_HEIGHT):
            return
        state["selected_ball"] = (res, ((y - M) / PLAY_HEIGHT * W, (x - M) / PLAY_WIDTH * L))
        render_ball_selector(ball_selector, x, y + R_BALL + 60, del_button=res["type"] != "table")

    def on_ball_select(e):
        if state["stage"] != "check_table" or state["selected_ball"] is None:
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
        if state["stage"] != "manual_pockets":
            return
        x, y = e.image_x, e.image_y

        def redraw_pockets():
            svg = ""

            for x1, y1, x2, y2 in state["pocket_rects"].values():
                svg += (
                    f'<rect x="{x1}" y="{y1}" '
                    f'width="{x2 - x1}" height="{y2 - y1}" '
                    f'fill="none" stroke="red" stroke-width="3" '
                    f'pointer-events="none" />'
                )

            if state["pending_corner"] is not None:
                x, y = state["pending_corner"]
                svg += f'<circle cx="{x}" cy="{y}" r="5" fill="yellow" pointer-events="none" />'

            captured_image.set_content(svg)

        for rect_id, (x1, y1, x2, y2) in list(state["pocket_rects"].items()):
            if x1 <= x <= x2 and y1 <= y <= y2:
                del state["pocket_rects"][rect_id]
                state["pending_corner"] = None
                redraw_pockets()
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

        redraw_pockets()

    def cache_balls():
        state["balls_screen"].clear()
        state["unresolved_balls_screen"].clear()
        if state["layout"] is not None:
            for ball_id, (x, y) in state["layout"].balls_info.items():
                state["balls_screen"][ball_id] = to_screen(x, y)

            for ball_id, (x, y) in state["layout"].unresolved_balls_info.items():
                state["unresolved_balls_screen"][ball_id] = to_screen(x, y)

    state_msg = ui.label(f"Stage: {state["stage"]}")

    captured_image = ui.interactive_image(
        source=state["captured_image"],
        on_mouse=on_image_click,
    ).classes("max-w-2xl")

    table = ui.interactive_image(
        size=(840, 440),
        on_mouse=on_table_click,
    ).classes("w-full max-w-2xl")
    ball_selector = table.add_layer()
    ball_selector.on("svg:pointerup", on_ball_select)

    mode = ui.toggle(["Executor", "Planner"], value="Executor")

ui.run()
