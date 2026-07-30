from ultralytics import YOLO
from pathlib import Path

import torch
import numpy as np
import cv2

from dataclasses import dataclass, field

folder = Path(__file__).resolve().parent
MODEL_PATH = folder / "models" / "yolo_3.pt"

CLASS_MAPPING = {
    0: "black ball 8",
    1: "blue ball 2",
    2: "blue ball - t 10",
    3: "brown ball 7",
    4: "brown ball - t 15",
    5: "green ball 6",
    6: "green ball - t 14",
    7: "orange ball 5",
    8: "orange ball - t 13",
    9: "pink ball 4",
    10: "pink ball - t 12",
    11: "pocket",
    12: "red ball 3",
    13: "red ball - t 11",
    14: "table",
    15: "triangle rack",
    16: "white ball",
    17: "yellow ball 1",
    18: "yellow ball - t 9",
}

BALL_CLASSES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 13, 16, 17, 18,
]
POCKET_ID = 11

BALL_OVERLAP_IOU = 0.80
BALL_OVERLAP_CENTER_DIST = 0.10
POCKET_OVERLAP_IOU = 0.30
BALL_CLASS_DUPE_CONF = 0.6
BALL_CLASS_DUPE_CONF_DIFF = 0.3
POCKET_MIN_CONF = 0.7

BALL_CLASS_TO_ID = {
    0: "8",
    1: "2",
    2: "10",
    3: "7",
    4: "15",
    5: "6",
    6: "14",
    7: "5",
    8: "13",
    9: "4",
    10: "12",
    12: "3",
    13: "11",
    16: "cue",
    17: "1",
    18: "9"
}


@dataclass
class Detection:
    id: int
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2, (y1 + y2) / 2

    def __repr__(self):
        return (f"Detection(\n"
                f"   id={self.id}\n"
                f"   class_id={self.class_id}\n"
                f"   xyxy={self.xyxy}\n"
                f"   confidence={self.confidence:.3f}\n"
                f")")


@dataclass
class ImageState:
    image: np.ndarray

    balls: list[Detection]
    ball_classes: set[int]
    pockets: list[Detection]
    table: list[Detection] | None = None
    unresolved_balls: list[Detection] = field(default_factory=list)
    unresolved_pockets: list[Detection] = field(default_factory=list)
    unused_ball_classes: set[int] = field(default_factory=set)

    resolved: bool = False

    def __repr__(self):
        return (f"ImageState(\n"
                f"  balls={self.balls},\n"
                f"  ball_classes={self.ball_classes},\n\n"
                f"  pockets={self.pockets},\n\n"
                f"  table={self.table}\n\n"
                f"  unresolved_balls={self.unresolved_balls},\n"
                f"  unresolved_pockets={self.unresolved_pockets},\n"
                f"  resolved={self.resolved}\n"
                f")")

@dataclass
class TableLayout:
    image_state: ImageState

    balls_info: dict[str, tuple[float, float]]
    ball_ids: set[str] = field(default_factory=set)
    unused_ball_ids: set[str] = field(default_factory=set)


class TableVision:
    def __init__(self, image: np.ndarray, model_path=None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model = YOLO(model_path)
        self.image = image
        self.image_state = None

    def predict(self, image=None):
        if image is None:
            image = self.image

        results = self.model.predict(
            source=image,
            imgsz=960,
            conf=0.5,
            iou=0.5,
            save=False,
            verbose=True,
        )

        ball_detections = []
        ball_classes = set()
        pocket_detections = []

        for result in results:
            for i, box in enumerate(result.boxes):
                cls_id = int(box.cls.item())
                cls_name = CLASS_MAPPING[cls_id]
                conf = float(box.conf.item())
                xyxy = tuple(map(float, box.xyxy[0]))

                detection = Detection(i, cls_id, cls_name, xyxy, conf)

                if cls_id in BALL_CLASSES:
                    ball_detections.append(detection)
                    ball_classes.add(cls_id)
                elif cls_id == POCKET_ID:
                    pocket_detections.append(detection)
                else:
                    print(f"{cls_name} detected, conf={conf}")

        all_detections = ball_detections + pocket_detections
        self.image_state = ImageState(
            image=image,
            balls=ball_detections,
            ball_classes=ball_classes,
            pockets=pocket_detections,
            unused_ball_classes=set(BALL_CLASSES) - ball_classes,
        )
        return self.image_state

    @staticmethod
    def iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]):
        x_min_1, y_min_1, x_max_1, y_max_1 = box1
        x_min_2, y_min_2, x_max_2, y_max_2 = box2

        left = max(x_min_1, x_min_2)
        top = max(y_min_1, y_min_2)
        right = min(x_max_1, x_max_2)
        bottom = min(y_max_1, y_max_2)
        intersection = max(0, right - left) * max(0, bottom - top)
        area_1 = max(0, x_max_1 - x_min_1) * max(0, y_max_1 - y_min_1)
        area_2 = max(0, x_max_2 - x_min_2) * max(0, y_max_2 - y_min_2)
        union = area_1 + area_2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    def remove_ball_overlaps(self, image_state: ImageState = None):
        if image_state is None:
            image_state = self.image_state
        detections = image_state.balls

        kept_indices = []
        confidence_order = sorted(
            range(len(detections)),
            key=lambda index: detections[index].confidence,
            reverse=True,
        )
        for i in confidence_order:
            detection = detections[i]
            overlaps_kept_ball = False
            for kept_index in kept_indices:
                kept_detection = detections[kept_index]
                iou = self.iou(detection.xyxy, kept_detection.xyxy)
                center_distance = np.linalg.norm(
                    np.asarray(detection.center) - np.asarray(kept_detection.center)
                )
                x1, y1, x2, y2 = detection.xyxy
                kept_x1, kept_y1, kept_x2, kept_y2 = kept_detection.xyxy
                center_scale = min(
                    x2 - x1,
                    y2 - y1,
                    kept_x2 - kept_x1,
                    kept_y2 - kept_y1,
                )
                if center_scale <= 0:
                    continue

                if (
                        iou >= BALL_OVERLAP_IOU
                        and center_distance / center_scale < BALL_OVERLAP_CENTER_DIST
                ):
                    overlaps_kept_ball = True
                    break

            if not overlaps_kept_ball:
                kept_indices.append(i)

        kept_indices = set(kept_indices)
        image_state.balls = [
            detection
            for index, detection in enumerate(detections)
            if index in kept_indices
        ]
        image_state.ball_classes = set([ball.class_id for ball in image_state.balls])
        return image_state

    def remove_pocket_overlaps(self, image_state: ImageState = None):
        if image_state is None:
            image_state = self.image_state
        detections = image_state.pockets

        kept_indices = []
        confidence_order = sorted(
            range(len(detections)),
            key=lambda index: detections[index].confidence,
            reverse=True,
        )
        for index in confidence_order:
            if all(
                    self.iou(detections[index].xyxy, detections[kept_index].xyxy)
                    < POCKET_OVERLAP_IOU
                    for kept_index in kept_indices
            ):
                kept_indices.append(index)

        kept_indices = set(kept_indices)
        image_state.pockets = [
            detection
            for index, detection in enumerate(detections)
            if index in kept_indices
        ]
        return image_state

    def remove_ball_cls_dupes(self, image_state: ImageState = None):
        if image_state is None:
            image_state = self.image_state

        detections = image_state.balls
        indices_by_class = {}
        for index, ball in enumerate(detections):
            indices_by_class.setdefault(ball.class_id, []).append(index)

        kept_indices = set()
        unresolved_balls = []
        for class_indices in indices_by_class.values():
            confidence_order = sorted(
                class_indices,
                key=lambda index: detections[index].confidence,
                reverse=True,
            )
            winner_index = confidence_order[0]
            kept_indices.add(winner_index)

            if len(confidence_order) == 1:
                continue

            winner = detections[winner_index]
            runner_up = detections[confidence_order[1]]
            if (
                    winner.confidence >= BALL_CLASS_DUPE_CONF
                    and winner.confidence - runner_up.confidence >= BALL_CLASS_DUPE_CONF_DIFF
            ):
                continue

            unresolved_balls.extend(
                detections[index]
                for index in confidence_order[1:]
            )

        image_state.balls = [
            detection
            for index, detection in enumerate(detections)
            if index in kept_indices
        ]
        image_state.ball_classes = set([ball.class_id for ball in image_state.balls])
        image_state.unresolved_balls.extend(unresolved_balls)
        return image_state

    def remove_pocket_dupes(self, image_state: ImageState = None):
        """
        For now, use top k (k ≤ 6) thresholded pocket confidences to eliminate duplicates
        """
        if image_state is None:
            image_state = self.image_state

        detections = image_state.pockets

        confidence_order = sorted(
            range(len(detections)),
            key=lambda index: detections[index].confidence,
            reverse=True,
        )

        kept_indices = []
        for i in confidence_order:
            detection = detections[i]
            if detection.confidence < POCKET_MIN_CONF or i >= 6:
                image_state.unresolved_pockets.append(detection)
                continue
            kept_indices.append(i)

        image_state.pockets = [
            detection for i, detection in enumerate(detections)
            if i in kept_indices
        ]

        return image_state

    def resolve_image_state(self):
        """
        NMS Postprocessing after prediction
        - Remove overlap/duplicate balls
        - Remove overlap/duplicate pockets
        :return: ImageState(resolved=True)
        """
        if self.image_state is None:
            raise ValueError("No ImageState found.")

        if self.image_state.resolved:
            return self.image_state

        self.remove_ball_overlaps()
        self.remove_pocket_overlaps()
        self.remove_ball_cls_dupes()
        self.remove_pocket_dupes()

        self.image_state.table = self.image_state.balls + self.image_state.pockets
        self.image_state.resolved = True

        return self.image_state


if __name__ == "__main__":
    test_img_path = Path(__file__).parent / "tests" / "test7.jpg"
    image = cv2.imread(str(test_img_path))

    table_vision = TableVision(image)
    table_vision.predict()
    print(table_vision.image_state, end="\n\n")
    img_state = table_vision.resolve_image_state()
    print(img_state)
