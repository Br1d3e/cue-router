from typing import Any

from numpy import dtype, ndarray
from ultralytics import YOLO
from pathlib import Path

import torch
import numpy as np
import cv2

from dataclasses import dataclass, field

from itertools import permutations, combinations

from cue_router.core.table import ChineseTableSpecs

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

VALID_RAIL_POCKETS_THRESHOLD = 0.10

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

L = ChineseTableSpecs.TABLE_LENGTH
W = ChineseTableSpecs.TABLE_WIDTH
CANONICAL_POCKETS = {
    "lb": (0, 0),
    "lc": (0, L / 2),
    "lt": (0, L),
    "rb": (W, 0),
    "rc": (W, L / 2),
    "rt": (W, L),
}


@dataclass
class Detection:
    id: int
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float

    @property
    def center(self):
        x1, y1, x2, y2 = self.xyxy
        return np.asarray(((x1 + x2) / 2, (y1 + y2) / 2), dtype=float)

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

    def __repr__(self):
        return (f"TableLayout(\n"
                f"  image_state={self.image_state},\n"
                f"  balls_info={self.balls_info},\n"
                f"  ball_ids={self.ball_ids},\n"
                f"  unused_ball_ids={self.unused_ball_ids},\n"
                )

    def to_planner_layout(self):
        ...

    def to_executor_layout(self, target_ball, target_pocket, target_region):
        ...


class TableVision:
    def __init__(self, image: np.ndarray | None = None, model_path=None, homography=None):
        if model_path is None:
            model_path = MODEL_PATH
        self.model = YOLO(model_path)
        self.image = image
        self.image_state = None

        self.H = None
        if homography is not None:
            self.set_homography(homography)

    def set_homography(self, homography):
        homography = np.asarray(homography, dtype=float)
        if homography.shape != (3, 3):
            raise ValueError(f"Homography must have shape (3, 3), got {homography.shape}.")
        if not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
            raise ValueError("Homography must be finite and invertible.")
        self.H = homography.copy()

    def predict(self, image=None):
        if image is None:
            image = self.image
        if image is None:
            raise ValueError("No image provided.")
        self.image = image

        results = self.model.predict(
            source=image,
            imgsz=960,
            conf=0.5,
            iou=0.5,
            save=False,
            verbose=False,
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

        self.image_state = ImageState(
            image=image,
            balls=ball_detections,
            ball_classes=ball_classes,
            pockets=pocket_detections,
            unused_ball_classes=set(BALL_CLASSES) - ball_classes,
        )
        return self.image_state

    def predict_state(self, image=None) -> ImageState:
        """Predict and postprocess one image."""
        self.predict(image)
        return self.resolve_image_state()

    def predict_layout(self, image=None, *, recalibrate=False, preview=False) -> TableLayout:
        """Predict one image and return physical table coordinates."""
        image_state = self.predict_state(image)
        if recalibrate or self.H is None:
            self.calibrate_pocket_homography(image_state)
        return self.transform_table_homography(image_state, preview=preview)

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


    def order_3_pockets(self, pockets):
        if len(pockets) != 3:
            raise ValueError(f"Instance of pockets must be 3, got {len(pockets)}.")

        pairs = list(combinations(pockets, 2))
        i, j = max(pairs, key=lambda pair: np.linalg.norm(pair[0].center - pair[1].center))
        k = next(pocket for pocket in pockets if pocket is not i and pocket is not j)
        return [i, j, k]

    def perp_deviation(self, a, b, m):
        v = b - a
        w = m - a
        perp_dist = abs(np.cross(v, w)) / np.linalg.norm(v)

        return perp_dist / np.linalg.norm(v)

    def m_projection_err(self, a, b, m, boundary=0.2):
        v = b - a
        w = m - a
        projection = np.dot(v, w) / (np.linalg.norm(v) ** 2)

        if 0 + boundary <= projection <= 1 - boundary:
            return 0.0
        else:
            return 1.0

    def rail_pockets_score(self, pockets: tuple[Detection, Detection, Detection]):
        if len(pockets) < 3:
            return 0.0
        ordered = self.order_3_pockets(pockets)
        a = np.asarray(ordered[0].center, dtype=float)
        b = np.asarray(ordered[1].center, dtype=float)
        m = np.asarray(ordered[2].center, dtype=float)

        return self.perp_deviation(a, b, m) + self.m_projection_err(a, b, m)

    def pca_line_error(self, pockets, eps=0.1):
        points = [pocket.center for pocket in pockets]
        points = np.asarray(points, dtype=float)

        center = points.mean(axis=0)
        centered = points - center
        covariance = centered.T @ centered / points.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        lambda_min = eigenvalues[0]
        lambda_max = eigenvalues[1]

        if lambda_max < 1e-9:
            return float("inf")

        return np.sqrt(lambda_min / (lambda_max + eps))

    def canonical_rail_pockets(self, two_rails: tuple[list, list]):
        r1, r2 = two_rails

        def sort_rail(rail):
            x_range = max(pocket.center[0] for pocket in rail) - min(pocket.center[0] for pocket in rail)
            y_range = max(pocket.center[1] for pocket in rail) - min(pocket.center[1] for pocket in rail)
            axis = 0 if x_range > y_range else 1
            return sorted(rail, key=lambda pocket: pocket.center[axis], reverse=True)

        r1 = sort_rail(r1)
        r2 = sort_rail(r2)

        return {
            "lb": r1[0],
            "lc": r1[1],
            "lt": r1[2],
            "rb": r2[0],
            "rc": r2[1],
            "rt": r2[2],
        }

    def find_pocket_identities(self, image_state: ImageState = None):
        if image_state is None:
            image_state = self.image_state

        pockets = image_state.pockets

        if len(pockets) != 6:
            return [], []

        rails = []
        scores = []
        complements = []
        for s1 in combinations(pockets, 3):
            s2 = tuple([p for p in pockets if p not in s1])
            complements.append(s2)
            if s1 in complements:
                continue

            score_1 = self.rail_pockets_score(s1)
            score_2 = self.rail_pockets_score(s2)
            # score_1 = self.pca_line_error(s1, eps=0.2)
            # score_2 = self.pca_line_error(s2, eps=0.2)

            if score_1 < VALID_RAIL_POCKETS_THRESHOLD and score_2 < VALID_RAIL_POCKETS_THRESHOLD:
                rails.append((s1, s2))
                scores.append((score_1, score_2))

        identities = []
        for pair in rails:
            identities.append(self.canonical_rail_pockets(pair))

        return identities, scores

    def calibrate_pocket_homography(self, image_state: ImageState = None):
        if image_state is None:
            image_state = self.image_state
        if image_state is None:
            raise ValueError("No ImageState found.")
        if len(image_state.pockets) != 6:
            raise RuntimeError(
                "Automatic homography calibration requires 6 resolved pockets, "
                f"got {len(image_state.pockets)}."
            )
        pocket_identities, scores = self.find_pocket_identities(image_state)

        if not pocket_identities:
            raise RuntimeError("Could not determine pocket identities for homography calibration.")

        best_idx = min(range(len(scores)), key=lambda i: sum(scores[i]))
        pockets = pocket_identities[best_idx]
        pocket_centers = [pocket.center for pocket in pockets.values()]
        src_points = np.asarray(pocket_centers, dtype=float)
        dst_points = np.asarray(list(CANONICAL_POCKETS.values()), dtype=float)

        H, mask = cv2.findHomography(src_points, dst_points, method=0)

        if H is None:
            raise RuntimeError("Homography estimation failed")

        self.set_homography(H)
        return self.H, mask

    def transform_table_homography(self, image_state=None, H=None, preview=False) -> TableLayout:
        if image_state is None:
            image_state = self.image_state
        if H is None:
            H = self.H
        if image_state is None:
            raise ValueError("No ImageState found.")
        if H is None:
            raise RuntimeError("No homography available. Calibrate or set one first.")

        img_balls = image_state.balls
        img_balls_pos = [ball.center for ball in img_balls]
        ball_pos_cv = np.asarray(img_balls_pos, dtype=float).reshape(-1, 1, 2)

        mapped = cv2.perspectiveTransform(ball_pos_cv, H).reshape(-1, 2)

        if preview:
            scale = 500
            width_px = round(W * scale)
            height_px = round(L * scale)

            preview_scale = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ])

            preview_H = preview_scale @ H

            warped_img = cv2.warpPerspective(
                image_state.image,
                preview_H,
                (width_px, height_px),
            )

            cv2.imshow('Original', image_state.image)
            cv2.imshow('Warped Output', warped_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return self.build_table_layout(image_state, mapped)


    def build_table_layout(self, image_state, mapped_balls: np.ndarray) -> TableLayout:
        img_balls = image_state.balls
        ball_classes = [ball.class_id for ball in img_balls]

        balls_info = {}
        ball_ids = set()
        for i, pos in enumerate(mapped_balls):
            cls = ball_classes[i]
            ball_id = BALL_CLASS_TO_ID[cls]
            balls_info[ball_id] = tuple(pos)
            ball_ids.add(ball_id)
        unused_ball_ids = set(BALL_CLASS_TO_ID.values()) - ball_ids

        return TableLayout(image_state, balls_info, ball_ids, unused_ball_ids)



if __name__ == "__main__":
    test_img_path = Path(__file__).parent / "tests" / "test7.jpg"
    image = cv2.imread(str(test_img_path))

    table_vision = TableVision()
    print(table_vision.predict_layout(image, preview=True))
