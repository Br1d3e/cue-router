from ultralytics import YOLO
from pathlib import Path

import numpy as np
import cv2

from dataclasses import dataclass, field

from itertools import combinations

from cue_router.core.table import ChineseTableSpecs
from cue_router.ui.layout import PlannerLayout, ExecutorLayout

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
POCKET_OVERLAP_IOU = 0.2
BALL_CLASS_DUPE_CONF = 0.6
BALL_CLASS_DUPE_CONF_DIFF = 0.3
POCKET_MIN_CONF = 0.1

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
R = ChineseTableSpecs.BALL_RADIUS
CANONICAL_POCKETS = {
    pocket_id: tuple(ChineseTableSpecs.chinese_table.pockets[pocket_id].center[:2])
    for pocket_id in ChineseTableSpecs.POCKET_ID
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
        return (
            f"Detection(\n"
            f"   id={self.id}\n"
            f"   class_id={self.class_id}\n"
            f"   class_name={self.class_name}\n"
            f"   xyxy={self.xyxy}\n"
            f"   confidence={self.confidence:.3f}\n"
            f")"
        )


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
        return (
            f"ImageState(\n"
            f"  balls={self.balls},\n"
            f"  ball_classes={self.ball_classes},\n\n"
            f"  pockets={self.pockets},\n\n"
            f"  table={self.table}\n\n"
            f"  unresolved_balls={self.unresolved_balls},\n"
            f"  unresolved_pockets={self.unresolved_pockets},\n"
            f"  resolved={self.resolved}\n"
            f")"
        )


@dataclass
class TableLayout:
    image_state: ImageState

    balls_info: dict[str, tuple[float, float]]
    ball_ids: set[str] = field(default_factory=set)
    unused_ball_ids: set[str] = field(default_factory=set)
    unresolved_balls_info: dict[str, tuple[float, float]] = field(default_factory=dict)

    orientation: list[str] | str | None = None

    def __repr__(self):
        return (
            f"TableLayout(\n"
            f"  image_state={self.image_state},\n"
            f"  balls_info={self.balls_info},\n"
            f"  ball_ids={self.ball_ids},\n"
            f"  unused_ball_ids={self.unused_ball_ids},"
            f")\n"
        )

    def to_planner_layout(self):
        cue_ball_pos = self.balls_info["cue"]
        other_balls_info = self.balls_info.copy()
        other_balls_info.pop("cue")
        return PlannerLayout(
            cue_ball_pos=cue_ball_pos,
            balls_info=other_balls_info,
        )

    def flip_horizontal(self, width=W):
        for ball_id, pos in self.balls_info.items():
            x, y = pos
            self.balls_info[ball_id] = (width - x, y)
        if self.orientation is None:
            self.orientation = ["horizontal_180"]
        elif "horizontal_180" in self.orientation:
            self.orientation.remove("horizontal_180")
        else:
            self.orientation.append("horizontal_180")
        return self

    def flip_vertical(self, length=L):
        for ball_id, pos in self.balls_info.items():
            x, y = pos
            self.balls_info[ball_id] = (x, length - y)
        if self.orientation is None:
            self.orientation = ["vertical_180"]
        elif "horizontal_180" in self.orientation:
            self.orientation.remove("vertical_180")
        else:
            self.orientation.append("vertical_180")
        return self

    def full_unresolved_layout(self):
        image_state = self.image_state
        balls_info = self.balls_info.copy()
        ball_ids = self.ball_ids.copy()

        for ball_id, pos in self.unresolved_balls_info.items():
            balls_info[ball_id] = pos
            ball_ids.add(ball_id)
        unused_ball_ids = set(BALL_CLASS_TO_ID.values()) - ball_ids
        return TableLayout(image_state, balls_info, ball_ids, unused_ball_ids)


class TableVision:
    def __init__(self, image: np.ndarray | None = None, model_path=None, homography=None, ball_offset_alpha=0.25):
        if model_path is None:
            model_path = MODEL_PATH
        self.model = YOLO(model_path)
        self.image = image
        self.image_state = None

        self.H = None
        if homography is not None:
            self.set_homography(homography)
        self.ball_offset_alpha = ball_offset_alpha

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
            conf=0.1,
            iou=0.3,
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

    def predict_layout(self, image=None, *, image_state=None, recalibrate=False, preview=False) -> TableLayout:
        """Predict one image and return physical table coordinates."""
        if image_state is None or not isinstance(image_state, ImageState):
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

    def sym_reprojection_error(self, src_points, dst_points, H=None):
        if H is None:
            H = self.H

        mapped = cv2.perspectiveTransform(
            src_points.reshape(-1, 1, 2), H
        ).reshape(-1, 2)

        err_forward = np.mean(
            np.linalg.norm(mapped - dst_points, axis=1) ** 2
        )

        H_inv = np.linalg.inv(H)
        mapped = cv2.perspectiveTransform(
            dst_points.reshape(-1, 1, 2), H_inv
        ).reshape(-1, 2)

        err_backward = np.mean(
            np.linalg.norm(mapped - src_points, axis=1) ** 2
        )

        pairs = np.asarray(list(combinations(src_points, 2)))
        pairs_diff = pairs[:, 0] - pairs[:, 1]
        max_dist = np.linalg.norm(pairs_diff, axis=1).max()
        err_backward_norm = err_backward / (max_dist ** 2)

        return err_forward + err_backward_norm

    def table_containment_error(self, balls: list[Detection], H=None, table_length=L, table_width=W):
        if H is None:
            H = self.H

        balls_pos = self._ball_pos_with_offset(balls=balls)
        mapped = cv2.perspectiveTransform(
            balls_pos.reshape(-1, 1, 2), H
        ).reshape(-1, 2)


        contain = (
                (mapped[:, 1] >= 0) &
                (mapped[:, 1] <= table_length) &
                (mapped[:, 0] >= 0) &
                (mapped[:, 0] <= table_width)
        )

        return 1 - contain.sum() / len(balls)


    @staticmethod
    def _sort_rail(rail):
        x_range = max(pocket.center[0] for pocket in rail) - min(pocket.center[0] for pocket in rail)
        y_range = max(pocket.center[1] for pocket in rail) - min(pocket.center[1] for pocket in rail)
        axis = 0 if x_range > y_range else 1
        return sorted(rail, key=lambda pocket: pocket.center[axis], reverse=True)

    @staticmethod
    def _order_rails(r1, r2):
        center_1 = np.mean([pocket.center for pocket in r1], axis=0)
        center_2 = np.mean([pocket.center for pocket in r2], axis=0)
        axis = np.argmax(np.abs(center_1 - center_2))
        if center_1[axis] > center_2[axis]:
            return r2, r1
        return r1, r2

    def canonical_rail_pockets(self, two_rails: tuple[list, list]):
        r1, r2 = two_rails

        r1, r2 = self._order_rails(r1, r2)
        r1 = self._sort_rail(r1)
        r2 = self._sort_rail(r2)

        return {
            "lb": r1[0],
            "lc": r1[1],
            "lt": r1[2],
            "rb": r2[0],
            "rc": r2[1],
            "rt": r2[2],
        }

    def find_5_pocket_identities(self, image_state: ImageState = None):
        if image_state is None:
            image_state = self.image_state
        pockets = image_state.pockets
        balls = image_state.balls

        if len(pockets) != 5:
            return [], []

        scores = []
        identities = []
        for s1 in combinations(pockets, 3):
            s2 = tuple([p for p in pockets if p not in s1])

            line_err_1 = self.pca_line_error(s1, eps=0.2)
            if line_err_1 > VALID_RAIL_POCKETS_THRESHOLD:
                continue

            r1 = self._sort_rail(s1)
            r2 = self._sort_rail(s2)
            r1, r2 = self._order_rails(r1, r2)

            left_pockets = ["lb", "lc", "lt"]
            right_pockets = ["rb", "rc", "rt"]
            missing_candidates = left_pockets if len(r1) == 2 else right_pockets
            for missing_pocket in missing_candidates:
                canonical_left = [pocket for pocket in left_pockets if pocket != missing_pocket]
                canonical_right = [pocket for pocket in right_pockets if pocket != missing_pocket]
                canonical_5_pockets = canonical_left + canonical_right

                pocket_centers = [pocket.center for pocket in r1 + r2]
                canonical_centers = [CANONICAL_POCKETS[pocket] for pocket in canonical_5_pockets]

                src_points = np.asarray(pocket_centers, dtype=float)
                dst_points = np.asarray(canonical_centers, dtype=float)

                fit_indices = [0, 1, 2, 4] if len(r1) == 2 else [0, 2, 3, 4]
                src_4_points = src_points[fit_indices]
                dst_4_points = dst_points[fit_indices]

                H, _ = cv2.findHomography(src_4_points, dst_4_points)
                h_error = self.sym_reprojection_error(src_points, dst_points, H=H)

                missing_position = np.asarray(
                    CANONICAL_POCKETS[missing_pocket], dtype=float
                ).reshape(1, 1, 2)
                missing_position = cv2.perspectiveTransform(
                    missing_position, np.linalg.inv(H)
                ).reshape(2)
                height, width = image_state.image.shape[:2]
                # missing_is_visible = (
                #         0 <= missing_position[0] < width
                #         and 0 <= missing_position[1] < height
                # )

                # score = line_err_1 + h_error + float(missing_is_visible)

                contain_error = self.table_containment_error(balls, H)
                score = line_err_1 + h_error + contain_error
                identity = dict([(pocket, pos) for pocket, pos in zip(canonical_5_pockets, r1 + r2)])

                scores.append(float(score))
                identities.append(identity)

        return identities, scores

    def find_pocket_identities(self, image_state: ImageState = None):
        if image_state is None:
            image_state = self.image_state

        pockets = image_state.pockets

        if len(pockets) == 5:
            return self.find_5_pocket_identities(image_state)
        if len(pockets) != 6:
            return [], []

        identities = []
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
                identities.append(self.canonical_rail_pockets((s1, s2)))
                scores.append(score_1 + score_2)

        return identities, scores

    def calibrate_pocket_homography(self, image_state: ImageState=None):
        if image_state is None:
            image_state = self.image_state
        if image_state is None:
            raise ValueError("No ImageState found.")
        if len(image_state.pockets) not in (5, 6):
            # raise RuntimeError(
            #     "Automatic homography calibration requires 5 or 6 resolved pockets, "
            #     f"got {len(image_state.pockets)}."
            # )
            raise RuntimeError("NOT_ENOUGH_POCKETS")
        pocket_identities, scores = self.find_pocket_identities(image_state)

        if not pocket_identities:
            # raise RuntimeError("Could not determine pocket identities for homography calibration.")
            raise RuntimeError("POCKET_IDENTITIES_NOT_FOUND")

        best_idx = min(range(len(scores)), key=lambda i: scores[i])
        pockets = pocket_identities[best_idx]
        pocket_centers = [pocket.center for pocket in pockets.values()]
        src_points = np.asarray(pocket_centers, dtype=float)
        dst_points = np.asarray(
            [CANONICAL_POCKETS[pocket_id] for pocket_id in pockets],
            dtype=float,
        )

        H, mask = cv2.findHomography(src_points, dst_points, method=0)

        if H is None:
            raise RuntimeError("Homography estimation failed")

        self.set_homography(H)
        return self.H, mask

    def _ball_pos_with_offset(self, image_state: ImageState = None, balls=None, alpha=0.25):
        if image_state is None and balls is None:
            image_state = self.image_state
        if balls is None:
            balls = image_state.balls

        balls_pos = np.zeros([len(balls), 2], dtype=float)
        for i, ball in enumerate(balls):
            x1, y1, x2, y2 = ball.xyxy
            height = abs(y1 - y2)
            balls_pos[i] = np.asarray([ball.center[0], ball.center[1] - alpha * height], dtype=float)

        return balls_pos

    def transform_table_homography(self, image_state=None, H=None, preview=False) -> TableLayout:
        if image_state is None:
            image_state = self.image_state
        if H is None:
            H = self.H
        if image_state is None:
            raise ValueError("No ImageState found.")
        if H is None:
            raise RuntimeError("No homography available. Calibrate or set one first.")

        img_balls_pos = self._ball_pos_with_offset(image_state, alpha=self.ball_offset_alpha)
        ball_pos_cv = np.asarray(img_balls_pos, dtype=float).reshape(-1, 1, 2)

        mapped = cv2.perspectiveTransform(ball_pos_cv, H).reshape(-1, 2)

        if not image_state.unresolved_balls:
            mapped_unresolved = None
        else:
            unresolved_balls_pos = self._ball_pos_with_offset(balls=image_state.unresolved_balls,
                                                              alpha=self.ball_offset_alpha)
            unresolved_balls_cv = np.asarray(unresolved_balls_pos, dtype=float).reshape(-1, 1, 2)
            mapped_unresolved = cv2.perspectiveTransform(unresolved_balls_cv, H).reshape(-1, 2)

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

        return self.build_table_layout(image_state, mapped, mapped_unresolved)

    def build_table_layout(self, image_state, mapped_balls: np.ndarray, mapped_unresolved) -> TableLayout:
        img_balls = image_state.balls
        ball_classes = [ball.class_id for ball in img_balls]

        balls_info = {}
        ball_ids = set()
        mapped_balls = self.correct_ball_pos(mapped_balls)
        for i, pos in enumerate(mapped_balls):
            cls = ball_classes[i]
            ball_id = BALL_CLASS_TO_ID[cls]
            balls_info[ball_id] = tuple(pos)
            ball_ids.add(ball_id)
        unused_ball_ids = set(BALL_CLASS_TO_ID.values()) - ball_ids

        if not image_state.unresolved_balls or mapped_unresolved is None:
            return TableLayout(image_state, balls_info, ball_ids, unused_ball_ids)

        unresolved_classes = [ball.class_id for ball in image_state.unresolved_balls]
        unresolved_balls_info = {}
        for i, pos in enumerate(mapped_unresolved):
            cls = unresolved_classes[i]
            ball_id = BALL_CLASS_TO_ID[cls]
            unresolved_balls_info[ball_id] = tuple(pos)

        return TableLayout(image_state, balls_info, ball_ids, unused_ball_ids, unresolved_balls_info)

    def correct_ball_pos(self, mapped_balls: np.ndarray, threshold=0.1, table_length=L, table_width=W,
                         r=R, epsilon=0.005):
        """
        Checks ball radius against cushions and other balls;
        Corrects their positions if invalid
        :param mapped_balls: homography transformed ball positions in ndarray
        :return: corrected_balls: corrected ball positions
        """
        corrected_balls = np.zeros_like(mapped_balls, dtype=float)
        for i, pos in enumerate(mapped_balls):
            x, y = pos
            corrected_balls[i][0] = min(max(x, r + epsilon), table_width - r - epsilon)
            corrected_balls[i][1] = min(max(y, r +epsilon), table_length - r - epsilon)

        min_distance = 2 * r + 1e-6
        for _ in range(200):
            for i, j in combinations(range(len(corrected_balls)), 2):
                delta = corrected_balls[j] - corrected_balls[i]
                distance = np.linalg.norm(delta)
                if distance >= min_distance:
                    continue
                direction = delta / distance if distance > 0 else np.array([1.0, 0.0])
                correction = direction * (min_distance - distance) / 2
                corrected_balls[i] -= correction
                corrected_balls[j] += correction
                corrected_balls[[i, j]] = np.clip(
                    corrected_balls[[i, j]], r, [table_width - r, table_length - r]
                )

            if all(
                np.linalg.norm(corrected_balls[j] - corrected_balls[i]) >= min_distance - 1e-8
                for i, j in combinations(range(len(corrected_balls)), 2)
            ):
                break
        else:
            raise ValueError("Could not separate overlapping balls within the table bounds.")

        for i in range(len(corrected_balls)):
            dist = np.linalg.norm(corrected_balls[i] - mapped_balls[i])
            if dist > threshold:
                print(f"[Warning] Large correction: {dist:.2f}m. original={mapped_balls[i]}, corrected={corrected_balls[i]}")

        return corrected_balls



if __name__ == "__main__":
    test_img_path = Path(__file__).parent / "tests" / "test8.jpg"
    image = cv2.imread(str(test_img_path))

    table_vision = TableVision()
    print(table_vision.predict_layout(image, preview=True))
