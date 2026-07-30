from ultralytics import YOLO

from pathlib import Path


DATASETS_FOLDER = Path(__file__).resolve().parent
MODEL_PATH = DATASETS_FOLDER.parent / "models" / "yolo_3.pt"
CAPTURES_FOLDER = DATASETS_FOLDER / "captures"

PRELABELS_FOLDER = DATASETS_FOLDER / "prelabels"

POCKET_INFO = [
    (633.00, 270.95, 74.01, 45.92),
    (1296.90, 268.91, 91.67, 50.39),
    (518.66, 907.07, 106.86, 66.75),
    (1412.75, 903.09, 117.93, 67.62),
    (570.67, 541.76, 80.93, 64.50),
    (1360.81, 541.70, 89.62, 63.40),
]

POCKET_CLASS_ID = 11
REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

REPLACE_POCKETS = True
BALL_CLASS_IDS = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 13, 16, 17, 18,
}
BALL_OVERLAP_IOU = 0.85


def format_label(class_id: int, box: tuple[float, float, float, float]) -> str:
    x_center, y_center, width, height = box
    return (
        f"{class_id} {x_center:g} {y_center:g} "
        f"{width:g} {height:g}"
    )


def normalized_pockets() -> list[tuple[int, tuple[float, float, float, float]]]:
    return [
        (
            POCKET_CLASS_ID,
            (
                x_center / REFERENCE_WIDTH,
                y_center / REFERENCE_HEIGHT,
                width / REFERENCE_WIDTH,
                height / REFERENCE_HEIGHT,
            ),
        )
        for x_center, y_center, width, height in POCKET_INFO
    ]


def replace_incomplete_pockets(detections):
    pocket_count = sum(
        class_id == POCKET_CLASS_ID
        for class_id, _ in detections
    )
    if pocket_count >= len(POCKET_INFO):
        return detections

    non_pockets = [
        detection
        for detection in detections
        if detection[0] != POCKET_CLASS_ID
    ]
    return non_pockets + normalized_pockets()


def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left = max(x1 - w1 / 2, x2 - w2 / 2)
    top = max(y1 - h1 / 2, y2 - h2 / 2)
    right = min(x1 + w1 / 2, x2 + w2 / 2)
    bottom = min(y1 + h1 / 2, y2 + h2 / 2)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = w1 * h1 + w2 * h2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def remove_ball_overlaps(detections, confidences):
    if len(detections) != len(confidences):
        raise ValueError("length of detections and confidences must match")

    kept_indices = {
        index
        for index, (class_id, _) in enumerate(detections)
        if class_id not in BALL_CLASS_IDS
    }
    kept_ball_indices = []
    ball_indices = sorted(
        (
            index
            for index, (class_id, _) in enumerate(detections)
            if class_id in BALL_CLASS_IDS
        ),
        key=lambda index: (-confidences[index], index),
    )

    for index in ball_indices:
        box = detections[index][1]
        if all(
            iou(box, detections[kept_index][1]) <= BALL_OVERLAP_IOU
            for kept_index in kept_ball_indices
        ):
            kept_ball_indices.append(index)
            kept_indices.add(index)

    return (
        [
            detection
            for index, detection in enumerate(detections)
            if index in kept_indices
        ],
        [
            confidence
            for index, confidence in enumerate(confidences)
            if index in kept_indices
        ],
    )


def main() -> None:
    model = YOLO(MODEL_PATH)

    for match_folder in sorted(CAPTURES_FOLDER.iterdir()):
        if match_folder.name != "horizontal_view_2":
            continue

        if "horizontal_view" in match_folder.name:
            image_folder = match_folder
            replace_pockets = False
        else:
            image_folder = match_folder / "images"
            replace_pockets = REPLACE_POCKETS
        if not image_folder.is_dir():
            continue

        output_folder = PRELABELS_FOLDER / match_folder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        results = model.predict(
            source=image_folder,
            imgsz=960,
            conf=0.3,
            iou=0.50,
            save=False,
            stream=True,
        )

        for result in results:
            detections = []
            confidences = []
            if result.boxes is not None:
                detections = [
                    (int(class_id), tuple(box))
                    for class_id, box in zip(
                        result.boxes.cls.tolist(),
                        result.boxes.xywhn.tolist(),
                    )
                ]
                confidences = result.boxes.conf.tolist()

            detections, confidences = remove_ball_overlaps(
                detections,
                confidences,
            )

            if replace_pockets:
                detections = replace_incomplete_pockets(detections)

            lines = [
                format_label(class_id, box)
                for class_id, box in detections
            ]
            label_path = output_folder / f"{Path(result.path).stem}.txt"
            label_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
