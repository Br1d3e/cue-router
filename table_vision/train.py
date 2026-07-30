from ultralytics import YOLO

from pathlib import Path
import torch


folder = Path(__file__).resolve().parent
model_path = folder / "models" / "yolo_2.pt"
dataset_folder = folder / "datasets" / "dataset_3_fix"
data_path = dataset_folder / "data.yaml"

model = YOLO(model_path)

model.train(
    data=data_path,
    imgsz=960,
    batch=8,
    cache=False,

    epochs=80,
    patience=20,
    lr0=1e-4,
    lrf=0.1,
    weight_decay=1e-4,

    device=torch.device("mps") if torch.backends.mps.is_available() else None,
    optimizer='AdamW',

    degrees=1.0,
    translate=0.05,
    scale=0.20,
    shear=0.0,
    perspective=0.0002,
    fliplr=0.5,
    flipud=0.0,

    mosaic=0.10,
    mixup=0.0,
    close_mosaic=10,

    seed=42,
    deterministic=True,

    project="train",
    name="train_3_fix",
)