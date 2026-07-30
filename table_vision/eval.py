from ultralytics import YOLO
from pathlib import Path
import torch


folder = Path(__file__).resolve().parent
model_path = folder / "models" / "yolo_3.pt"
dataset_folder = folder / "datasets" / "dataset_3_fix"
data_path = dataset_folder / "data.yaml"

model = YOLO(model_path)

model.val(
    data=data_path,
    imgsz=960,
    split="test",
    conf=0.001,
    plots=True,
    device=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"),
    project="train",
    name="train_3_fix_test",
)