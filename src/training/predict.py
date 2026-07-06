from pathlib import Path
from ultralytics import YOLO

project_root = Path(__file__).resolve().parents[2]

model = YOLO(
    project_root / "results" / "YOLOv8n-2" / "weights" / "best.pt"
)

test_folder = project_root / "data" / "processed" / "images" / "test"

model.predict(
    source=str(test_folder),
    save=True,
    conf=0.25
)