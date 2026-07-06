from pathlib import Path

from ultralytics import YOLO


def main():

    root = Path(__file__).resolve().parents[2]

    model = YOLO("yolov10n.pt")

    model.train(
        data=root / "data/processed/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project=root / "results",
        name="YOLOv10n"
    )


if __name__ == "__main__":
    main()