from ultralytics import YOLO
import time
from pathlib import Path


def evaluate_yolo(model_path, dataset_images_dir):
    model = YOLO(model_path)

    images = list(Path(dataset_images_dir).glob("*.png"))

    start = time.perf_counter()

    for img in images:
        model.predict(img, verbose=False)

    end = time.perf_counter()

    fps = len(images) / (end - start)

    return {
        "fps": fps,
        "num_images": len(images)
    }