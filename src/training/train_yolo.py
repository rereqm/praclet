from ultralytics import YOLO

from configs.settings import (
    DATASET_YAML,
    RESULTS_DIR,
    YOLO_MODEL,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    DEVICE
)


def train():

    model = YOLO(YOLO_MODEL)

    model.train(
        data=str(DATASET_YAML),
        imgsz=IMAGE_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(RESULTS_DIR),
        name="YOLOv8n"
    )


if __name__ == "__main__":
    train()