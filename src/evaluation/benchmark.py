from pathlib import Path
import time

from ultralytics import YOLO

from src.evaluation.evaluate import evaluate_model
from src.models.ssd import get_model
from src.dataset.loader import GTSDBDataset, collate_fn

from torch.utils.data import DataLoader
import torch


def main():

    root = Path(__file__).resolve().parents[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- DATASET ----------------
    dataset = GTSDBDataset(
        root / "data/processed/images/val",
        root / "data/processed/labels/val"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    # ================= SSD =================
    print("\n[SSD]")
    ssd = get_model().to(device)

    ssd_metrics = evaluate_model(ssd, dataloader, device)

    print(f"SSD FPS: {ssd_metrics['fps']:.2f}")

    # ================= YOLOv8 =================
    print("\n[YOLOv8]")
    yolo8_path = root / "results/YOLOv8n-2/weights/best.pt"
    model8 = YOLO(str(yolo8_path))

    start = time.perf_counter()
    for img in dataset.images:
        model8.predict(str(img), verbose=False)
    yolo8_fps = len(dataset) / (time.perf_counter() - start)

    print(f"YOLOv8 FPS: {yolo8_fps:.2f}")

    # ================= YOLOv10 =================
    print("\n[YOLOv10]")
    yolo10_path = root / "results/YOLOv10n/weights/best.pt"
    model10 = YOLO(str(yolo10_path))

    start = time.perf_counter()
    for img in dataset.images:
        model10.predict(str(img), verbose=False)
    yolo10_fps = len(dataset) / (time.perf_counter() - start)

    print(f"YOLOv10 FPS: {yolo10_fps:.2f}")

    # ================= SUMMARY =================
    print("\n===== FINAL RESULTS =====")
    print(f"SSD     : {ssd_metrics['fps']:.2f} FPS")
    print(f"YOLOv8  : {yolo8_fps:.2f} FPS")
    print(f"YOLOv10 : {yolo10_fps:.2f} FPS")


if __name__ == "__main__":
    main()