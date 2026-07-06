from pathlib import Path

# Корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Данные
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# YAML
DATASET_YAML = PROCESSED_DATA_DIR / "dataset.yaml"

# Результаты
RESULTS_DIR = PROJECT_ROOT / "results"

WEIGHTS_DIR = RESULTS_DIR / "weights"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Обучение
IMAGE_SIZE = 640
BATCH_SIZE = 32
EPOCHS = 100

# Модель
YOLO_MODEL = "yolov8n.pt"

# Device
DEVICE = 0