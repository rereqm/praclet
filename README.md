# Traffic Sign Detection

Проект по обнаружению дорожных знаков с использованием:

- YOLOv8
- YOLOv10
- Faster R-CNN
- SSD300

## Dataset

GTSDB

## Запуск

```bash
pip install -r requirements.txt
```

Обучение YOLOv8

```bash
python -m src.training.train_yolov8
```

Обучение Faster R-CNN

```bash
python -m src.training.train_faster_rcnn
```

Оценка моделей

```bash
python -m src.evaluation.benchmark
```