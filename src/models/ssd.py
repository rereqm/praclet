import torch
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from configs.classes import CLASSES

NUM_CLASSES = len(CLASSES) + 1  # фон + классы


def get_model():
    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

    # ❗ ПРАВИЛЬНО: НЕ трогаем in_channels вообще
    from torchvision.models.detection.ssd import SSDClassificationHead

    # берём реальные параметры модели
    num_anchors = model.anchor_generator.num_anchors_per_location()
    in_channels = [512, 1024, 512, 256, 256, 256]

    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES
    )

    return model