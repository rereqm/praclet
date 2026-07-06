import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


NUM_CLASSES = 44  # 43 класса + фон


def get_model():

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )

    return model