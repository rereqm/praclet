from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

print(model.head.classification_head)

print("\n-------------------------\n")

print(dir(model.head.classification_head))