import torch
from torchvision import models


def make_model(num_classes, device):
    model = models.segmentation.deeplabv3_resnet101(
        weights="DeepLabV3_ResNet101_Weights.DEFAULT", progress=True
    )

    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, 1)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)
