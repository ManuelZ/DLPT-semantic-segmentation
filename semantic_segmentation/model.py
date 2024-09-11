import torch
from torchvision import models
def make_deeplabv3_resnet101(num_classes, device):
    """ """

    model = models.segmentation.deeplabv3_resnet101(
        weights="DeepLabV3_ResNet101_Weights.DEFAULT", progress=True
    )

    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Fresh new last layers of the classifiers
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, 1)

    return model.to(device)


def unfreeze_deeplabv3_resnet101(model, layers: list[str] | None = None):
    """ """

    resnet101_layers = ["layer1", "layer2", "layer3", "layer4"]

    # Unfreeze whole DeeplabV3
    if layers is None:
        for param in model.parameters():
            param.requires_grad = True
    else:  # Unfreeze only certain specific layers of the backbone
        for layer in layers:
            assert layer in resnet101_layers, f"Layer {layer} isn't part of Resnet101"
            for param in getattr(model.backbone, layer).parameters():
                param.requires_grad = True
            print(f"Unfreezing parameters of layer {layer}")

