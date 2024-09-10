import torch
from torchvision import models


def make_model(num_classes, device):
    """ """

    model = models.segmentation.deeplabv3_resnet101(
        weights="DeepLabV3_ResNet101_Weights.DEFAULT", progress=True
    )

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, 1)

    return model.to(device)


def unfreeze_deeplabv3(model, layers=None):

    if layers is None:
        # Unfreeze whole backbone
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        print("To be done")
        pass

    # Set the last backbone layers to be trainable
    # for param in model.backbone.layer4.parameters():
    #     param.requires_grad = True

    # for param in model.backbone.layer3.parameters():
    #     param.requires_grad = True

    # for param in model.backbone.layer2.parameters():
    #     param.requires_grad = True

    # for param in model.backbone.layer1.parameters():
    #     param.requires_grad = True
