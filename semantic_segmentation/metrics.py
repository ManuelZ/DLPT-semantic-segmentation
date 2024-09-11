import torch
from torcheval.metrics import MulticlassConfusionMatrix
import segmentation_models_pytorch as smp


class DiceScore(torch.nn.Module):
    """ """

    def __init__(self, num_classes, ignore_index=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index: int = ignore_index
        self.eps = 1e-6
        self.metric = MulticlassConfusionMatrix(self.num_classes)

    def __call__(self, pred, target):
        """
        pred: NxHxW
        target: NxCxHxW
        """
        self.metric.reset()

        self.metric.update(pred.flatten(), target.flatten())
        conf_matrix = self.metric.compute()

        if self.ignore_index is not None:
            # set column values of ignore classes to 0
            conf_matrix[:, self.ignore_index] = 0
            # set row values of ignore classes to 0
            conf_matrix[self.ignore_index, :] = 0

        true_positive = torch.diag(conf_matrix)
        false_positive = torch.sum(conf_matrix, 0) - true_positive
        false_negative = torch.sum(conf_matrix, 1) - true_positive

        DSC = (2 * true_positive + self.eps) / (
            2 * true_positive + false_positive + false_negative + self.eps
        )

        return DSC


class IOU(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def __call__(self, pred, target):
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred, target, mode="multiclass", num_classes=self.num_classes
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        return iou_score
