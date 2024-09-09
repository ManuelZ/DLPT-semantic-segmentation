import torch
from torcheval.metrics import MulticlassConfusionMatrix


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

        if self.ignore_indices is not None:
            # set column values of ignore classes to 0
            conf_matrix[:, self.ignore_indices] = 0
            # set row values of ignore classes to 0
            conf_matrix[self.ignore_indices, :] = 0

        true_positive = torch.diag(conf_matrix)
        false_positive = torch.sum(conf_matrix, 0) - true_positive
        false_negative = torch.sum(conf_matrix, 1) - true_positive

        DSC = (2 * true_positive + self.eps) / (
            2 * true_positive + false_positive + false_negative + self.eps
        )

        return DSC
