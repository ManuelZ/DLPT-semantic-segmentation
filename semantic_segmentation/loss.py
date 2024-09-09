import torch


class SoftDiceLoss(torch.nn.Module):
    """
    Implementation of the Soft-Dice Loss function.

    Arguments:
        num_classes (int): number of classes.
        eps (float): value of the floating point epsilon.
    """

    def __init__(self, num_classes, eps=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, preds, targets):
        """
        Compute Soft-Dice Loss.

        Arguments:
            preds (torch.FloatTensor):
                tensor of predicted labels. The shape of the tensor is (B, num_classes, H, W).
            targets (torch.LongTensor):
                tensor of ground-truth labels. The shape of the tensor is (B, H, W).
        Returns:
            mean_loss (float32): mean loss by class  value.
        """

        loss = 0
        for cls in range(self.num_classes):

            # get ground truth for the current class
            target = (targets == cls).float()

            # get prediction for the current class
            pred = preds[:, cls]

            # calculate intersection
            intersection = (pred * target).sum()

            # compute dice coefficient
            dice = (2 * intersection + self.eps) / (
                pred.sum() + target.sum() + self.eps
            )

            # compute negative logarithm from the obtained dice coefficient
            loss = loss - dice.log()

        # get mean loss by class value
        loss = loss / self.num_classes

        return loss


class CombinedLoss(torch.nn.Module):
    """
    Linear combination of two loss functions.
    """

    def __init__(self, loss_fn1, loss_fn2, weight1=0.5, weight2=0.5):
        super().__init__()
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.weight1 = weight1
        self.weight2 = weight2

    def forward(self, preds_logits, targets):

        if isinstance(preds_logits, dict):
            preds_logits = preds_logits["out"]

        preds_probs = preds_logits.softmax(dim=1)

        loss1 = self.loss_fn1(preds_probs, targets)
        loss2 = self.loss_fn2(preds_logits, targets)

        combined_loss = self.weight1 * loss1 + self.weight2 * loss2

        return combined_loss
