import torch
import torch.nn as nn
from torch.nn import functional as F

# build loss module
class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()
        # self.add_aux = add_aux 
        self.loss_func = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, y_pred, y_true):
        batch_loss = self.loss_func(y_pred, y_true)
        final_loss = torch.abs(batch_loss.mean())
        return final_loss

class AttributeLoss(nn.Module):
    def __init__(self):
        super(AttributeLoss, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.m = nn.Sigmoid()

    def forward(self, attr_pred, attr_true):
        # print("attr true > ", attr_true, "attr pred > ", attr_pred)
        output = self.loss_func(attr_pred, attr_true)
        return output



class NoFussCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    pytorch's CrossEntropyLoss is fussy: 1) needs Long (int64) targets only, and 2) only 1D.
    This function satisfies these requirements
    """

    def forward(self, inp, target):
        return F.cross_entropy(inp, target.long().squeeze(), weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)
