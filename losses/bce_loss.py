import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def __init__(self, logits=True, reduce=True):
        super(BCELoss, self).__init__()
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')

        if self.reduce:
            return torch.mean(bce)
        else:
            return bce


def bce(*argv, **kwargs):
    return BCELoss(*argv, **kwargs)