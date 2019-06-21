import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
            
        p = torch.exp(-bce)
        f_loss = self.alpha * (1 - p)**self.gamma * bce

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss
        
        
def focal(*argv, **kwargs):
    return FocalLoss(*argv, **kwargs)