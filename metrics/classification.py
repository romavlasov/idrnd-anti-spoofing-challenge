import torch


def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        res = torch.sum(output == target)
        return res.float() / target.size(0)
    
    
def min_c(output, target):
    # FP/(FP+TN) + 19⋅FN/(FN+TP)
    with torch.no_grad():
        eps=1e-9
        
        TP = (output & target).sum().float()
        TN = (~output & ~target).sum().float()
        FP = (output & ~target).sum().float()
        FN = (~output & target).sum().float()
        
        res = FP / (FP + TN + eps) + 19 * FN / (FN + TP + eps)
        return res / target.size(0)