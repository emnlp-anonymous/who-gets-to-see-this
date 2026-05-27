import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary and multiclass classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
     Args:
        gamma      : focusing parameter (default 2.0)
        alpha      : class weights tensor or None (same shape as num_classes)
        reduction  : 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float, alpha, reduction: str):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        ce_loss = F.cross_entropy(logits, labels, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)                         # probability of correct class
        focal_loss = (1.0 - pt) ** self.gamma * ce_loss
 
        if self.reduction == "mean":
            return focal_loss.mean()    
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

def get_loss_fn(config: dict):
    """
    Expected config keys:
        loss_type  : "cross_entropy" | "focal"
        focal_gamma: float (only used when loss_type == "focal", default 2.0)
        class_weights: list[float] | None  — applied to both CE and focal
    """
    loss_type = config.get("loss_type", "cross_entropy")
    weights = config.get("class_weights", None)
    weight_tensor = torch.tensor(weights, dtype=torch.float) if weights is not None else None
 
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight_tensor)
 
    elif loss_type == "focal":
        gamma = config.get("focal_gamma", 2.0)
        alpha = weights  # reuse class_weights as alpha
        return FocalLoss(gamma=gamma, alpha=alpha, reduction='mean')
 
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}. Choose 'cross_entropy' or 'focal'.")
