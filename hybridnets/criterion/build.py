import pytorch_lightning as pl
import torch.nn as nn

from .focal_loss import FocalLoss

class CriterionCompose(pl.LightningModule):
    def __init__(self, criterions, weights):
        super().__init__()
        self.criterions = nn.ModuleList(criterions)
        self.weights = weights

    def __call__(self, inp, target):
        losses = {}
        for criterion, weight in zip(self.criterions, self.weights):
            criterion_loss = criterion(inp, target)
            for loss_name, loss in criterion_loss.items():
                losses[loss_name] = weight * loss
        losses["loss"] = sum([v for v in losses.values()])
        return losses


class DetectionLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.det_criterion = FocalLoss()
    
    def __call__(self, inp, target):
        cls_loss, reg_loss = self.det_criterion(inp, target)

        losses = {
            "classification_loss": cls_loss,
            "regression_loss": reg_loss,
        }
        return losses
        
        
def build_criterion(cfg):
    criterions = [
        DetectionLoss(),
    ]
    weights = [1, 1]
    
    criterion = CriterionCompose(criterions, weights)
    return criterion
