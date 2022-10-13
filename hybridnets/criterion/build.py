import pytorch_lightning as pl
import torch.nn as nn

from hybridnets.loss import FocalLoss, FocalLossSeg, TverskyLoss

class CriterionCompose(pl.LightningModule):
    def __init__(self, criterions, weights):
        super().__init__()
        self.criterions = nn.ModuleList(criterions)
        self.weights = weights

    def __call__(self, inp, out):
        losses = {}
        for criterion, weight in zip(self.criterions, self.weights):
            criterion_loss = criterion(inp, out)
            for loss_name, loss in criterion_loss.items():
                losses[loss_name] = weight * loss
        losses["loss"] = sum([v for v in losses.values()])
        return losses


class DetectionLoss(pl.LightningModule):
    def __init__(self, params):
        pass
    
    def __call__(self, inp, out):
        cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        

def build_criterion(cfg):
    criterions = []
    weights = []
    
    criterion = CriterionCompose(criterions, weights)
    return criterion
