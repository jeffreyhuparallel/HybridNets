import pytorch_lightning as pl
import torch.nn as nn

from hybridnets.criterion.loss import FocalLoss, FocalLossSeg, TverskyLoss

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
    def __init__(self, cfg):
        super().__init__()
        self.det_criterion = FocalLoss()
    
    def __call__(self, inp, target):
        cls_loss, reg_loss = self.det_criterion(inp, target)

        losses = {
            "classification_loss": cls_loss,
            "regression_loss": reg_loss,
        }
        return losses

class SegmentationLoss(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.seg_criterion1 = TverskyLoss(alpha=0.7, beta=0.3, gamma=4.0/3, from_logits=True)
        self.seg_criterion2 = FocalLossSeg(alpha=0.25)
    
    def __call__(self, inp, target):
        seg_annot = inp['segmentation']
        segmentation = target["segmentation"]
        tversky_loss = self.seg_criterion1(segmentation, seg_annot)
        focal_loss = self.seg_criterion2(segmentation, seg_annot)
        seg_loss = tversky_loss + 1 * focal_loss
        
        losses = {
            "segmentation_loss": seg_loss,
        }
        return losses
        
        
def build_criterion(cfg):
    criterions = [
        DetectionLoss(cfg),
        # SegmentationLoss(cfg),
    ]
    weights = [1, 1]
    
    criterion = CriterionCompose(criterions, weights)
    return criterion
