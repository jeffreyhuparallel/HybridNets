import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua
    return IoU


class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, target):
        boxes = inp["detection_boxes"]
        labels = inp["detection_labels"]
        regressions = target["regression"]
        classifications = target["classification"]
        anchors = target["anchors"]
        
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            
            box = boxes[j]
            label = labels[j]
            box = box[label != 0]
            label = label[label != 0] - 1
            bbox_annotation = torch.cat([box, label.unsqueeze(dim=1)], dim=1)

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones_like(classification) * alpha
                alpha_factor = alpha_factor.cuda()
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                regression_losses.append(torch.tensor(0).to(dtype).cuda())
                classification_losses.append(cls_loss.sum())
                continue

            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])
            
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)
           

            targets = torch.zeros_like(classification)
            
            if torch.cuda.is_available():
                targets = targets.cuda()
            
            assigned_annotations = bbox_annotation[IoU_argmax, :]
            
            positive_indices = torch.full_like(IoU_max,False,dtype=torch.bool) #torch.ge(IoU_max, 0.2) 
                        
            tensorA = (assigned_annotations[:, 2] - assigned_annotations[:, 0]) * (assigned_annotations[:, 3] - assigned_annotations[:, 1]) > 10 * 10

            
            positive_indices[torch.logical_or(torch.logical_and(tensorA,IoU_max >= 0.5),torch.logical_and(~tensorA,IoU_max >= 0.15))] = True

            num_positive_anchors = positive_indices.sum()
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
    
            alpha_factor = torch.ones_like(targets) * alpha
            alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    
        cls_loss = torch.stack(classification_losses).mean(dim=0, keepdim=True)
        det_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50 # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233
        return cls_loss, det_loss
                 