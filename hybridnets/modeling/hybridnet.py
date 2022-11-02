import os
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl

from railyard.dataclasses import pad_detections_tensor
from railyard.env import MODEL_ZOO_DIR
from railyard.util.categories import lookup_category_list
from railyard.util.visualization import normalize_tensor, apply_color, overlay_images_batch, draw_bounding_boxes

from .backbone import build_backbone
from .fpn import build_fpn
from .components import Regressor, Classifier, BiFPNDecoder, Anchors

class HybridNet(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.size = cfg.INPUT.SIZE
        self.pretrained = cfg.MODEL.DETECTION_HEAD.PRETRAINED
        self.anchors_scales = cfg.MODEL.DETECTION_HEAD.ANCHORS_SCALES
        self.anchors_ratios = cfg.MODEL.DETECTION_HEAD.ANCHORS_RATIOS
        self.nms_threshold = cfg.MODEL.DETECTION_HEAD.NMS_THRESHOLD
        self.vis_threshold = cfg.MODEL.DETECTION_HEAD.VIS_THRESHOLD
        
        self.backbone_coef = cfg.MODEL.BACKBONE.COEFFICIENT
        self.cat_list = lookup_category_list(cfg.MODEL.DETECTION_HEAD.CATEGORY_LIST)
        self.num_classes = len(self.cat_list) - 1
        self.num_scales = len(self.anchors_scales)
        self.num_anchors = len(self.anchors_ratios) * self.num_scales
        
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scales = [1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,]

        self.encoder = build_backbone(cfg)
        self.bifpn, self.fpn_num_channels = build_fpn(cfg)
        self.initialize_decoder(self.bifpn)

        # Detection
        self.anchors = Anchors(self.anchors_scales, self.anchors_ratios, self.size,
                               anchor_scale=self.anchor_scales[self.backbone_coef],
                                pyramid_levels=(torch.arange(self.pyramid_levels[self.backbone_coef]) + 3).tolist())

        self.regressor = Regressor(in_channels=self.fpn_num_channels,
                                   num_anchors=self.num_anchors,
                                   num_layers=self.box_class_repeats[self.backbone_coef],
                                   pyramid_levels=self.pyramid_levels[self.backbone_coef],
                                   onnx_export=False)

        self.classifier = Classifier(in_channels=self.fpn_num_channels,
                                     num_anchors=self.num_anchors,
                                     num_classes=self.num_classes,
                                     num_layers=self.box_class_repeats[self.backbone_coef],
                                     pyramid_levels=self.pyramid_levels[self.backbone_coef],
                                     onnx_export=False)

        # Segmentation
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_channels)
        self.initialize_decoder(self.bifpndecoder)
    
        self.initialize_weights()
        
        if self.pretrained:
            backbone_name = cfg.MODEL.BACKBONE.NAME
            if "efficientnet" in backbone_name:
                weights_path = os.path.join(MODEL_ZOO_DIR, "hybrid_efficientnet_d3.pth")
            elif "regnet" in backbone_name:
                weights_path = os.path.join(MODEL_ZOO_DIR, "hybrid_regnetx_004.pth")
            else:
                raise Exception(f"Backbone not recognized: {backbone_name}")
            self.load_state_dict(torch.load(weights_path), strict=False)

    def forward(self, inp):
        x = inp['image']
        
        p2, p3, p4, p5 = self.encoder(x)[-4:]
        features = self.bifpn((p3, p4, p5))

        # Detection
        anchors = self.anchors.get_anchor_boxes()
        regression = self.regressor(features)
        classification = self.classifier(features)
        
        # Segmentation
        p3,p4,p5,p6,p7 = features
        outputs = self.bifpndecoder((p2,p3,p4,p5,p6,p7))
        
        target = {
            "anchors": anchors,
            "regression": regression,
            "classification": classification,
        }
        return target
    
    def postprocess(self, target):
        anchors = target["anchors"]
        regression = target["regression"]
        classification = target["classification"]
        
        anchors_transformed = transform_anchors(anchors, regression, self.size)
        
        boxes_all = []
        scores_all = []
        labels_all = []
        for i in range(classification.shape[0]):
            classification_per = classification[i]
            scores_per, labels_per = classification_per.max(dim=1)
            anchors_per = anchors_transformed[i]
            
            nms_idxs = torchvision.ops.boxes.batched_nms(anchors_per, scores_per, labels_per, iou_threshold=self.nms_threshold)
            boxes = anchors_per[nms_idxs]
            labels = labels_per[nms_idxs]
            scores = scores_per[nms_idxs]
            
            boxes, labels, scores = pad_detections_tensor(boxes, labels, scores)

            boxes_all.append(boxes)
            labels_all.append(labels)
            scores_all.append(scores)
        boxes_all = torch.stack(boxes_all)
        labels_all = torch.stack(labels_all)
        scores_all = torch.stack(scores_all)

        # Add background class
        labels_all += 1
        
        out = {
            "detection_boxes": boxes_all,
            "detection_scores": scores_all,
            "detection_labels": labels_all,
        }
        return out

    def visualize(self, batch, vis) -> None:
        image_batch = vis["main_vis"]
        boxes_all = batch["detection_boxes"]
        labels_all = batch["detection_labels"]
        scores_all = batch["detection_scores"]

        images = []
        for image, boxes, labels, scores in zip(
            image_batch, boxes_all, labels_all, scores_all
        ):
            boxes = boxes.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy().astype(int)
            scores = scores.cpu().detach().numpy()

            boxes = boxes[scores > self.vis_threshold]
            labels = labels[scores > self.vis_threshold]
            scores = scores[scores > self.vis_threshold]

            cat_names = [self.cat_list[int(l)] for l in labels]
            captions = [f"{c} {s:.2f}" for c, s in zip(cat_names, scores)]
            colors = apply_color(labels)

            image = torchvision.transforms.ToPILImage()(image)
            image = draw_bounding_boxes(image, boxes, captions, colors)
            image = torchvision.transforms.ToTensor()(image)
            images.append(image)

        vis["main_vis"] = images

    def initialize_decoder(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if "conv_list" or "header" in name:
                    variance_scaling_(module.weight.data)
                else:
                    nn.init.kaiming_uniform_(module.weight.data)

                if module.bias is not None:
                    if "classifier.header" in name:
                        bias_value = -np.log((1 - 0.01) / 0.01)
                        torch.nn.init.constant_(module.bias, bias_value)
                    else:
                        module.bias.data.zero_()

def transform_anchors(anchors, regression, size):
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.
    
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)
    boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
    boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)
    boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=size[0] - 1)
    boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=size[1] - 1)
    return boxes


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return nn.init._no_grad_normal_(tensor, 0., std)