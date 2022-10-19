import numpy as np
import torch
from torch import nn
import timm
import torchvision

from hybridnets.encoders import get_encoder
from hybridnets.modeling.model import BiFPN, Regressor, Classifier, BiFPNDecoder
from hybridnets.modeling.model import SegmentationHead
from hybridnets.utils.utils import Anchors, init_weights
from hybridnets.utils.utils import BBoxTransform, ClipBoxes, postprocess
from hybridnets.utils.constants import *

from railyard.util.categories import lookup_category_list
from railyard.util.visualization import normalize_tensor, apply_color, overlay_images_batch, draw_bounding_boxes

class HybridNetsBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone_name = cfg.MODEL.BACKBONE.NAME
        self.compound_coef = cfg.MODEL.BACKBONE.COMPOUND_COEF
        self.seg_classes = 2
        self.seg_mode = "multiclass"
        self.num_classes = cfg.MODEL.DETECTION_HEAD.NUM_CLASSES
        self.anchors_scales = cfg.MODEL.DETECTION_HEAD.ANCHORS_SCALES
        self.anchors_ratios = cfg.MODEL.DETECTION_HEAD.ANCHORS_RATIOS
        self.num_scales = len(self.anchors_scales)
        self.conf_thres = 0.25
        self.iou_thres = 0.3
        self.obj_list = lookup_category_list(cfg.DATASETS.TRAIN[0], include_background=False)

        self.num_anchors = len(self.anchors_ratios) * self.num_scales
        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,]
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }
        if "regnet" in self.backbone_name:
            conv_channel_coef = {3: [64, 160, 384]} # regnetx_004

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[self.compound_coef],
                    True if _ == 0 else False,
                    attention=True if self.compound_coef < 6 else False,
                    use_p8=self.compound_coef > 7,
                    onnx_export=False)
              for _ in range(self.fpn_cell_repeats[self.compound_coef])])

        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=self.num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef],
                                   onnx_export=False)

        '''Modified by Dat Vu'''
        # self.decoder = DecoderModule()
        self.bifpndecoder = BiFPNDecoder(pyramid_channels=self.fpn_num_filters[self.compound_coef])

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1 if self.seg_mode == BINARY_MODE else self.seg_classes+1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        self.classifier = Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=self.num_anchors,
                                     num_classes=self.num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef],
                                     onnx_export=False)

        if self.backbone_name == "efficientnet":
            # EfficientNet_Pytorch
            self.encoder = get_encoder(
                'efficientnet-b' + str(self.backbone_compound_coef[self.compound_coef]),
                in_channels=3,
                depth=5,
                weights='imagenet',
            )
        else:
            self.encoder = timm.create_model(self.backbone_name, pretrained=True, features_only=True, out_indices=(1,2,3,4))  # P2,P3,P4,P5

        self.anchors = Anchors(self.anchors_scales, self.anchors_ratios, anchor_scale=self.anchor_scale[self.compound_coef],
                                pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                                onnx_export=False)
    
        self.initialize_decoder(self.bifpndecoder)
        self.initialize_head(self.segmentation_head)
        self.initialize_decoder(self.bifpn)
        self.initialize_weights()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inp):
        x = inp['img']
        
        p2, p3, p4, p5 = self.encoder(x)[-4:]

        features = (p3, p4, p5)
        features = self.bifpn(features)
        p3,p4,p5,p6,p7 = features
        
        outputs = self.bifpndecoder((p2,p3,p4,p5,p6,p7))

        segmentation = self.segmentation_head(outputs)
        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(x, x.dtype)
        
        target = {
            "img": x,
            "features": features,
            "regression": regression,
            "classification": classification,
            "anchors": anchors,
            "segmentation": segmentation,
        }
        return target
    
    def postprocess(self, target):
        image = target["img"]
        features = target["features"]
        regression = target["regression"]
        classification = target["classification"]
        anchors = target["anchors"]
        seg = target["segmentation"]
        
        _, seg = torch.max(seg, dim=1)
        
        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        detection = postprocess(image,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        self.conf_thres, self.iou_thres)
        
        out = {
            "segmentation": seg,
            "detection": detection,
        }
        return out
    
    def visualize(self, batch):
        print(batch.keys())
        img_batch = batch["img"].cpu().detach()
        seg_batch = batch["segmentation"].cpu().detach()
        
        batch_size = img_batch.shape[0]
        img_batch = normalize_tensor(img_batch)

        seg_color_batch = apply_color(seg_batch)
        seg_vis_batch = overlay_images_batch(img_batch, seg_color_batch)
        
        vis = {
            "seg_vis": seg_vis_batch,
        }

        if "detection" in batch:
            det_batch = batch["detection"]
            det_vis_batch = []
            for i in range(batch_size):
                image = img_batch[i]
                det = det_batch[i]
                
                boxes = det['rois']
                scores = det['scores']
                cat_ids = np.array(det["class_ids"], dtype=int)
                cat_names = [self.obj_list[cat_id] for cat_id in cat_ids]
                captions = [f'{cat_name}: {score:.2f}' for cat_name, score in zip(cat_names, scores)]
                colors = apply_color(cat_ids + 1)
                
                det_vis = torchvision.transforms.ToPILImage()(image)
                det_vis = draw_bounding_boxes(det_vis, boxes, captions, colors)
                det_vis = torchvision.transforms.ToTensor()(det_vis)
                det_vis_batch.append(det_vis)
            
            vis["det_vis"] = det_vis_batch
        return vis

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


    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def initialize_weights(self):
        init_weights(self)
