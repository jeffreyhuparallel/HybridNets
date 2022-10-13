import torch
from torch import nn
import timm

from hybridnets.encoders import get_encoder
from hybridnets.model import BiFPN, Regressor, Classifier, BiFPNDecoder
from hybridnets.model import SegmentationHead
from hybridnets.utils.utils import Anchors, init_weights
from hybridnets.utils.constants import *

class HybridNetsBackbone(nn.Module):
    def __init__(self, params):
        super(HybridNetsBackbone, self).__init__()
        self.compound_coef = params.compound_coef
        self.num_classes = len(params.obj_list)
        self.seg_classes = len(params.seg_list)
        self.seg_mode = params.seg_mode
        self.backbone_name = params.backbone_name
        self.anchors_scales = eval(params.anchors_scales)
        self.anchors_ratios = eval(params.anchors_ratios)
        self.num_scales = len(self.anchors_scales)

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
            "features": features,
            "regression": regression,
            "classification": classification,
            "anchors": anchors,
            "segmentation": segmentation,
        }
        return target
    
    def postprocess(self, inp, target):
        out = {}
        return out

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
