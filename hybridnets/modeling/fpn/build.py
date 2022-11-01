import torch.nn as nn

from .bifpn import BiFPN

def build_fpn(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone_coef = cfg.MODEL.BACKBONE.COEFFICIENT
    
    fpn_num_cells_coef = [3, 4, 5, 6, 7, 7, 8, 8, 8]
    fpn_num_channels_coef = [64, 88, 112, 160, 224, 288, 384, 384, 384]
    if "efficientnet" in backbone_name:
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
    elif "regnet" in backbone_name:
        conv_channel_coef = {
            3: [64, 160, 384], # regnetx_004
        }
    else:
        raise Exception(f"Backbone not recognized: {backbone_name}")
    
    fpn_num_cells = fpn_num_cells_coef[backbone_coef]
    fpn_num_channels = fpn_num_channels_coef[backbone_coef]
    conv_channels = conv_channel_coef[backbone_coef]
    
    fpn_cells = [BiFPN(fpn_num_channels,
                conv_channels,
                first_time=(i == 0),
                attention=backbone_coef < 6,
                use_p8=backbone_coef > 7,
                onnx_export=False)
            for i in range(fpn_num_cells)]
    fpn = nn.Sequential(*fpn_cells)
    return fpn, fpn_num_channels
