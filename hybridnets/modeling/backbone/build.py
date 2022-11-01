import timm

from .encoders import get_encoder

def build_hybrid_backbone(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    
    if "efficientnet" in backbone_name:
        backbone = get_encoder(
            backbone_name,
            in_channels=3,
            depth=5,
            weights='imagenet',
        )
    elif "regnet" in backbone_name:
        backbone = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(1,2,3,4))
    else:
        raise Exception(f"Backbone not recognized: {backbone_name}")
    return backbone