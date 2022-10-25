import torch

from .hybridnet import HybridNet

def build_model(cfg, pretrained=False):
    model = HybridNet(cfg)
    if pretrained:
        if cfg.MODEL.BACKBONE.NAME == "efficientnet":
            weights_path = "./output/hybridnet_saved/checkpoints/hybridnets-d3_199.pth"
        else:
            weights_path = "./output/regnet_saved/checkpoints/hybridnets-d3_199.pth"
        model.load_state_dict(torch.load(weights_path), strict=False)
        print(f"Loaded weights from {weights_path}")
    return model