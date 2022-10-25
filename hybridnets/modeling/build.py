import torch

from .backbone import HybridNetsBackbone

def build_model(cfg, pretrained=False):
    model = HybridNetsBackbone(cfg)
    if pretrained:
        weights_path = "./output/hybridnet_saved/checkpoints/hybridnets-d3_199.pth"
        model.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path}")
    return model