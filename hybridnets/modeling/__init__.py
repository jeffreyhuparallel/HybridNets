from .backbone import HybridNetsBackbone

def build_model(cfg):
    return HybridNetsBackbone(cfg)