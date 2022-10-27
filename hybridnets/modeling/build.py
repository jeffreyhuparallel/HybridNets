import torch

from .hybridnet import HybridNet

def build_model(cfg):
    model = HybridNet(cfg)
    return model