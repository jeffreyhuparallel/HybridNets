from .hydra_module import HydraModule


def build_module(cfg, ckpt=None):
    if ckpt is None:
        module = HydraModule(cfg)
    else:
        module = HydraModule.load_from_checkpoint(ckpt, cfg=cfg)
    return module
