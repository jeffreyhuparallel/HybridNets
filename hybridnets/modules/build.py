from .hydra_module import HydraModule


def build_module(cfg, ckpt=None):
    module_name = cfg.MODEL.META_ARCHITECTURE
    if module_name in ["hydranet", "fcn", "vhc", "fasterrcnn"]:
        if ckpt is None:
            module = HydraModule(cfg)
        else:
            module = HydraModule.load_from_checkpoint(ckpt, cfg=cfg)
    else:
        raise Exception(f"Module name not recognized: {module_name}")
    return module
