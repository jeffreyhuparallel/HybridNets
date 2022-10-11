import os
import argparse

import torch
from tqdm import tqdm

from hybridnets.data import build_data_loader
from hybridnets.backbone import HybridNetsBackbone
from hybridnets.utils.utils import Params


def main(args):
    params = Params(args.config_file)
    model = HybridNetsBackbone(params)
    model.eval()
    model = model.cuda()

    dataloader = build_data_loader(params, split="val")
    for idx, data in enumerate(tqdm(dataloader)):
        imgs = data['img']
        imgs = imgs.cuda()

        with torch.no_grad():
            features, regression, classification, anchors, seg = model(imgs)
            for f in features:
                print(f.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-file", required=True, help="Path to config file"
    )
    parser.add_argument(
        "-p", "--ckpt", default=None, type=str, help="Path to checkpoint"
    )
    args = parser.parse_args()
    print(args)
    main(args)
