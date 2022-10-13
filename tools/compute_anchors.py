import argparse

from hybridnets.utils.utils import Params
from hybridnets.data import build_data_loader
from hybridnets.autoanchor import run_anchor

def main(args):
    params = Params(args.config_file)

    train_dataloader = build_data_loader(params, split="train")
    anchors_scales, anchors_ratios = run_anchor(None, train_dataloader.dataset)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-file", required=True, help="Path to config file"
    )
    args = parser.parse_args()
    print(args)
    main(args)
