import argparse

from hybridnets.config import get_cfg
from hybridnets.data import build_data_loader
from hybridnets.autoanchor import run_anchor

def main(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    print(f"Running with config:\n{cfg}")

    train_dataloader = build_data_loader(cfg, split="train")
    anchors_scales, anchors_ratios = run_anchor(None, train_dataloader.dataset)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-file", required=True, help="Path to config file"
    )
    args = parser.parse_args()
    print(args)
    main(args)
