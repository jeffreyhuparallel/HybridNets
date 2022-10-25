import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from hybridnets.modules import build_module

from railyard.config import get_cfg
from railyard.util import cli_logo


def main(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    print(f"Running with config:\n{cfg}")

    output_dir = cfg.OUTPUT_DIR

    module = build_module(cfg)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir=output_dir,
        logger=TensorBoardLogger(output_dir),
    )
    trainer.test(module, ckpt_path=args.ckpt)


if __name__ == "__main__":
    cli_logo()
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
