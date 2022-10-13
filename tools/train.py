import argparse
import datetime
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm

from hybridnets.config import Params
from hybridnets.backbone import HybridNetsBackbone
from hybridnets.data import build_data_loader
from hybridnets.criterion import build_criterion

@torch.no_grad()
def val(params, model, criterion, val_dataloader, writer, step):
    model.eval()

    losses_all = defaultdict(list)
    for idx, inp in enumerate(tqdm(val_dataloader)):
        inp['img'] = inp['img'].cuda()
        inp['annot'] = inp['annot'].cuda()
        inp['segmentation'] = inp['segmentation'].cuda()

        out = model(inp)
        losses = criterion(inp, out)
        
        for k, v in losses.items():
            losses_all[k].append(v)
        
    for k,v in losses.items():
        writer.add_scalar(f"val/{k}", torch.mean(v), step)

    model.train()

def main(args):
    params = Params(args.config_file)
    epoch = 0
    step = 0

    checkpoint_dir = params.output_dir + f'/checkpoints/'
    summary_dir = params.output_dir + f'/tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    writer = SummaryWriter(summary_dir)

    train_dataloader = build_data_loader(params, split="train")
    val_dataloader = build_data_loader(params, split="val")

    criterion = build_criterion(params)
    model = HybridNetsBackbone(params)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    optimizer = torch.optim.AdamW(model.parameters(), params.lr)

    model = model.cuda()
    model.train()
    for epoch in range(params.num_epochs):
        for idx, inp in enumerate(tqdm(train_dataloader)):
            for k, v in inp.items():
                inp[k] = v.cuda() if torch.is_tensor(v) else v

            optimizer.zero_grad(set_to_none=True)
            
            target = model(inp)
            losses = criterion(inp, target)
            
            losses["loss"].backward()
            optimizer.step()
            
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)
            for k, v in losses.items():
                writer.add_scalar(f'train/{k}', v, step)

            step += 1

        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch={epoch}.pth'))

        val(params, model, criterion, val_dataloader, writer=writer, step=step)

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
