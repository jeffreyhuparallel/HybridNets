import argparse
import datetime
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from tqdm import tqdm

from hybridnets.backbone import HybridNetsBackbone
from hybridnets.utils.utils import save_checkpoint, Params
from hybridnets.data import build_data_loader
from hybridnets.autoanchor import run_anchor
from hybridnets.model import ModelWithLoss

@torch.no_grad()
def val(params, model, val_dataloader, writer, step):
    model.eval()

    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    for idx, inp in enumerate(tqdm(val_dataloader)):
        inp['img'] = inp['img'].cuda()
        inp['annot'] = inp['annot'].cuda()
        inp['segmentation'] = inp['segmentation'].cuda()
        
        imgs = inp['img']
        annot = inp['annot']
        seg_annot = inp['segmentation']

        losses, out = model(inp)
        
        cls_loss = losses["cls_loss"].mean()
        reg_loss = losses["reg_loss"].mean()
        seg_loss = losses["seg_loss"].mean()
        loss = cls_loss + reg_loss + seg_loss

        loss_classification_ls.append(cls_loss.item())
        loss_regression_ls.append(reg_loss.item())
        loss_segmentation_ls.append(seg_loss.item())

    cls_loss = np.mean(loss_classification_ls)
    reg_loss = np.mean(loss_regression_ls)
    seg_loss = np.mean(loss_segmentation_ls)
    loss = cls_loss + reg_loss + seg_loss

    writer.add_scalar('val/loss', loss, step)
    writer.add_scalar('val/regression_loss', reg_loss, step)
    writer.add_scalar('val/classification_loss', cls_loss, step)
    writer.add_scalar('val/segmentation_loss', seg_loss, step)

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

    if params.need_autoanchor:
        params.anchors_scales, params.anchors_ratios = run_anchor(None, train_dataloader.dataset)

    model = HybridNetsBackbone(params)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    model = ModelWithLoss(model, params)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), params.lr)

    model.train()
    for epoch in range(params.num_epochs):
        for idx, inp in enumerate(tqdm(train_dataloader)):
            inp['img'] = inp['img'].cuda()
            inp['annot'] = inp['annot'].cuda()
            inp['segmentation'] = inp['segmentation'].cuda()
            
            imgs = inp['img']
            annot = inp['annot']
            seg_annot = inp['segmentation']

            optimizer.zero_grad(set_to_none=True)
            
            losses, out = model(inp)
            regression = out["regression"]
            classification = out["classification"]
            anchors = out["anchors"]
            segmentation = out["segmentation"]
            
            cls_loss = losses["cls_loss"].mean()
            reg_loss = losses["reg_loss"].mean()
            seg_loss = losses["seg_loss"].mean()
            
            loss = cls_loss + reg_loss + seg_loss
            loss.backward()
            optimizer.step()

            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/regression_loss', reg_loss, step)
            writer.add_scalar('train/classification_loss', cls_loss, step)
            writer.add_scalar('train/segmentation_loss', seg_loss, step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)

            step += 1

        torch.save(model.model.state_dict(), os.path.join(checkpoint_dir, f'hybridnets-d{params.compound_coef}_{epoch}.pth'))

        val(params, model, val_dataloader, writer=writer, step=step)

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
