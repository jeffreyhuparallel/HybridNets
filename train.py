import argparse
import datetime
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from backbone import HybridNetsBackbone
from utils.utils import init_weights, save_checkpoint, DataLoaderX, Params
from hybridnets.dataset import BddDataset
from hybridnets.custom_dataset import CustomDataset
from hybridnets.autoanchor import run_anchor
from hybridnets.model import ModelWithLoss
from utils.constants import MULTILABEL_MODE, MULTICLASS_MODE, BINARY_MODE

@torch.no_grad()
def val(params, model, val_generator, writer, step):
    model.eval()

    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    for iter, data in enumerate(tqdm(val_generator)):
        imgs = data['img']
        annot = data['annot']
        seg_annot = data['segmentation']

        imgs = imgs.cuda()
        annot = annot.cuda()
        seg_annot = seg_annot.cuda()

        cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                seg_annot,
                                                                                                obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        seg_loss = seg_loss.mean()
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

    checkpoint_dir = params.output_dir + f'/checkpoints/'
    summary_dir = params.output_dir + f'/tensorboard/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    writer = SummaryWriter(summary_dir)

    seg_mode = MULTILABEL_MODE if params.seg_multilabel else MULTICLASS_MODE if len(params.seg_list) > 1 else BINARY_MODE

    train_dataset = BddDataset(
        params=params,
        is_train=True,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=False
    )

    training_generator = DataLoaderX(
        train_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        collate_fn=BddDataset.collate_fn
    )

    valid_dataset = BddDataset(
        params=params,
        is_train=False,
        inputsize=params.model['image_size'],
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=params.mean, std=params.std
            )
        ]),
        seg_mode=seg_mode,
        debug=False
    )

    val_generator = DataLoaderX(
        valid_dataset,
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True,
        collate_fn=BddDataset.collate_fn
    )

    if params.need_autoanchor:
        params.anchors_scales, params.anchors_ratios = run_anchor(None, train_dataset)

    model = HybridNetsBackbone(num_classes=len(params.obj_list), compound_coef=params.compound_coef,
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=params.backbone_name,
                               seg_mode=seg_mode)
    if args.ckpt is None:
        print('[Info] initializing weights...')
        init_weights(model)
    else:
        try:
            state_dict = torch.load(args.ckpt)
            state_dict = state_dict.get('model', state_dict)
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    print('[Info] Successfully!!!')

    model = ModelWithLoss(model, debug=False)
    model = model.to(memory_format=torch.channels_last)
    model = model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    step = 0
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(params.num_epochs):
        epoch_loss = []
        for iter, data in enumerate(tqdm(training_generator)):
            imgs = data['img']
            annot = data['annot']
            seg_annot = data['segmentation']

            imgs = imgs.to(device="cuda", memory_format=torch.channels_last)
            annot = annot.cuda()
            seg_annot = seg_annot.cuda()

            optimizer.zero_grad(set_to_none=True)
            
            cls_loss, reg_loss, seg_loss, regression, classification, anchors, segmentation = model(imgs, annot,
                                                                                                    seg_annot,
                                                                                                    obj_list=params.obj_list)
            cls_loss = cls_loss.mean()
            reg_loss = reg_loss.mean()
            seg_loss = seg_loss.mean()
            loss = cls_loss + reg_loss + seg_loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss))

            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/regression_loss', reg_loss, step)
            writer.add_scalar('train/classification_loss', cls_loss, step)
            writer.add_scalar('train/segmentation_loss', seg_loss, step)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], step)

            step += 1

        scheduler.step(np.mean(epoch_loss))

        save_checkpoint(model, checkpoint_dir, f'hybridnets-d{params.compound_coef}_{epoch}.pth')
        val(params, model, val_generator, writer=writer, step=step)

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
