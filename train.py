import argparse
import datetime
import os
import traceback

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torchvision import transforms
from tqdm.autonotebook import tqdm

from val import val
from backbone import HybridNetsBackbone
from utils.utils import get_last_weights, init_weights, boolean_string, \
    save_checkpoint, DataLoaderX, Params
from hybridnets.dataset import BddDataset
from hybridnets.custom_dataset import CustomDataset
from hybridnets.autoanchor import run_anchor
from hybridnets.model import ModelWithLoss
from utils.constants import *
from collections import OrderedDict
from torchinfo import summary


def get_args():
    parser = argparse.ArgumentParser('HybridNets: End-to-End Perception Network - DatVu')
    parser.add_argument('-p', '--project', type=str, default='bdd100k', help='Project file that contains parameters')
    parser.add_argument('-bb', '--backbone', type=str, help='Use timm to create another backbone replacing efficientnet. '
                                                            'https://github.com/rwightman/pytorch-image-models')
    parser.add_argument('-c', '--compound_coef', type=int, default=3, help='Coefficient of efficientnet backbone')
    parser.add_argument('-n', '--num_workers', type=int, default=8, help='Num_workers of dataloader')
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Number of images per batch among all devices')
    parser.add_argument('--freeze_backbone', type=boolean_string, default=False,
                        help='Freeze encoder and neck (effnet and bifpn)')
    parser.add_argument('--freeze_det', type=boolean_string, default=False,
                        help='Freeze detection head')
    parser.add_argument('--freeze_seg', type=boolean_string, default=False,
                        help='Freeze segmentation head')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='Select optimizer for training, '
                                                                   'suggest using \'adamw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=5000, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which '
                             'training will be stopped. Set to 0 to disable this technique')
    parser.add_argument('--data_path', type=str, default='datasets/', help='The root folder of dataset')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='Whether to load weights from a checkpoint, set None to initialize,'
                             'set \'last\' to load last checkpoint')
    args = parser.parse_args()
    return args

def train(opt):
    params = Params(f'projects/{opt.project}.yml')

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
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
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
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=params.pin_memory,
        collate_fn=BddDataset.collate_fn
    )

    if params.need_autoanchor:
        params.anchors_scales, params.anchors_ratios = run_anchor(None, train_dataset)

    model = HybridNetsBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                               ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                               seg_classes=len(params.seg_list), backbone_name=opt.backbone,
                               seg_mode=seg_mode)

    # load last weights
    ckpt = {}
    if opt.load_weights:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(checkpoint_dir)

        try:
            ckpt = torch.load(weights_path)
            model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
    else:
        print('[Info] initializing weights...')
        init_weights(model)

    print('[Info] Successfully!!!')

    if opt.freeze_backbone:
        model.encoder.requires_grad_(False)
        model.bifpn.requires_grad_(False)
        print('[Info] freezed backbone')

    if opt.freeze_det:
        model.regressor.requires_grad_(False)
        model.classifier.requires_grad_(False)
        model.anchors.requires_grad_(False)
        print('[Info] freezed detection head')

    if opt.freeze_seg:
        model.bifpndecoder.requires_grad_(False)
        model.segmentation_head.requires_grad_(False)
        print('[Info] freezed segmentation head')

    model = ModelWithLoss(model, debug=False)
    model = model.to(memory_format=torch.channels_last)
    model = model.cuda()

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)
    if opt.load_weights is not None and ckpt.get('optimizer', None):
        optimizer.load_state_dict(ckpt['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    last_step = ckpt['step'] if opt.load_weights is not None and ckpt.get('step', None) else 0
    best_fitness = ckpt['best_fitness'] if opt.load_weights is not None and ckpt.get('best_fitness', None) else 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)
    for epoch in range(opt.num_epochs):
        epoch_loss = []
        progress_bar = tqdm(training_generator, ascii=True)
        for iter, data in enumerate(progress_bar):
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
            cls_loss = cls_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")
            reg_loss = reg_loss.mean() if not opt.freeze_det else torch.tensor(0, device="cuda")
            seg_loss = seg_loss.mean() if not opt.freeze_seg else torch.tensor(0, device="cuda")
            loss = cls_loss + reg_loss + seg_loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss))

            progress_bar.set_description(
                'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Seg loss: {:.5f}. Total loss: {:.5f}'.format(
                    step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                    reg_loss.item(), seg_loss.item(), loss.item()))
            writer.add_scalar('train/loss', loss, step)
            writer.add_scalar('train/regression_loss', reg_loss, step)
            writer.add_scalar('train/classification_loss', cls_loss, step)
            writer.add_scalar('train/segmentation_loss', seg_loss, step)

            # log learning_rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('learning_rate', current_lr, step)

            step += 1

            if step % opt.save_interval == 0 and step > 0:
                save_checkpoint(model, checkpoint_dir, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}.pth')
                print('checkpoint...')

        scheduler.step(np.mean(epoch_loss))

        if epoch % opt.val_interval == 0:
            val(model, val_generator, params, opt, seg_mode, writer=writer, epoch=epoch, step=step)


if __name__ == '__main__':
    opt = get_args()
    train(opt)
