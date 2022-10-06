import torch
import numpy as np
import argparse
from tqdm.autonotebook import tqdm
import os

from utils import smp_metrics
from utils.utils import ConfusionMatrix, postprocess, scale_coords, process_batch, ap_per_class, fitness, \
    save_checkpoint, DataLoaderX, BBoxTransform, ClipBoxes, boolean_string, Params
from backbone import HybridNetsBackbone
from hybridnets.dataset import BddDataset
from hybridnets.custom_dataset import CustomDataset
from torchvision import transforms
import torch.nn.functional as F
from hybridnets.model import ModelWithLoss
from utils.constants import *


@torch.no_grad()
def val(model, val_generator, params, opt, seg_mode, is_training, **kwargs):
    model.eval()

    optimizer = kwargs.get('optimizer', None)
    writer = kwargs.get('writer', None)
    epoch = kwargs.get('epoch', 0)
    step = kwargs.get('step', 0)
    best_fitness = kwargs.get('best_fitness', 0)
    best_loss = kwargs.get('best_loss', 0)
    best_epoch = kwargs.get('best_epoch', 0)

    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    stats, ap, ap_class = [], [], []
    iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()
    names = {i: v for i, v in enumerate(params.obj_list)}
    nc = len(names)
    ncs = 1 if seg_mode == BINARY_MODE else len(params.seg_list) + 1
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    s_seg = ' ' * (15 + 11 * 8)
    s = ('%-15s' + '%-11s' * 8) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95', 'mIoU', 'mAcc')
    for i in range(len(params.seg_list)):
            s_seg += '%-33s' % params.seg_list[i]
            s += ('%-11s' * 3) % ('mIoU', 'IoU', 'Acc')
    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    iou_ls = [[] for _ in range(ncs)]
    acc_ls = [[] for _ in range(ncs)]
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    val_loader = tqdm(val_generator, ascii=True)
    for iter, data in enumerate(val_loader):
        imgs = data['img']
        annot = data['annot']
        seg_annot = data['segmentation']
        filenames = data['filenames']
        shapes = data['shapes']

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
        if loss == 0 or not torch.isfinite(loss):
            continue

        loss_classification_ls.append(cls_loss.item())
        loss_regression_ls.append(reg_loss.item())
        loss_segmentation_ls.append(seg_loss.item())

    cls_loss = np.mean(loss_classification_ls)
    reg_loss = np.mean(loss_regression_ls)
    seg_loss = np.mean(loss_segmentation_ls)
    loss = cls_loss + reg_loss + seg_loss

    print(
        'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
            epoch, opt.num_epochs if is_training else 0, cls_loss, reg_loss, seg_loss, loss))
    if is_training:
        writer.add_scalars('Loss', {'val': loss}, step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
        writer.add_scalars('Segmentation_loss', {'val': seg_loss}, step)

    # if not calculating map, save by best loss
    if is_training and loss + opt.es_min_delta < best_loss:
        best_loss = loss
        best_epoch = epoch

        save_checkpoint(model, opt.saved_path, f'hybridnets-d{opt.compound_coef}_{epoch}_{step}_best.pth')

    # Early stopping
    if is_training and epoch - best_epoch > opt.es_patience > 0:
        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
        exit(0)

    model.train()
    return (best_fitness, best_loss, best_epoch) if is_training else 0
