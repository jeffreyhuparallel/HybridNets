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
def val(model, val_generator, params, seg_mode, **kwargs):
    model.eval()

    writer = kwargs.get('writer', None)
    epoch = kwargs.get('epoch', 0)
    step = kwargs.get('step', 0)

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

    print(
        'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(
            epoch, params.num_epochs, cls_loss, reg_loss, seg_loss, loss))
    writer.add_scalar('val/loss', loss, step)
    writer.add_scalar('val/regression_loss', reg_loss, step)
    writer.add_scalar('val/classification_loss', cls_loss, step)
    writer.add_scalar('val/segmentation_loss', seg_loss, step)

    model.train()
