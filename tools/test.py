import torch
import numpy as np
import argparse
from tqdm import tqdm
import os
from torchvision import transforms
import torch.nn.functional as F

from hybridnets.data import build_data_loader
from hybridnets.utils import smp_metrics
from hybridnets.utils.utils import postprocess, scale_coords, process_batch, ap_per_class, fitness, BBoxTransform, ClipBoxes, Params
from hybridnets.backbone import HybridNetsBackbone
from hybridnets.model import ModelWithLoss


@torch.no_grad()
def test(model, val_dataloader, params):
    model.eval()

    loss_regression_ls = []
    loss_classification_ls = []
    loss_segmentation_ls = []
    stats, ap, ap_class = [], [], []
    iou_thresholds = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()
    names = {i: v for i, v in enumerate(params.obj_list)}
    nc = len(names)
    ncs = len(params.seg_list) + 1
    seen = 0
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

    for iter, data in enumerate(tqdm(val_dataloader)):
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

        out = postprocess(imgs.detach(),
                            torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regression.detach(),
                            classification.detach(),
                            regressBoxes, clipBoxes,
                            params.conf_thres, params.iou_thres)  # 0.5, 0.3

        for i in range(annot.size(0)):
            seen += 1
            labels = annot[i]
            labels = labels[labels[:, 4] != -1]

            ou = out[i]
            nl = len(labels)

            pred = np.column_stack([ou['rois'], ou['scores']])
            pred = np.column_stack([pred, ou['class_ids']])
            pred = torch.from_numpy(pred).cuda()

            target_class = labels[:, 4].tolist() if nl else []  # target class

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                    torch.Tensor(), torch.Tensor(), target_class))
                continue

            if nl:
                pred[:, :4] = scale_coords(imgs[i][1:], pred[:, :4], shapes[i][0], shapes[i][1])
                labels = scale_coords(imgs[i][1:], labels, shapes[i][0], shapes[i][1])
                correct = process_batch(pred, labels, iou_thresholds)
            else:
                correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))
    
        segmentation = segmentation.log_softmax(dim=1).exp()
        _, segmentation = torch.max(segmentation, 1)  # (bs, C, H, W) -> (bs, H, W)

        tp_seg, fp_seg, fn_seg, tn_seg = smp_metrics.get_stats(segmentation, seg_annot, mode=params.seg_mode,
                                                                threshold=None, num_classes=ncs)
        iou = smp_metrics.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
        acc = smp_metrics.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')

        for i in range(ncs):
            iou_ls[i].append(iou.T[i].detach().cpu().numpy())
            acc_ls[i].append(acc.T[i].detach().cpu().numpy())

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
        'Val. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Segmentation loss: {:1.5f}. Total loss: {:1.5f}'.format(cls_loss, reg_loss, seg_loss, loss))

    for i in range(ncs):
        iou_ls[i] = np.concatenate(iou_ls[i])
        acc_ls[i] = np.concatenate(acc_ls[i])
    iou_score = np.mean(iou_ls)
    acc_score = np.mean(acc_ls)

    miou_ls = []
    for i in range(len(params.seg_list)):
        miou_ls.append(np.mean( (iou_ls[0] + iou_ls[i+1]) / 2))

    for i in range(ncs):
        iou_ls[i] = np.mean(iou_ls[i])
        acc_ls[i] = np.mean(acc_ls[i])

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    ap50 = None

    # Compute metrics
    if len(stats) and stats[0].any():
        p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=params.output_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    print(s_seg)
    print(s)
    pf = ('%-15s' + '%-11i' * 2 + '%-11.3g' * 6) % ('all', seen, nt.sum(), mp, mr, map50, map, iou_score, acc_score)
    for i in range(len(params.seg_list)):
        tmp = i+1
        pf += ('%-11.3g' * 3) % (miou_ls[i], iou_ls[tmp], acc_ls[tmp])
    print(pf)

    # Print results per class
    if nc > 1 and len(stats):
        pf = '%-15s' + '%-11i' * 2 + '%-11.3g' * 4
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    results = (mp, mr, map50, map, iou_score, acc_score, loss)
    fi = fitness(
        np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95, iou, acc, loss ]

    model.train()

def main(args):
    params = Params(args.config_file)

    model = HybridNetsBackbone(params)
    if args.ckpt is not None:
        model.load_state_dict(torch.load(args.ckpt))

    model = ModelWithLoss(model, debug=False)
    model.requires_grad_(False)
    model.cuda()

    val_dataloader = build_data_loader(params, split="val")
    test(model, val_dataloader, params)


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
