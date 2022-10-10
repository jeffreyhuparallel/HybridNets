import time
import torch
from torch.backends import cudnn
import cv2
import numpy as np
from glob import glob
import os
from torchvision import transforms
import argparse
from collections import OrderedDict
from torch.nn import functional as F

from hybridnets.backbone import HybridNetsBackbone
from hybridnets.utils.constants import MULTILABEL_MODE, MULTICLASS_MODE, BINARY_MODE
from hybridnets.utils.plot import STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
from hybridnets.utils.utils import letterbox, scale_coords, postprocess, BBoxTransform, ClipBoxes, restricted_float, \
    boolean_string, Params

def main(args):
    params = Params(args.config_file)
    source = "demo/image"
    output = os.path.join(params.output_dir, "demo_result")
    os.makedirs(output, exist_ok=True)
    
    img_path = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')

    color_list_seg = {}
    for seg_class in params.seg_list:
        color_list_seg[seg_class] = list(np.random.choice(range(256), size=3))
    compound_coef = params.compound_coef
    seg_mode = params.seg_mode
    input_imgs = []
    shapes = []
    det_only_imgs = []

    anchors_ratios = params.anchors_ratios
    anchors_scales = params.anchors_scales

    threshold = 0.25
    iou_threshold = 0.3

    cudnn.fastest = True
    cudnn.benchmark = True

    obj_list = params.obj_list
    seg_list = params.seg_list

    color_list = standard_to_bgr(STANDARD_COLORS)
    ori_imgs = [cv2.imread(i, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION) for i in img_path]
    ori_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in ori_imgs]
    ori_imgs = [cv2.resize(i, (640,384)) for i in ori_imgs]
    print(f"FOUND {len(ori_imgs)} IMAGES")
    resized_shape = params.model['image_size']
    if isinstance(resized_shape, list):
        resized_shape = max(resized_shape)
    normalize = transforms.Normalize(
        mean=params.mean, std=params.std
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    for ori_img in ori_imgs:
        h0, w0 = ori_img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        input_img = cv2.resize(ori_img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_AREA)
        h, w = input_img.shape[:2]

        (input_img, _), ratio, pad = letterbox((input_img, None), resized_shape, auto=True,
                                                scaleup=False)

        input_imgs.append(input_img)
        shapes.append(((h0, w0), ((h / h0, w / w0), pad)))  # for COCO mAP rescaling

    x = torch.stack([transform(fi).cuda() for fi in input_imgs], 0)
    x = x.to(torch.float32)
    
    # model = HybridNetsBackbone(compound_coef=compound_coef, num_classes=len(obj_list), ratios=eval(anchors_ratios),
    #                         scales=eval(anchors_scales), seg_classes=len(seg_list), backbone_name=params.backbone_name,
    #                         seg_mode=seg_mode)
    model = HybridNetsBackbone(params)
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))

    model.requires_grad_(False)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        features, regression, classification, anchors, seg = model(x)

        seg_mask_list = []
        # (B, C, W, H) -> (B, W, H)
        _, seg_mask = torch.max(seg, 1)
        seg_mask_list.append(seg_mask)
        # (B, W, H) -> (W, H)
        for i in range(seg.size(0)):
            for seg_class_index, seg_mask in enumerate(seg_mask_list):
                seg_mask_ = seg_mask[i].squeeze().cpu().numpy()
                pad_h = int(shapes[i][1][1][1])
                pad_w = int(shapes[i][1][1][0])
                seg_mask_ = seg_mask_[pad_h:seg_mask_.shape[0]-pad_h, pad_w:seg_mask_.shape[1]-pad_w]
                seg_mask_ = cv2.resize(seg_mask_, dsize=shapes[i][0][::-1], interpolation=cv2.INTER_NEAREST)
                color_seg = np.zeros((seg_mask_.shape[0], seg_mask_.shape[1], 3), dtype=np.uint8)
                for index, seg_class in enumerate(params.seg_list):
                        color_seg[seg_mask_ == index+1] = color_list_seg[seg_class]
                color_seg = color_seg[..., ::-1]  # RGB -> BGR

                color_mask = np.mean(color_seg, 2)  # (H, W, C) -> (H, W), check if any pixel is not background
                # prepare to show det on 2 different imgs
                # (with and without seg) -> (full and det_only)
                det_only_imgs.append(ori_imgs[i].copy())
                seg_img = ori_imgs[i].copy() if seg_mode == MULTILABEL_MODE else ori_imgs[i]  # do not work on original images if MULTILABEL_MODE
                seg_img[color_mask != 0] = seg_img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
                seg_img = seg_img.astype(np.uint8)
                seg_filename = f'{output}/{i}_{params.seg_list[seg_class_index]}_seg.jpg' if seg_mode == MULTILABEL_MODE else \
                            f'{output}/{i}_seg.jpg'
                cv2.imwrite(seg_filename, cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR))

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

        for i in range(len(ori_imgs)):
            out[i]['rois'] = scale_coords(ori_imgs[i][:2], out[i]['rois'], shapes[i][0], shapes[i][1])
            for j in range(len(out[i]['rois'])):
                x1, y1, x2, y2 = out[i]['rois'][j].astype(int)
                obj = obj_list[out[i]['class_ids'][j]]
                score = float(out[i]['scores'][j])
                plot_one_box(ori_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                            color=color_list[get_index_label(obj, obj_list)])
                plot_one_box(det_only_imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                                color=color_list[get_index_label(obj, obj_list)])

            cv2.imwrite(f'{output}/{i}_det.jpg',  cv2.cvtColor(det_only_imgs[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f'{output}/{i}.jpg', cv2.cvtColor(ori_imgs[i], cv2.COLOR_RGB2BGR))

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
