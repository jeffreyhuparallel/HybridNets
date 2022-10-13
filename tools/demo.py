from typing import Union, List, Optional, Tuple
import time
import torch
import torchvision
import cv2
from PIL import Image
import numpy as np
from glob import glob
import os
from torchvision import transforms
import argparse
from collections import OrderedDict
from torch.nn import functional as F

from hybridnets.config import Params
from hybridnets.backbone import HybridNetsBackbone
from hybridnets.utils.utils import BBoxTransform, ClipBoxes, postprocess

from railyard.util import read_file, save_file
from railyard.util.visualization import apply_color, overlay_images_batch, overlay_images

def normalize_tensor(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = True,
) -> torch.Tensor:
    """
    Normalize a tensor of images by shifting images to the range (0, 1), 
    by the min and max values specified by ``value_range``.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``True``.
    Returns:
        grid (Tensor): the tensor containing normalized images.
    """
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if value_range is not None and not isinstance(value_range, tuple):
        raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)
    return tensor


def visualize_bbox(img, bbox, label, color=(255,255,255), thickness=1):
    """Visualizes a single bounding box on the image"""
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = int(bbox[2])
    y_max = int(bbox[3])
    color = tuple(color)
   
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
    ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)    
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), color, -1)
    cv2.putText(
        img,
        text=label,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.3, 
        color=(0, 0, 0), 
        lineType=cv2.LINE_AA,
    )
    return img

def visualize_bboxes(image, bboxes, labels, colors):
    img = image.copy()
    bboxes = bboxes[::-1]
    labels = labels[::-1]
    colors = colors[::-1]
    for bbox, label, color in zip(bboxes, labels, colors):
        img = visualize_bbox(img, bbox, label, color=color)
    return img

def main(args):
    params = Params(args.config_file)
    obj_list = params.obj_list
    output_dir = params.output_dir
    
    threshold = 0.25
    iou_threshold = 0.3
    source = "demo/image"
    img_path = glob(f'{source}/*.jpg') + glob(f'{source}/*.png')
    ori_imgs = [read_file(i) for i in img_path]
    print(f"FOUND {len(ori_imgs)} IMAGES")
        
    transform = transforms.Compose([
        transforms.Resize((384, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.mean, std=params.std),
    ])

    x = torch.stack([transform(img).cuda() for img in ori_imgs], 0)
    x = x.to(torch.float32)
    
    model = HybridNetsBackbone(params)
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))

    model.requires_grad_(False)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        inp = {"img": x}
        out = model(inp)
        
        features = out["features"]
        regression = out["regression"]
        classification = out["classification"]
        anchors = out["anchors"]
        seg = out["segmentation"]
        
        _, seg = torch.max(seg, dim=1)
        
        img_batch = normalize_tensor(x)
        img_batch = img_batch.cpu().detach()
        seg_batch = seg.cpu().detach()

        seg_color_batch = apply_color(seg_batch)
        seg_vis_batch = overlay_images_batch(img_batch, seg_color_batch)
        for i in range(x.shape[0]):
            seg_vis = torchvision.transforms.ToPILImage()(seg_vis_batch[i])
            save_file(seg_vis, os.path.join(output_dir, f"demo_result/{i}_seg.jpg"))

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()
        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)
        det_vis_batch = []
        for i in range(x.shape[0]):
            boxes = out[i]['rois']
            cat_names = [obj_list[cat_id] for cat_id in out[i]['class_ids']]
            scores = out[i]['scores']
            labels = [f'{cat_name}: {score:.2f}' for cat_name, score in zip(cat_names, scores)]
            colors = apply_color(out[i]['class_ids'] + 1).tolist()

            det_vis = np.array(img_batch[i] * 255, dtype=np.uint8).transpose((1,2,0))
            det_vis = visualize_bboxes(det_vis, boxes, labels, colors=colors)
            det_vis = torchvision.transforms.ToPILImage()(det_vis)
            save_file(det_vis, os.path.join(output_dir, f"demo_result/{i}_det.png"))
            

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
