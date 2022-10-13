from typing import Union, List, Optional, Tuple
import time
import torch
import torchvision
import cv2
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import argparse
from tqdm import tqdm

from hybridnets.config import Params
from hybridnets.backbone import HybridNetsBackbone

from railyard.util import read_file, save_file, get_file_names
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
    
    batch_size = 1
    image_dir = "demo/image"
    file_names = get_file_names(image_dir, ext=".jpg")
    sample_names = [os.path.splitext(fn)[0] for fn in file_names]
    
    transform = transforms.Compose([
        transforms.Resize((384, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.mean, std=params.std),
    ])
    
    model = HybridNetsBackbone(params)
    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'))

    model.requires_grad_(False)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        for sample_name in tqdm(sample_names):
            image = read_file(os.path.join(image_dir, f"{sample_name}.jpg"))
            x = transform(image)
            x = torch.unsqueeze(x, dim=0)
            x = x.to(torch.float32).cuda()
            inp = {"img": x}
            
            target = model(inp)
            out = model.postprocess(inp, target)
            
            img_batch = normalize_tensor(inp["img"]).cpu().detach()
            seg_batch = out["segmentation"].cpu().detach()
            det_batch = out["detection"]

            seg_color_batch = apply_color(seg_batch)
            seg_vis_batch = overlay_images_batch(img_batch, seg_color_batch)
            for i in range(batch_size):
                seg_vis = torchvision.transforms.ToPILImage()(seg_vis_batch[i])
                save_file(seg_vis, os.path.join(output_dir, f"demo_result/seg_vis/{sample_name}.jpg"))

            for i in range(batch_size):
                det = det_batch[i]
                
                boxes = det['rois']
                cat_names = [obj_list[cat_id] for cat_id in det['class_ids']]
                scores = det['scores']
                labels = [f'{cat_name}: {score:.2f}' for cat_name, score in zip(cat_names, scores)]
                colors = apply_color(det['class_ids'] + 1).tolist()

                det_vis = np.array(img_batch[i] * 255, dtype=np.uint8).transpose((1,2,0))
                det_vis = visualize_bboxes(det_vis, boxes, labels, colors=colors)
                det_vis = torchvision.transforms.ToPILImage()(det_vis)
                save_file(det_vis, os.path.join(output_dir, f"demo_result/det_vis/{sample_name}.png"))
            

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
