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
from railyard.util.visualization import apply_color, overlay_images_batch, overlay_images, draw_bounding_boxes

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

def main(args):
    params = Params(args.config_file)
    obj_list = params.obj_list
    output_dir = os.path.join(params.output_dir, "demo")
    image_dir = "demo/image"
    batch_size = 1
    
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
                save_file(seg_vis, os.path.join(output_dir, f"seg_vis/{sample_name}.jpg"))

            for i in range(batch_size):
                image = img_batch[i]
                det = det_batch[i]
                
                boxes = det['rois']
                scores = det['scores']
                cat_ids = np.array(det["class_ids"], dtype=int)
                cat_names = [obj_list[cat_id] for cat_id in cat_ids]
                captions = [f'{cat_name}: {score:.2f}' for cat_name, score in zip(cat_names, scores)]
                colors = apply_color(cat_ids + 1)
                
                det_vis = torchvision.transforms.ToPILImage()(image)
                det_vis = draw_bounding_boxes(det_vis, boxes, captions, colors)
                save_file(det_vis, os.path.join(output_dir, f"det_vis/{sample_name}.png"))
            

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
