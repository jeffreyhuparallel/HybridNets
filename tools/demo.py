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

from hybridnets.config import get_cfg
from hybridnets.data import build_transform
from hybridnets.modeling import build_model

from railyard.util import read_file, save_file, get_file_names
from railyard.util.categories import lookup_category_list
from railyard.util.visualization import apply_color, overlay_images_batch, overlay_images, draw_bounding_boxes, normalize_tensor

def main(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    print(f"Running with config:\n{cfg}")

    dataset_name = cfg.DATASETS.PREDICT[0]
    output_dir = cfg.OUTPUT_DIR
    
    batch_size = 1
    obj_list = lookup_category_list(dataset_name, include_background=False)
    
    image_dir = "demo/image"
    file_names = get_file_names(image_dir, ext=".jpg")
    sample_names = [os.path.splitext(fn)[0] for fn in file_names]
    transform = build_transform(cfg, split="predict")
    
    model = build_model(cfg)
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
            out = model.postprocess(target)
            out.update({k: v for k, v in inp.items() if k not in out})
            vis = model.visualize(out)
            
            for k, v in vis.items():
                image = torchvision.transforms.ToPILImage()(v[0])
                save_file(image, os.path.join(output_dir, f"demo/{k}/{sample_name}.jpg"))
            

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
