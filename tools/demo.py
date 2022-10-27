import os
import argparse
import torch
import torchvision
from tqdm import tqdm

from railyard.config import get_cfg
from railyard.dataclasses import Sample
from railyard.data.transforms import build_transform
from hybridnets.modeling import build_model
from railyard.util import read_file, save_file, get_file_names
from railyard.util.visualization import normalize_tensor

def main(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    print(f"Running with config:\n{cfg}")

    image_dir = "demo/image"
    output_dir = cfg.OUTPUT_DIR
    
    file_names = get_file_names(image_dir, ext=".jpg")
    sample_names = [os.path.splitext(fn)[0] for fn in file_names]
    
    transform = build_transform(cfg, split="predict")
    model = build_model(cfg)
    model = model.cuda()
    model.eval()

    with torch.no_grad():
        for sample_name in tqdm(sample_names):
            image = read_file(os.path.join(image_dir, f"{sample_name}.jpg"))
            sample = Sample(image=image)
            
            inp = transform(sample)
            inp["image"] = torch.unsqueeze(inp["image"], dim=0).cuda()
            
            target = model(inp)
            out = model.postprocess(target)
            
            # Visualize
            vis = {"main_vis": normalize_tensor(inp["image"])}
            model.visualize(out, vis)
            
            for k, v in vis.items():
                image = torchvision.transforms.ToPILImage()(v[0])
                save_file(image, os.path.join(output_dir, f"demo/{k}/{sample_name}.jpg"))
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config-file", required=True, help="Path to config file"
    )
    args = parser.parse_args()
    print(args)
    main(args)
