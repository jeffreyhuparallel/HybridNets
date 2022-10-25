import os

import matplotlib
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

from ..criterion import build_criterion
from ..data import build_data_loader
from railyard.evaluation import build_evaluator
from ..modeling import build_model
from railyard.util import save_file


class HydraModule(pl.LightningModule):
    """Hydranet Module.

    For training multitask networks.
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.cfg = cfg
        self.learning_rate = cfg.SOLVER.BASE_LR
        self.output_dir = cfg.OUTPUT_DIR

        self.net = build_model(cfg, pretrained=False)
        self.criterion = build_criterion(cfg)
        self.evaluator = build_evaluator(cfg)

    def forward(self, x):
        return self.net(x)

    def training_step(self, inp, batch_idx):
        target = self(inp)
        losses = self.criterion(inp, target)
        for k, v in losses.items():
            self.log(f"train/{k}", v)

        # Visualize
        if batch_idx == 0:
            out = self.net.postprocess(target)
            self.log_visualizations(inp, out, split="train")
        return losses["loss"]

    def validation_step(self, inp, batch_idx):
        target = self(inp)
        losses = self.criterion(inp, target)
        for k, v in losses.items():
            self.log(f"val/{k}", v, on_epoch=True)

        # Visualize
        if batch_idx == 0:
            out = self.net.postprocess(target)
            self.log_visualizations(inp, out, split="val")
        return losses["loss"]

    def test_step(self, inp, batch_idx):
        target = self(inp)
        out = self.net.postprocess(target)

        self.evaluator.update(inp, out)

    def on_test_epoch_end(self):
        metrics, metrics_vis = self.evaluator.compute()
        save_file(metrics, os.path.join(self.output_dir, "metrics.json"))

    def predict_step(self, inp, batch_idx):
        target = self(inp)
        out = self.net.postprocess(target)
        self.save_predictions(inp, out, split="predict")
        self.save_visualizations(inp, out, split="predict")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[80, 120], gamma=0.1)
        return [opt], [sch]

    def save_predictions(self, inp, out, split="train"):
        dataset = self.get_dataloader(split).dataset
        sample_names = [dataset.get_sample_name(idx) for idx in inp["idx"]]
        for i, sample_name in enumerate(sample_names):
            main = {}
            for k, v in out.items():
                obj = v[i].cpu().detach().numpy()
                if "segmentation" not in k:
                    main[k] = obj.tolist()
                else:
                    obj = obj.astype(np.uint8)
                    save_file(
                        obj,
                        os.path.join(self.output_dir, f"{k}/{sample_name}.png"),
                    )
            save_file(
                main,
                os.path.join(self.output_dir, f"main/{sample_name}.json"),
            )

    def save_visualizations(self, inp, out, split="train"):
        out.update({k: v for k, v in inp.items() if k not in out})
        vis_pr = self.net.visualize(out)

        dataset = self.get_dataloader(split).dataset
        sample_names = [dataset.get_sample_name(idx) for idx in inp["idx"]]
        for i, sample_name in enumerate(sample_names):
            for k, v in vis_pr.items():
                image = torchvision.transforms.ToPILImage()(v[i])
                save_file(
                    image,
                    os.path.join(self.output_dir, f"{k}/{sample_name}.jpg"),
                )

    def log_visualizations(self, inp, out, split="train"):
        out.update({k: v for k, v in inp.items() if k not in out})
        vis_gt = self.net.visualize(inp)
        vis_pr = self.net.visualize(out)
        self.log_images(vis_gt, postfix=f"/gt/{split}")
        self.log_images(vis_pr, postfix=f"/pr/{split}")

    def log_images(self, vis, postfix=""):
        for k, v in vis.items():
            if type(v) == matplotlib.figure.Figure:
                self.logger.experiment.add_figure(f"{k}{postfix}", v, self.global_step)
            else:
                grid = torchvision.utils.make_grid(v, nrow=4)
                self.logger.experiment.add_image(
                    f"{k}{postfix}", grid, self.global_step
                )

    def train_dataloader(self):
        return build_data_loader(self.cfg, split="train")

    def val_dataloader(self):
        return build_data_loader(self.cfg, split="val")

    def test_dataloader(self):
        return build_data_loader(self.cfg, split="test")

    def predict_dataloader(self):
        return build_data_loader(self.cfg, split="predict")

    def get_dataloader(self, split):
        if split == "train":
            return self.trainer.train_dataloaders[0]
        elif split == "val":
            return self.trainer.val_dataloaders[0]
        elif split == "test":
            return self.trainer.test_dataloaders[0]
        elif split == "predict":
            return self.trainer.predict_dataloaders[0]
        else:
            raise Exception(f"Split not recognized: {split}")
