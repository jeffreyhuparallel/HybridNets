import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

import itertools
              
class Anchors(pl.LightningModule):

    def __init__(self, scales, ratios, size, anchor_scale=4., pyramid_levels=None):
        super().__init__()
        self.scales = np.array(scales)
        self.ratios = ratios
        self.size = size
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        self.strides = [2 ** x for x in self.pyramid_levels]
        
        anchor_boxes = self.make_anchor_boxes()
        anchor_boxes = torch.from_numpy(anchor_boxes)
        anchor_boxes = anchor_boxes.unsqueeze(0)
        self.register_buffer("anchor_boxes", anchor_boxes)

    def get_anchor_boxes(self):
        return self.anchor_boxes
    
    def make_anchor_boxes(self):
        w, h = self.size
        boxes_all = []
        for stride in self.strides:
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                if w % stride != 0 or h % stride != 0:
                    raise ValueError(f'input size must be divided by the stride. ({w},{h})')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                x = np.arange(stride / 2, w, stride)
                y = np.arange(stride / 2, h, stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                # y1,x1,y2,x2
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = anchor_boxes.astype(dtype=np.float32)
        return anchor_boxes
    