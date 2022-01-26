# Modifications copyright (C) 2019 <Haojie Zhao>

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import os
import cv2
import numpy as np
import torch
from siamtrack.pysot.core.config import cfg
from siamtrack.pysot.models.model_builder import ModelBuilder
from siamtrack.pysot.tracker.tracker_builder import build_tracker
from siamtrack.pysot.utils.model_load import load_pretrain

from utils import overlap_ratio
from nms import nms

import argparse
parser = argparse.ArgumentParser(description='GPU selection and SRE selection', prog='tracker')
parser.add_argument("--gpu", default=0, type=int)
args, unknown_1 = parser.parse_known_args()
# os.environ['CUDA_VISIBLE_DEVICES']= '1'
os.environ['CUDA_VISIBLE_DEVICES']= str(args.gpu)

class SiamMask():
    def __init__(self, cfg_file, snapshot):
        # load config
        cfg.merge_from_file(cfg_file)

        cur_dir = os.path.dirname(os.path.realpath(__file__))

        # create model
        self.model = ModelBuilder()
        # load model
        self.model = load_pretrain(self.model, snapshot).cuda().eval()
        # build tracker
        self.tracker = build_tracker(self.model)

    def init(self, img, box):  # [x y w h]
        self.tracker.init(img, box)
        return {'target_bbox': box, 'time': 0}

    def track(self, img, thres=0.9):
        outputs = self.tracker.track(img)

        box = np.array(outputs['bbox'])
        score = outputs['best_score']

        box_mask = np.array(outputs['polygon']).astype(np.float32).reshape(-1)

        # box_mask[2] = box_mask[2] + box_mask[0] - 1
        # box_mask[3] = box_mask[3] + box_mask[1] - 1

        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
        mask = mask.astype(np.uint8)
        mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)

        if score > thres:
            find = True
        else:
            find = False

        box[2:] = box[2:] + box[:2]  # [x y w h] to [x1 y1 x2 y2]
        box = np.array(box).astype(np.float32).reshape(-1)

        box[0::2] = np.clip(box[0::2], 0, img.shape[1]-1)
        box[1::2] = np.clip(box[1::2], 0, img.shape[0]-1)

        box_mask[0::2] = np.clip(box_mask[0::2], 0, img.shape[1]-1)
        box_mask[1::2] = np.clip(box_mask[1::2], 0, img.shape[0]-1)
        return {'box': box, 'score': score,
                'mask': mask, 'box_mask': box_mask, 'find': find}  # [x1 y1 x2 y2] score

    def track_zf(self, img, zf_list):
        outputs, outputs_mask = self.tracker.track_zf(img, zf_list=zf_list)  # [x y w h, score] [x1 y1 x2 y2, score]

        outputs[:, 2:4] = outputs[:, 2:4] + outputs[:, :2]  # [x y w h] to [x1 y1 x2 y2]

        return outputs_mask
