# Modifications copyright (C) 2019 <Haojie Zhao>

import os
import numpy as np
import torch
from siamtrack.pysot.core.config import cfg
from siamtrack.pysot.models.model_builder import ModelBuilder
from siamtrack.pysot.tracker.tracker_builder import build_tracker
from siamtrack.pysot.utils.model_load import load_pretrain

from utils import overlap_ratio
from nms import nms


class SiamRPN():
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

        if score > thres:
            find = True
        else:
            find = False

        box[2:] = box[2:] + box[:2]  # [x y w h] to [x1 y1 x2 y2]
        box = np.array(box).astype(np.float32).reshape(-1)

        return {'box': box, 'score': score, 'flag': find}  # [x1 y1 x2 y2] score

    def track_zf(self, img, zf_list):
        outputs = self.tracker.track_zf(img, zf_list=zf_list)  # [x y w h, score]

        outputs[:, 2:4] = outputs[:, 2:4] + outputs[:, :2]  # [x y w h] to [x1 y1 x2 y2]

        return outputs
