# Copyright (c) SenseTime. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import numpy as np
import vot
import sys
import time

'''Refine module & Pytracking base trackers'''
import os
from pytracking.evaluation import Tracker
from pytracking.Refine_module_bcm_complete import Refine_module_bcm
'''other utils'''
from pytracking.vot20_utils import *
torch.set_num_threads(1)
torch.cuda.set_device(0) # set GPU id
'''使用dimp50_vot19参数文件'''
'''2020.4.8 修复了RGB和BGR的bug'''
class DIMP_Alpha(object):
    def __init__(self,threshold=0.65):
        self.THRES = threshold
        '''create tracker'''
        '''DIMP'''
        tracker_info = Tracker('dimp', 'dimp50_vot19', None)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        params.visdom_info = {'use_visdom': False, 'server': '127.0.0.1', 'port': 8097}
        self.dimp = tracker_info.tracker_class(params)
        '''Alpha-Refine'''
        project_path = '/home/space/Documents/code/Alpha_Refine/Alpha-Refine-shared'
        refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/SEbcm/')
        refine_model_name = 'SEbcm'
        refine_path = os.path.join(refine_root, refine_model_name)
        self.alpha = Refine_module_bcm(
        '/home/space/Documents/code/Alpha_Refine/SEbcmnet_ep0040.pth.tar',
        '/home/space/Documents/code/Alpha_Refine/Branch_Selector_ep0030.pth.tar')
    def initialize(self, img_RGB, region):
        # region = rect_from_mask(mask)
        self.H, self.W, _ = img_RGB.shape
        gt_bbox_np = np.array(region).astype(np.float32)
        '''Initialize dimp for specific video'''
        gt_bbox_torch = torch.from_numpy(gt_bbox_np)
        init_info = {}
        init_info['init_bbox'] = gt_bbox_torch
        _ = self.dimp.initialize(img_RGB, init_info)
        '''initilize refinement module for specific video'''
        self.alpha.initialize(img_RGB, np.array(gt_bbox_np))
    def track(self, img_RGB):
        '''TRACK'''
        '''base tracker'''
        outputs = self.dimp.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        '''Step1: Post-Process'''
        x1, y1, w, h = pred_bbox
        # add boundary and min size limit
        x1, y1, x2, y2 = bbox_clip(x1, y1, x1 + w, y1 + h, (self.H, self.W))
        w = x2 - x1
        h = y2 - y1
        new_pos = torch.from_numpy(np.array([y1 + h / 2, x1 + w / 2]).astype(np.float32))
        new_target_sz = torch.from_numpy(np.array([h, w]).astype(np.float32))
        new_scale = torch.sqrt(new_target_sz.prod() / self.dimp.base_target_sz.prod())
        ##### update
        self.dimp.pos = new_pos.clone()
        self.dimp.target_sz = new_target_sz
        self.dimp.target_scale = new_scale
        bbox_new = [x1,y1,w,h]
        '''Step2: Mask report'''
        pred_mask = self.alpha.get_mask(img_RGB, np.array(bbox_new))
        bbox_new = self.alpha.refine(
            img_RGB, np.array(bbox_new), mode='mask', use_selector=False)['bbox_report']
        final_mask = (pred_mask > self.THRES).astype(np.uint8)
        return bbox_new, final_mask



if __name__ == '__main__':
    tracker_name = 'baseline_mask_reset_mdL_reflow'
    vids = os.listdir('/data1/Dataset/VOT/RGBD19')
    vids.sort()
    vids.remove('list.txt')
    vids.remove('VOT2019-RGBD.json')

    results_dir = '../VOT2019-RGBD'

    for vid, title in enumerate(vids[0:]):
        title = 'box_darkroom_noocc_7'
        path = '/home/space/data/VOT2019-RGBD/{}'.format(title)
        num = len(os.listdir(os.path.join(path, 'color')))

        base_results_path = '{}/{}/rgbd-unsupervised/{}'.format(results_dir, tracker_name, title)
        results_path = '{}/{}_001.txt'.format(base_results_path, title)
        scores_path = '{}/{}_001_confidence.value'.format(base_results_path, title)
        times_path = '{}/{}_time.txt'.format(base_results_path, title)
        lost_path = '{}/{}_lost.txt'.format(base_results_path, title)

        if os.path.isfile(results_path):
            print('{:0>2d} Tracker: {} ,  Sequence: {}'.format(vid, tracker_name, title))
            continue

        gt_list = np.loadtxt(os.path.join(path, 'groundtruth.txt'), delimiter=',')
        tracker = DIMP_Alpha(threshold=0.45)

        out_box = []
        out_score = []
        out_score_md = []
        out_score_md2 = []
        tic = time.time()

        lost_frame = []
        for frame_i in range(0, num):

            x, y, w, h = gt_list[frame_i]
            im = cv2.imread(os.path.join(path, 'color/{:0>8d}.jpg'.format(frame_i + 1)))
            # im_d = cv.imread(os.path.join(path, 'depth/{:0>8d}.png'.format(frame_i + 1)), cv.IMREAD_GRAYSCALE)
            # im_show = im.copy()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if frame_i == 0:
                tracker.initialize(im, [x, y, w, h])
            else:
                b1, m = tracker.track(im)

                b1 = tracker.alpha.refine(
                    im, np.array([x, y, w, h]), mode='mask', use_selector=False)['bbox_report']

                b = np.array(b1).astype(int)
                im = im.astype(float)
                im[:,:,2] = im[:,:,2] * 0.1 + m*255 * 0.9
                im = im.astype('uint8')
                cv2.rectangle(im, (b[0], b[1]), (b[0]+b[2]-1, b[1]+b[3]-1), (0, 255, 0), 2)

                cv2.imshow('', im)
                cv2.waitKey(1)

