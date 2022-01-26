import os
from pickle import FALSE


import numpy as np
import cv2 as cv
import random
import torch
import time
import vot
import sys

from siamtrack.interface_siammask import SiamMask
from rtmdnet.interface_rtmd_local import RTMD as RTMD_L
from centernet.interface_cdet import CDetector
from flownet2.interface_flownet2 import Flow2CS
from pytracking.vot20_utils import *
from pytracking.Refine_module_bcm_complete import Refine_module_bcm
from utils import overlap_ratio
import config_path as p

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''
function import
'''
from vot import Rectangle
import sys
sys.path.append('/home/yangjinyu/rgbd_tracker/evaluation_tool/')
from sre_tmp import Robustness
import os 
import argparse
parser = argparse.ArgumentParser(description='GPU selection and SRE selection', prog='tracker')
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--sre", default=0, type=int)
args, unknown_1 = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print('sre: '.format(args.sre))
import logging
log_path = os.path.join('/data1/yjy/rgbd_benchmark/all_benchmark/', 'normal.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='a', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
'''
    function import
'''

class Detector:
    def __init__(self, vot_flag):
        seed = 10
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.thres_local_rtmd = 0

        self.rtmdL = RTMD_L(p.rtmd_model)
        self.flow = Flow2CS(p.flow_model)

        self.siam = SiamMask(p.siammask_cfg, p.siammask_snap)

        self.cdet = CDetector(p.cdet_model, p.cdet_res)

        self.alpha = Refine_module_bcm(p.refineA, p.refineB)

        self.vot_flag = vot_flag

    def init(self, image, box):
        
        x, y, w, h = box
      
        self.rtmdL.init(image, box)
      
        self.alpha.initialize(cv.cvtColor(im, cv.COLOR_BGR2RGB), np.array(box))
     
        self.siam.init(image, box)

        self.final_box = np.array([x, y, w, h])
        self.final_last = self.final_box.copy()
        self.final_score = 1
        self.md_score = 0
        self.frame_i = 0

        self.lost_count = 0
        self.target_lost = False
        self.im_last = im.copy()
        

    def track(self, im, im_d, title):
        self.frame_i += 1
        # im_show = im.copy()

        flow = self.flow.flow(im, self.im_last)
        flow = np.abs(flow).max(0)
        flow = flow > 1
        # flow_d = im_d * flow

        cdet_box, cdet_score = self.cdet.detect_read(self.frame_i + 1, title)  # [x y x y]
        cdet_box = cdet_box[cdet_score > 0.0]

        tmp = []
        for bb in cdet_box:
            b = bb.astype(int)
            a = flow[b[1]:b[3], b[0]:b[2]].sum() / (b[3] - b[1]) / (b[2] - b[0])
            if a > 0.6:
                tmp.append(bb.tolist())
        flow_box = np.array(tmp)

        # === local search ===
        # ====================
        if not self.target_lost:
            self.lost_count = 0

            siam_out = self.siam.track(im)
            siam_box = siam_out['box']  # [x y x y]
            siam_score = siam_out['score']

            if siam_score > 0.01:
                rtmd_target = siam_box.copy()
                rtmd_target[2:] = rtmd_target[2:] - rtmd_target[:2] + 1  # [x y w h]
                rtmd_box, rtmd_score, rtmd_find \
                    = self.rtmdL.inference(im, rtmd_target, proposals=siam_box, thres=self.thres_local_rtmd)
                if rtmd_find:
                    iou = overlap_ratio(rtmd_box, cdet_box)
                    neg_s = cdet_box[iou < 0.3]
                else:
                    neg_s = np.array([])
                self.rtmdL.collect(im, rtmd_box, rtmd_find, neg_s)
                self.rtmdL.update(rtmd_find, self.frame_i)

                if rtmd_find:
                    self.final_box = siam_box.copy()  # [x y x y]
                    self.final_box[2:] = self.final_box[2:] - self.final_box[:2] + 1  # [x y w h]
                    self.final_last = self.final_box.copy()
                    final_score = siam_score

                    self.target_lost = False
                else:
                    self.target_lost = True
            else:
                self.target_lost = True

        # === global search ===
        # =====================
        if self.target_lost:
            self.lost_count += 1
            if flow_box.size > 0:
                fina_area = self.final_box[2] * self.final_box[3]
                tmp_cdet = flow_box[:, 2:] - flow_box[:, :2]
                tmp_cdet = tmp_cdet[:, 0] * tmp_cdet[:, 1]
                tmp_idx = np.logical_and(tmp_cdet > fina_area * 0.3, tmp_cdet < fina_area * 3)
                proposal = flow_box[tmp_idx]
            else:
                proposal = np.array([])

            if proposal.size > 0:
                tmp_depth = []
                tmp_depth_avg = []
                for bb in proposal:
                    b = bb.astype(int)
                    tmp_d = im_d[b[1]:b[3], b[0]:b[2]]
                    counts = np.bincount(tmp_d.reshape(-1))
                    tmp_depth.append(np.argmax(counts))
                    tmp_depth_avg.append(tmp_d.mean())
                tmp_depth = np.array(tmp_depth)

                proposal = proposal[tmp_depth < 50]

            if proposal.size > 0:
                tmp_box = self.final_last.copy()  # [x y w h]
                tmp_box[2:] = tmp_box[2:] + tmp_box[:2] - 1
                iou = overlap_ratio(proposal, tmp_box)
                if self.lost_count < 50:
                    tmp_thres = 0.65
                else:
                    tmp_thres = 0.0
                if iou.max() > tmp_thres:
                    self.final_box = proposal[iou.argmax()]  # [x y x y]
                    self.final_box[2:] = self.final_box[2:] - self.final_box[:2]
                    # final_last = final_box.copy()
                    final_score = 0.5

                    self.target_lost = False
                    rest_box = self.final_box.copy()
                    self.siam.tracker.center_pos = np.array([rest_box[0] + rest_box[2] / 2,
                                                             rest_box[1] + rest_box[3] / 2])
                    self.siam.tracker.size = np.array([rest_box[2], rest_box[3]])

                else:
                    rtmd_target = self.final_last.copy()  # [x y w h]
                    rtmd_box, rtmd_score, rtmd_find \
                        = self.rtmdL.eval(im, rtmd_target, proposals=proposal, thres=self.thres_local_rtmd)
                    if rtmd_find:

                        self.final_box = rtmd_box[0]  # [x y x y]
                        self.final_box[2:] = self.final_box[2:] - self.final_box[:2]
                        self.final_last = self.final_box.copy()
                        final_score = 0.5

                        self.target_lost = False
                        rest_box = self.final_box.copy()
                        self.siam.tracker.center_pos = np.array([rest_box[0] + rest_box[2] / 2,
                                                                 rest_box[1] + rest_box[3] / 2])
                        self.siam.tracker.size = np.array([rest_box[2], rest_box[3]])
                    else:
                        self.final_box = self.final_last.copy()  # [x y w h]
                        final_score = np.nan
            else:
                self.final_box = self.final_last.copy()  # [x y w h]
                final_score = np.nan

        # # == debug ==
        # b = self.final_box.astype(int)
        # cv.rectangle(im_show, (b[0], b[1]), (b[0]+b[2]-1, b[1]+b[3]-1), (0, 255, 0), 2)
        #
        # cv.imshow('', im_show)
        # cv.waitKey(1)

        x, y, w, h = self.final_box
        b1 = self.alpha.refine(
            cv.cvtColor(im, cv.COLOR_BGR2RGB),
            np.array([x, y, w, h]),
            use_selector=True)['bbox_report']
        x, y, w, h = b1

        confidence = np.around(final_score, decimals=4)

        if self.vot_flag:
            return vot.Rectangle(x, y, w, h), confidence
        else:
            return np.around([x, y, w, h], decimals=4), confidence

print("VOT_FLAG")
VOT_FLAG = True
# VOT_FLAG = True
print("VOT_FLAG: ", VOT_FLAG)
if VOT_FLAG:

    #seq_path = '/home/space/Documents/experiment/VOT20RGBD/'
    # seq_path = '/home/yangjinyu/votRGBD2019/workspace'
    handle = vot.VOT("rectangle", 'rgbd')
    selection = handle.region()
    '''
    robustness sre
    '''
    shift = Robustness(selection)
    shift.functions(args.sre)
    region_shift = shift.region
    selection = Rectangle(x=region_shift[0], y=region_shift[1], width=region_shift[2], height=region_shift[3])
    '''
        logging
    '''
    colorimage, depthimage = handle.frame()
    logging.info('tracker: TSDM sre_type:{} gpu:{} image_file1:{}'.format(args.sre, args.gpu, colorimage))
    if not colorimage:
        sys.exit(0)

    im = cv.imread(colorimage)
    im_d = cv.imread(depthimage, cv.IMREAD_GRAYSCALE)
    # im = cv.imread(os.path.join(seq_path, colorimage))
    # im_d = cv.imread(os.path.join(seq_path, depthimage), cv.IMREAD_GRAYSCALE)
    selection = np.array([selection.x, selection.y, selection.width, selection.height])
    
    D = Detector(VOT_FLAG)
    print("Detector vot flag ini come in")
    D.init(im, selection)
    print("loop come in")
    while True:
        colorimage, depthimage = handle.frame()
        print(colorimage)
        if not colorimage:
            break

        im = cv.imread(colorimage)
        im_d = cv.imread(depthimage, cv.IMREAD_GRAYSCALE)
        # im = cv.imread(os.path.join(seq_path, colorimage))
        # im_d = cv.imread(os.path.join(seq_path, depthimage), cv.IMREAD_GRAYSCALE)

        region, confidence = D.track(im, im_d, colorimage.split('/')[-3])

        handle.report(region, confidence)

        time.sleep(0.01)
else:
    tracker_name = 'CLGS_D'
    vids = os.listdir('/home/yangjinyu/rgbd_tracker/benchmark_workspace/depthtrack_workspace/sequences/')
    # vids = os.listdir('/home/yangjinyu/votRGBD2019/workspace/sequences')
    # vids = os.listdir('/home/yangjinyu/cvpr2022/metric/supplementary/vot_workspace/sequences/')
    vids.sort()
    vids.remove('list.txt')
    vids.remove('list.txt.lack1')
    # VOT2019-RGBD.json not found
    # vids.remove('VOT2019-RGBD.json')
    len_vids = len(vids)
    print(vids, len(vids))
    results_dir = '/home/yangjinyu/rgbd_tracker/benchmark_workspace/depthtrack_workspace/results/'
    # results_dir = '/home/yangjinyu/votRGBD2019/workspace/results'
    # results_dir = '/home/yangjinyu/cvpr2022/metric/supplementary/vot_workspace/results/'

    for vid, title in enumerate(vids[0:len_vids]):
        # path = '/home/yangjinyu/votRGBD2019/workspace/sequences/{}'.format(title)
        # path = '/home/yangjinyu/cvpr2022/metric/supplementary/vot_workspace/sequences/{}'.format(title)
        path = '/home/yangjinyu/rgbd_tracker/benchmark_workspace/depthtrack_workspace/sequences/{}'.format(title)
        num = len(os.listdir(os.path.join(path, 'color')))

        base_results_path = '{}/{}/rgbd-unsupervised/{}'.format(results_dir, tracker_name, title)
        results_path = '{}/{}_001.txt'.format(base_results_path, title)

        if os.path.isfile(results_path):
            print('{:0>2d} Tracker: {} ,  Sequence: {}'.format(vid, tracker_name, title))
            continue

        gt_list = np.loadtxt(os.path.join(path, 'groundtruth.txt'), delimiter=',')
        D = Detector(VOT_FLAG)

        box_list = []
        confidence_list = []
        time_list = []

        try:
            for frame_i in range(0, num):
                if not os.path.exists(os.path.join(path, 'color/{:0>8d}.jpg'.format(frame_i + 1))):
                    im = cv.imread(os.path.join(path, 'color/{:0>8d}.png'.format(frame_i + 1)))
                else:
                    im = cv.imread(os.path.join(path, 'color/{:0>8d}.jpg'.format(frame_i + 1)))
                im_d = cv.imread(os.path.join(path, 'depth/{:0>8d}.png'.format(frame_i + 1)), cv.IMREAD_GRAYSCALE)

                tic = time.time()
                if frame_i == 0:
                    x, y, w, h = gt_list[frame_i]
                    D.init(im, [x, y, w, h])

                    box_list.append('1\n')
                    confidence_list.append('\n')
                    time_list.append('{:.6f}\n'.format(time.time() - tic))
                else:
                    b1, final_score = D.track(im, im_d, title)

                    box_list.append('{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                        b1[0], b1[1], b1[2], b1[3]))
                    confidence_list.append('{:.6f}\n'.format(final_score))
                    time_list.append('{:.6f}\n'.format(time.time() - tic))

            print('{:0>2d} Tracker: {} ,  Sequence: {}'.format(vid, tracker_name, title))

            vid_path = os.path.join(results_dir, tracker_name, 'rgbd-unsupervised', '{:s}'.format(title))
            if not os.path.exists(vid_path):
                os.makedirs(vid_path)

            # save box
            if len(box_list) > 0:
                with open(os.path.join(vid_path, '{:s}_001.txt'.format(title)), 'w') as f:
                    f.writelines(box_list)
            # save confidence
            if len(confidence_list) > 0:
                with open(os.path.join(vid_path, '{:s}_001_confidence.value'.format(title)), 'w') as f:
                    f.writelines(confidence_list)
            # save time
            if len(time_list) > 0:
                with open(os.path.join(vid_path, '{:s}_time.value'.format(title)), 'w') as f:
                    f.writelines(time_list)

        except Exception as e:
            print(e)
            continue
