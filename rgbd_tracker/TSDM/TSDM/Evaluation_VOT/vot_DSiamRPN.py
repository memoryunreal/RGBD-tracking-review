#coding=utf-8
#!/usr/bin/python
import vot

#'''
import sys
from vot import Rectangle
# sys.path.append("/home/guo/zpy/Mypysot")
sys.path.append("/home/yangjinyu/rgbd_tracker/TSDM")
from os.path import realpath, dirname, join
# del os.environ['MKL_NUM_THREADS']

import time
import cv2
import torch
import numpy as np

# from mypysot.tracker.TSDMTrack import TSDMTracker
# from mypysot.tools.bbox import get_axis_aligned_bbox
from TSDM.tracker.TSDMTrack import TSDMTracker
from TSDM.tools.bbox import get_axis_aligned_bbox
'''
    function import
'''
import sys
sys.path.append('/home/yangjinyu/rgbd_tracker/evaluation_tool/')
from sre_tmp import Robustness
import os 
import argparse
parser = argparse.ArgumentParser(description='GPU selection and SRE selection')
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--sre", default=0, type=int)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print('sre: '.format(args.sre))
import logging
log_path = os.path.join('/data1/yjy/rgbd_benchmark/all_benchmark', 'robustness.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='a', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
'''
    function import
'''
print("vot_DSiamRPN come in")
# start to track
handle = vot.VOT("rectangle",'rgbd')
region = handle.region()

'''
    robustness sre
'''
shift = Robustness(region)
shift.functions(args.sre)
region_shift = shift.region
region = Rectangle(x=region_shift[0], y=region_shift[1], width=region_shift[2], height=region_shift[3])
'''
    logging
'''
image_file1, image_file2 = handle.frame()

logging.info('tracker: TSDM sre_type:{} gpu:{} image_file1:{}'.format(args.sre, args.gpu, image_file1))

print("image_file1: ", image_file1)
if not image_file1:
    sys.exit(0)

image_rgb = cv2.imread(image_file1)
image_depth = cv2.imread(image_file2, -1)
# SiamRes_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/modelMob.pth' #modelRes
# SiamMask_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/Mob20.pth' #Res20.pth'
# Dr_dir = '/home/guo/zpy/Mypysot/mypysot/data_and_result/weight/High-Low-two.pth.tar'
SiamRes_dir = '/home/yangjinyu/rgbd_tracker/TSDM/TSDM/data_and_result/weight/modelRes.pth' #modelRes
SiamMask_dir = '/home/yangjinyu/rgbd_tracker/TSDM/TSDM/data_and_result/weight/Res20.pth' #Res20.pth'
Dr_dir = '/home/yangjinyu/rgbd_tracker/TSDM/TSDM/data_and_result/weight/High-Low-two.pth.tar'
tracker = TSDMTracker(SiamRes_dir, SiamMask_dir, Dr_dir, image_rgb, image_depth, region)
print("TSDMloop in")
#track
while True:
    image_file1, image_file2 = handle.frame()
    if not image_file1:
        break
    image_rgb = cv2.imread(image_file1)
    image_depth = cv2.imread(image_file2, -1)
    state = tracker.track(image_rgb, image_depth)
    region_Dr, score = state['region_Siam'], state['score'] #state['region_Dr']
    if score > 0.3:
        score = 1
    else:
        score = 0

    handle.report(Rectangle(region_Dr[0],region_Dr[1],region_Dr[2],region_Dr[3]), score)
'''
handle = vot.VOT("rectangle",'rgbd')
region = handle.region()
image_file1, image_file2 = handle.frame()
while True:
    handle.report(Rectangle(2,2,1,1), 1)
'''


