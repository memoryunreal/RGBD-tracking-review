import time
import cv2
import torch
import numpy as np
import argparse
import sys
import os

sys.path.append("./")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from tracker.TSDMTrack import TSDMTracker
from tools.bbox import get_axis_aligned_bbox

parser = argparse.ArgumentParser(description='TSDM is running')
parser.add_argument('--data_dir', default='data_and_result/test_data/')
parser.add_argument('--sequence', default=1, type=int)

name = ['start/']# not be used
gt_bbox = [[1,1,1,1]]
name.append('teapot/') #1
gt_bbox.append([318.67,133.0,119.67,97.0 ]) 
name.append('XMG_outside/') #2
gt_bbox.append([338.23,164.54,98.909,54.001])
name.append('box_darkroom_noocc_1/') #3
gt_bbox.append([219.2,173.88,143.6,95.4])

args = parser.parse_args()
name = name[args.sequence]
gt_bbox = gt_bbox[args.sequence]


#init
args.data_dir
image_file1 = args.data_dir + name + 'color/00000001.jpg'
image_file2 = args.data_dir + name + 'depth/00000001.png'
if not os.path.exists(image_file1):
    sys.exit(0)
image_rgb = cv2.imread(image_file1)
image_depth = cv2.imread(image_file2, -1)
SiamRes_dir = 'data_and_result/weight/modelRes.pth' # for not retraining core
SiamMask_dir = 'data_and_result/weight/Res20.pth' # for retraining core
Dr_dir = 'data_and_result/weight/High-Low-two.pth.tar' # for D-r, an information fusion network 
tracker = TSDMTracker(SiamRes_dir, SiamMask_dir, Dr_dir, image_rgb, image_depth, gt_bbox)

#track
sumTime = 0
for i in range(2,5000):
    #track of the first stage
    image_file1 = args.data_dir + name + 'color/'
    image_file1 = image_file1 + str(i).zfill(8) + '.jpg'
    image_file2 = args.data_dir + name + 'depth/'
    image_file2 = image_file2 + str(i).zfill(8) + '.png'
    image_rgb = cv2.imread(image_file1)
    image_depth = cv2.imread(image_file2, -1)
    if not os.path.exists(image_file1):
        print('Avg. speed:')
        print(i/(sumTime+0.01))
        break

    start = time.perf_counter()
    state = tracker.track(image_rgb, image_depth)
    region_Siam, region_nms, region_Dr = state['region_Siam'], state['region_nms'], state['region_Dr']

    eachTime = time.perf_counter() - start
    sumTime += eachTime
    print("Using time:",eachTime)
    
    save_path = 'data_and_result/result/TSDM_' + str(i).zfill(8) + '.jpg'
    img = state['Xm']
    region = region_Siam
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(255,0,0),1)
    cv2.line(img,(x1,y2),(x2,y2),(255,0,0),1)
    cv2.line(img,(x1,y1),(x1,y2),(255,0,0),1)
    cv2.line(img,(x2,y1),(x2,y2),(255,0,0),1)

    region = region_nms
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(0,255,255),1)
    cv2.line(img,(x1,y2),(x2,y2),(0,255,255),1)
    cv2.line(img,(x1,y1),(x1,y2),(0,255,255),1)
    cv2.line(img,(x2,y1),(x2,y2),(0,255,255),1)

    region = region_Dr
    x1, y1 = int(region[0]), int(region[1])
    x2, y2 = int(region[0]+region[2]), int(region[1]+region[3])
    cv2.line(img,(x1,y1),(x2,y1),(0,0,255),1)
    cv2.line(img,(x1,y2),(x2,y2),(0,0,255),1)
    cv2.line(img,(x1,y1),(x1,y2),(0,0,255),1)
    cv2.line(img,(x2,y1),(x2,y2),(0,0,255),1)

    text =  'target score: ' + str(round(state['score'], 3))
    img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite(save_path, img)
    
    print(i)

