# Copyright (c) SenseTime. All Rights Reserved.
import sys
sys.path.append("/home/guo/zpy/Mypysot")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import time
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import argparse

from mypysot.models.zpyMyNet import MySiamRPN
from mypysot.tracker.zpyMyTrack import MySiamRPNTracker
from mypysot.utils.bbox import get_axis_aligned_bbox
from mypysot.utils.model_load import load_pretrain
from mypysot.RGBDNET.data_loader import RGBDDataLoader
from mypysot.RGBDNET.net import RGBDNET
from mypysot.RGBDNET.Depth_mask import Depth_mask

class ptb_track():
    def __init__(self):
        self.Noinit = True
    def track(self, name, gt_bbox_):
        model = MySiamRPN()
        model_load_path = '/home/guo/zpy/Mypysot/mypysot/dataset/weight/model.pth'
        model = load_pretrain(model, model_load_path).cuda().eval()
        mytrack = MySiamRPNTracker(model)

        Masker = Depth_mask()
        loader = RGBDDataLoader()
        net_file = '/home/guo/zpy/Siamese-RPN-pytorch-master/dataset/weight/RGBDnet_weightsV12.2-0050000.pth.tar'
        rgbdnet = RGBDNET().cuda()
        checkpoint = torch.load(net_file)
        rgbdnet.load_state_dict(checkpoint['state_dict'])

        s = '/home/guo/zpy/Mypysot/mypysot/result/result4'

        image_file1 = '/media/guo/DataUbuntu/PTB_dataset/' + name + '/rgb/00000001.png'
        image_file2 = '/media/guo/DataUbuntu/PTB_dataset/' + name + '/depth/00000001.png'
        if not os.path.exists(image_file1):
            sys.exit(0)
        img = cv2.imread(image_file1)
        mytrack.init(img, gt_bbox_)
        img_depth = Image.open(image_file2)
        Masker.start_mask(img, img_depth, gt_bbox_)

        #track
        confidence_count = 0
        for i in range(2,5000):
            #track of the first stage
            image_file1 = '/media/guo/DataUbuntu/PTB_dataset/' + name + '/rgb/'
            image_file1 = image_file1 + str(i).zfill(8) + '.png'
            image_file2 = '/media/guo/DataUbuntu/PTB_dataset/' + name + '/depth/'
            image_file2 = image_file2 + str(i).zfill(8) + '.png'
            if not os.path.exists(image_file1):
                print('confidence_count:')
                print(confidence_count)
                break
            img_rgb = cv2.imread(image_file1)
            img_depth = Image.open(image_file2)
         
            args_mask = True
            if args_mask:
                print(Masker.isWork)
                if Masker.isWork == True:
                    img = Masker.generral_mask(img_rgb.copy(), img_depth.copy())
                    masked = 0.85
                else:
                    img = img_rgb.copy()
                    masked = 1
            else:
                img = img_rgb.copy()
                masked = 1

            state = mytrack.track(img, masked)
            bbox = np.array(state['bbox'].copy())
            bboxes = np.array(state['bbox16'].copy())
            if(state['confidence'] == 1):
                confidence_count = confidence_count + 1
            print(state['confidence'])
            print(state['best_score'])

            
            for j in range(bboxes.shape[1]):
                region = [int(i) for i in bboxes[:,j]]
                x1, y1, x2, y2 = region[0], region[1], region[0]+region[2], region[1]+region[3]
                cv2.line(img,(x1,y1),(x2,y1),(255,0,0),1)
                cv2.line(img,(x1,y2),(x2,y2),(255,0,0),1)
                cv2.line(img,(x1,y1),(x1,y2),(255,0,0),1)
                cv2.line(img,(x2,y1),(x2,y2),(255,0,0),1)
            text =  'best_score: ' + str(round(state['best_score'], 3)) + '   confidence: ' + str(state['confidence'])
            img = cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not args_mask:
                save_path = '/home/guo/zpy/Mypysot/mypysot/result/result1/siam_predict_' + str(i).zfill(8) + '.jpg'
            else:
                save_path = '/home/guo/zpy/Mypysot/mypysot/result/result3/siam_predict_' + str(i).zfill(8) + '.jpg'
            cv2.imwrite(save_path, img)
            
           
         
            #nms+合并
            if not len(bboxes.shape) == 1:
                boxes_num = bboxes.shape[1]
                proposal_16_x1 = bboxes[0,:]
                proposal_16_y1 = bboxes[1,:]
                proposal_16_x2 = bboxes[0,:] + bboxes[2,:]
                proposal_16_y2 = bboxes[1,:] + bboxes[3,:]
                areas = (proposal_16_x2 - proposal_16_x1 + 1) * (proposal_16_y2 - proposal_16_y1 + 1)
                neeSuppress = np.zeros(boxes_num, dtype=np.int)
                keep = []
                for j in range(0,boxes_num):
                    if neeSuppress[j] == 0:
                        keep.append(j)
                    w = np.maximum(0.0, np.minimum(proposal_16_x2[j], proposal_16_x2) - np.maximum(proposal_16_x1[j], proposal_16_x1) + 1)
                    h = np.maximum(0.0, np.minimum(proposal_16_y2[j], proposal_16_y2) - np.maximum(proposal_16_y1[j], proposal_16_y1) + 1)
                    inter = w * h
                    iou = inter / (areas[j] + areas - inter)
                    neeSuppress[np.where( (iou > 0.9)|(iou <0.05) )[0]] = 1
                #print(keep)
                proposal_result_x1 = min(proposal_16_x1[keep])
                proposal_result_y1 = min(proposal_16_y1[keep])
                proposal_result_x2 = max(proposal_16_x2[keep])
                proposal_result_y2 = max(proposal_16_y2[keep])
                proposal_result_cx = (proposal_result_x1 + proposal_result_x2)//2
                proposal_result_cy = (proposal_result_y1 + proposal_result_y2)//2
                proposal_result_w  = proposal_result_x2 - proposal_result_x1
                proposal_result_h  = proposal_result_y2 - proposal_result_y1
                region = [proposal_result_x1, proposal_result_y1, proposal_result_w, proposal_result_h]
            else:
                region = [bboxes[0], bboxes[1], bboxes[2], bboxes[3]]
            region = [region[0]-region[2]*0.1, region[1]-region[3]*0.1, region[2]*1.2, region[3]*1.2]
            imgw, imgh = img.shape[1], img.shape[0]
            region_nms = region.copy()
            region_nms[0] = np.clip(region[0], 0, imgw)
            region_nms[1] = np.clip(region[1], 0, imgh)
            region_nms[2] = np.clip(region[2], 0, imgw - region_nms[0])
            region_nms[3] = np.clip(region[3], 0, imgh - region_nms[1])


            #track of the second stage
            image_file_rgb= '/media/guo/DataUbuntu/PTB_dataset/' + name +'/rgb/' + str(i).zfill(8) + '.png'
            image_file_d='/media/guo/DataUbuntu/PTB_dataset/' + name + '/depth/' + str(i).zfill(8) + '.png' 
            img_path = [image_file_rgb, image_file_d]
            ret = loader.__get__(img_path = img_path, region = region_nms.copy())
            result = rgbdnet(ret['img_rgb_tensor'].cuda(), ret['img_d3_tensor'].cuda())
            result = result.cpu().detach().numpy().reshape(-1)
            x1, y1, w, h = (result[2]-result[0])*100/ret['w_resized_ratio']+region[0], \
                           (result[3]-result[1])*100/ret['h_resized_ratio']+region[1], \
                            result[0]*100/ret['w_resized_ratio'], \
                            result[1]*100/ret['h_resized_ratio']
            if     abs(w*h-bbox[2]*bbox[3]) > (bbox[2]*bbox[3])*0.1:
                region = [x1, y1, w, h]
                print('RGBDNET works!-------------------')
            else:
                region = [bbox[0], bbox[1], bbox[2], bbox[3]]

            if args_mask:
                if Masker.isWork == True:
                    Masker.get_depth(img_depth, region, state['best_score']>=0.68)
                if Masker.isWork == False and state['best_score'] >= 0.9:
                    Masker.start_mask(img_rgb, img_depth.copy(), region)

            with open(os.path.join(s, name+'.txt'), 'a+') as f2:
                if state['best_score'] > 0.2:
                    f2.write(str('{:.1f}'.format(region[0])) + ',' +
                             str('{:.1f}'.format(region[1])) + ',' +
                             str('{:.1f}'.format(region[0]+region[2])) + ',' +
                             str('{:.1f}'.format(region[1]+region[3])))
                    f2.write('\n')
                else:
                    f2.write('NaN,NaN,NaN,NaN')
                    f2.write('\n')    

            save_path = '/home/guo/zpy/Mypysot/mypysot/result/result2/rgbd_predict_' + str(i).zfill(8) + '.jpg'
            image = ret['img_rgb'].copy()
            draw = ImageDraw.Draw(image)
            x1, y1 = region[0], region[1]
            x2, y2 = region[0]+region[2], region[1]+region[3]
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='blue')

            region = bbox.copy()
            x1, y1 = region[0], region[1]
            x2, y2 = region[0]+region[2], region[1]+region[3]
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='red')

            region = region_nms
            x1, y1 = region[0], region[1]
            x2, y2 = region[0]+region[2], region[1]+region[3]
            draw.line([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], width=2, fill='yellow')
            image.save(save_path)
         
            print(i)

