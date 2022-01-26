import numpy as np
import torch
import cv2
import time

from TSDM.tracker.SiamResTrack import MySiamRPNResTracker
from TSDM.tracker.SiamMobTrack import MySiamRPNMobTracker
from TSDM.tracker.SiamResForMaskTrack import MySiamRPNResForMaskTracker
from TSDM.tracker.SiamMobForMaskTrack import MySiamRPNMobForMaskTracker
from TSDM.models.SiamRPN.SiamNetRes import MySiamRPNRes
from TSDM.models.SiamRPN.SiamNetMob import MySiamRPNMob
from TSDM.tools.model_load import load_pretrain
from TSDM.models.Depth_refiner.data_loader import RGBDDataLoader
from TSDM.models.Depth_refiner.net import FuseNet
from TSDM.models.Mask_generator.Depth_mask import Depth_mask
class TSDMTracker():
    def __init__(self, SiamRes_model_path, SiamMask_model_path, Dr_model_path, image_rgb, image_depth, region):
        #M-g
        self.Masker = Depth_mask()
        self.Masker.start_mask(image_rgb, image_depth, region)
        #SiamRPN++
        model1 = MySiamRPNRes()
        model1 = load_pretrain(model1, SiamRes_model_path).cuda().eval()
        self.SiamRestracker = MySiamRPNResTracker(model1)
        self.SiamRestracker.init(image_rgb, region)
        model2 = MySiamRPNRes() #MySiamRPNRes() MySiamRPNMob()
        model2 = load_pretrain(model2, SiamMask_model_path).cuda().eval()
        self.SiamForMasktracker = MySiamRPNResForMaskTracker(model2)
        self.SiamForMasktracker.init(image_rgb, region)
        #D-r
        self.Dr_loader = RGBDDataLoader()
        self.Dr_net = FuseNet().cuda().eval()
        self.Dr_net.load_state_dict(torch.load(Dr_model_path)['state_dict'])
        #output
        self.result = {}

    def track(self, image_rgb, image_depth):
        # M-g
        if self.Masker.isWork == True:
            img_rgb = self.Masker.general_mask(image_rgb.copy(), image_depth.copy())
        else:
            img_rgb = image_rgb.copy()
        self.result['Xm'] = img_rgb

        # SiamRPN++
        if self.Masker.isWork == True:
            state = self.SiamForMasktracker.track(img_rgb)
            bbox = np.array(state['bbox'].copy())
            bboxes = np.array(state['bbox16'].copy())
            self.SiamRestracker.update_state(bbox)
        else:
            state = self.SiamRestracker.track(img_rgb)
            bbox = np.array(state['bbox'].copy())
            bboxes = np.array(state['bbox16'].copy())
            self.SiamForMasktracker.update_state(bbox)
        self.result['region_Siam'] = bbox
        self.result['score'] = state['best_score']

        #NMS+Amplification
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
        imgw, imgh = image_rgb.shape[1], image_rgb.shape[0]
        region_nms = region.copy()
        region_nms[0] = np.clip(region[0], 0, imgw)
        region_nms[1] = np.clip(region[1], 0, imgh)
        region_nms[2] = np.clip(region[2], 0, imgw - region_nms[0])
        region_nms[3] = np.clip(region[3], 0, imgh - region_nms[1])
        self.result['region_nms'] = region_nms

        #D-r
        ret = self.Dr_loader.__get__(image_rgb, image_depth, region = region_nms.copy())
        result = self.Dr_net(ret['img_rgb_tensor'].cuda(), ret['img_ddd_tensor'].cuda())
        result = result.cpu().detach().numpy().reshape(-1)
        x1, y1, w, h = (result[2]-result[0])*100/ret['w_resized_ratio']+region[0], \
                       (result[3]-result[1])*100/ret['h_resized_ratio']+region[1], \
                        result[0]*100/ret['w_resized_ratio'], \
                        result[1]*100/ret['h_resized_ratio']
        if abs(w*h-bbox[2]*bbox[3]) > (bbox[2]*bbox[3])*0.01:
            region = [x1, y1, w, h]
        else:
            region = [bbox[0], bbox[1], bbox[2], bbox[3]]
        self.result['region_Dr'] = region

        # M-g
        if self.Masker.isWork == True:
            self.Masker.get_depth(image_depth, region, state['best_score'])
        if self.Masker.isWork == False and state['best_score'] >= 0.92:
            self.Masker.start_mask(image_rgb, image_depth.copy(), region)
        
        return self.result

if __name__ == '__main__':
    print(1)
