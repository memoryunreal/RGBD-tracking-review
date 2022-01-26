import os
import sys
import torch
import numpy as np
import cv2
import torch.nn as nn
from pytracking.utils.loading import load_network
from ltr.data.processing_utils_SE import sample_target_SE, transform_image_to_crop_SE, map_mask_back
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)


def mask2bbox_new(mask, MASK_THRESHOLD=0.5, VOT=False):
    target_mask = (mask > MASK_THRESHOLD)
    target_mask = target_mask.astype(np.uint8)
    if cv2.__version__[-5] == '4':
        contours, _ = cv2.findContours(target_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(target_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    if len(contours) != 0 and np.max(cnt_area) > 100:
        contour = contours[np.argmax(cnt_area)]
        polygon = contour.reshape(-1, 2)
        if VOT:
            '''rotated bounding box, 8 points'''
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        else:
            '''axis-aligned box, 4 points'''
            prbox = cv2.boundingRect(polygon)
        return np.array(prbox).astype(np.float)
    else:  # empty mask
        return  np.zeros((1,))


def delta2bbox(delta):
    bbox_cxcywh = delta.clone()
    '''以位于中心的大小为(128,128)的框为基准'''
    bbox_cxcywh[:, :2] = 128.0 + delta[:, :2] * 128.0  # 中心偏移
    bbox_cxcywh[:, 2:] = 128.0 * torch.exp(delta[:, 2:])  # 宽高修正
    bbox_xywh = bbox_cxcywh.clone()
    bbox_xywh[:, :2] = bbox_cxcywh[:, :2] - 0.5 * bbox_cxcywh[:, 2:]
    return bbox_xywh
'''使用在裁剪区域时使用BORDER_CONSTANT,而不是BORDER_REPLICATE. 注意在训练和测试时应该保持统一'''
'''2020.2.24 新版本： 兼容之前开发的两代脚本，既可以使用selector来选择mode,也可以直接输入mode'''
class Refine_module_bcm(object):
    def __init__(self, refine_net_dir, selector_dir, search_factor=2.0, input_sz=256):
        self.refine_network = self.get_network(refine_net_dir)
        self.branch_selector = self.get_network(selector_dir)
        self.search_factor = search_factor
        self.input_sz = input_sz
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
        self.branch_list = ['bbox','corner','mask']
    def initialize(self, frame1, bbox1):
        '''
        :param frame1: cv array (H,W,3) RGB
        :param bbox1: ndarray (4,)
        :return:
        '''
        '''Step1: get cropped patch(tensor)'''
        patch1, h_f, w_f = sample_target_SE(frame1, bbox1, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        patch1_tensor = self.img_preprocess(patch1)
        '''Step2: get GT's cooridinate on the cropped patch(tensor)'''
        crop_sz = torch.Tensor((self.input_sz, self.input_sz))
        bbox1_tensor = self.gt_preprocess(bbox1) # (4,)
        bbox1_crop_tensor = transform_image_to_crop_SE(bbox1_tensor, bbox1_tensor, h_f, w_f, crop_sz).cuda()
        '''Step3: forward prop (reference branch)'''
        with torch.no_grad():
            self.refine_network.forward_ref(patch1_tensor, bbox1_crop_tensor)


    def refine(self, Cframe, Cbbox, VOT=True, use_selector=True, mode=None):
        '''
        :param Cframe: Current frame(cv2 array) RGB
        :param Cbbox: Current bbox (ndarray) (x1,y1,w,h)
        :return: polygon (list) (8 elements)
        '''
        if (mode != None) and (mode not in ['bbox','corner','mask','average']):
            raise ValueError ("mode should belong to 'bbox','corner','mask','average'.")
        if use_selector and (mode != None):
            raise ValueError ("'use_selector' and 'mode' can not be simutaneously specified.")
        if (not use_selector)  and (mode==None):
            raise ValueError ("please specify use_selector or mode.")
        '''Step1: get cropped patch(tensor)'''
        Cpatch, h_f, w_f = sample_target_SE(Cframe, Cbbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        Cpatch_tensor = self.img_preprocess(Cpatch)

        '''Step2: forward prop (test branch)'''
        output_dict = {}
        with torch.no_grad():
            fusion_feat = self.refine_network.get_test_feat(Cpatch_tensor)
            if use_selector:
                '''predict which branch to use'''
                branch_scores = self.branch_selector(fusion_feat)
                _, max_idx = torch.max(branch_scores.squeeze(), dim=0) # tensor
                max_idx = max_idx.item() # int
                '''update mode'''
                mode = self.branch_list[max_idx]
                # print(mode)
            if mode in ['bbox','corner','mask']:
                pred = self.refine_network.get_output(mode)
                if mode=='bbox' or mode=='corner':
                    Pbbox_arr = self.pred2bbox(pred,input_type=mode)
                    output_dict['bbox_report'] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
                    output_dict['bbox_state'] = output_dict['bbox_report']
                elif mode == 'mask':
                    Pmask_arr = self.pred2bbox(pred,input_type=mode)
                    mask_arr = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
                                                        mode=cv2.BORDER_CONSTANT)
                    mask_bbox = mask2bbox_new(mask_arr,VOT=False)
                    if mask_bbox.size == 1: # empty mask
                        '''output corner instead'''
                        pred = self.refine_network.get_output('corner')
                        Pbbox_arr = self.pred2bbox(pred, input_type='corner')
                        output_dict['bbox_report'] = self.bbox_back(Pbbox_arr, Cbbox, h_f, w_f)
                        output_dict['bbox_state'] = output_dict['bbox_report']
                    else: # mask is large enough
                        output_dict['bbox_state'] = mask_bbox
                        if VOT:
                            output_dict['bbox_report'] = mask2bbox_new(mask_arr,VOT=False).flatten().tolist()
                        else:
                            output_dict['bbox_report'] = output_dict['bbox_state']

            return output_dict
    def get_mask(self, Cframe, Cbbox):
        '''
        :param Cframe: Current frame(cv2 array)
        :param Cbbox: Current bbox (ndarray) (x1,y1,w,h)
        :return: mask
        '''
        '''Step1: get cropped patch(tensor)'''
        Cpatch, h_f, w_f = sample_target_SE(Cframe, Cbbox, self.search_factor, self.input_sz, mode=cv2.BORDER_CONSTANT)
        Cpatch_tensor = self.img_preprocess(Cpatch)

        '''Step2: forward prop (test branch)'''
        output_dict = {}
        with torch.no_grad():
            _ = self.refine_network.get_test_feat(Cpatch_tensor)
            '''mask'''
            pred = self.refine_network.get_output('mask')
            Pmask_arr = self.pred2bbox(pred, input_type='mask')
            mask_arr = map_mask_back(Cframe, Cbbox, self.search_factor, Pmask_arr,
                                     mode=cv2.BORDER_CONSTANT)
            return mask_arr
    '''new'''
    def pred2bbox(self, prediction, input_type=None):
        if input_type == 'bbox':
            Pbbox = prediction
            Pbbox = delta2bbox(Pbbox)
            Pbbox_arr = np.array(Pbbox.squeeze().cpu())
            return Pbbox_arr
        elif input_type == 'corner':
            Pcorner = prediction  # (x1,y1,x2,y2)
            Pbbox_arr = np.array(Pcorner.squeeze().cpu())
            Pbbox_arr[2:] = Pbbox_arr[2:] - Pbbox_arr[:2]  # (x1,y1,w,h)
            return Pbbox_arr
        elif input_type == 'mask':
            Pmask = prediction
            Pmask_arr = np.array(Pmask.squeeze().cpu())  # (H,W) (0,1)
            return Pmask_arr
        else:
            raise ValueError("input_type should be 'bbox' or 'mask' or 'corner' ")
    def bbox_back(self,bbox_crop, bbox_ori, h_f, w_f):
        '''
        :param bbox_crop: 在裁剪出的图像块(256x256)上的坐标 (x1,y1,w,h) (4,)
        :param bbox_ori: 原始跟踪结果 (x1,y1,w,h) (4,)
        :param h_f: h缩放因子
        :param w_f: w缩放因子
        :return: 在原图上的坐标
        '''
        x1_c,y1_c,w_c,h_c = bbox_crop.tolist()
        x1_o,y1_o,w_o,h_o = bbox_ori.tolist()
        x1_oo = x1_o - 0.5 * w_o
        y1_oo = y1_o - 0.5 * h_o
        delta_x1 = x1_c / w_f
        delta_y1 = y1_c / h_f
        delta_w = w_c / w_f
        delta_h = h_c / h_f
        return np.array([x1_oo + delta_x1, y1_oo + delta_y1,
                         delta_w, delta_h])
    def get_network(self,checkpoint_dir):
        network = load_network(checkpoint_dir)
        network.cuda()
        network.eval()
        return network
    def img_preprocess(self,img_arr):
        '''转化成Pytorch tensor(RGB),归一化(转换到-1到1,减均值,除标准差)
        input img_arr (H,W,3)
        output (1,1,3,H,W)
        '''
        norm_img = ((img_arr/255.0) - self.mean)/(self.std)
        img_f32 = norm_img.astype(np.float32)
        img_tensor = torch.from_numpy(img_f32).cuda()
        img_tensor = img_tensor.permute((2,0,1))
        return img_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    def gt_preprocess(self,gt_arr):
        '''
        :param gt: ndarray (4,)
        :return: torch tensor (4,)
        '''
        return torch.from_numpy(gt_arr.astype(np.float32))

def add_frame_mask(frame, mask, threshold=0.5):
    mask_new = (mask>threshold)*255 #(H,W)
    frame_new = frame.copy().astype(np.float)
    frame_new[...,1] += 0.3*mask_new
    frame_new = frame_new.clip(0,255).astype(np.uint8)
    return frame_new
def add_frame_bbox(frame, refined_box, color):
    x1, y1, w, h = refined_box.tolist()
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    return frame

import time
def main():
    '''refinement module的测试代码'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    project_path = '/home/masterbin-iiau/Desktop/scale-estimator'
    refine_root = os.path.join(project_path,'ltr/checkpoints/ltr/SEbcm/')
    refine_model_name = 'SEbcm'
    refine_path = os.path.join(refine_root, refine_model_name)
    selector_root = os.path.join(project_path,'ltr/checkpoints/ltr/selector')
    selector_model_name = 'selector_bcm'
    selector_path = os.path.join(selector_root, selector_model_name)
    SE_module = Refine_module_bcm(refine_path, selector_path)

    video_dir = '/media/masterbin-iiau/WIN_SSD/GOT10K/train/GOT-10k_Train_000002'
    gt_file = os.path.join(video_dir,'groundtruth.txt')
    gt = np.loadtxt(gt_file,dtype=np.float32,delimiter=',')
    frame1_path = os.path.join(video_dir, '00000001.jpg')
    # frame1_path = os.path.join(video_dir, 'img','00000001.jpg')
    frame1 = cv2.cvtColor(cv2.imread(frame1_path),cv2.COLOR_BGR2RGB)
    SE_module.initialize(frame1, gt[0])
    # for i in range(1,gt.shape[0]):
    #     frame_test_path = os.path.join(video_dir, '%08d.jpg' % (i + 1))
    #     frame_test = cv2.imread(frame_test_path)
    #     frame_test_RGB = cv2.cvtColor(frame_test, cv2.COLOR_BGR2RGB)
    #     _ = SE_module.refine(frame_test_RGB, gt[i], VOT=False, use_selector=True)
    idx = 5
    frame_test_path = os.path.join(video_dir,'%08d.jpg' % (idx + 1))
    frame_test = cv2.imread(frame_test_path)
    frame_test_RGB = cv2.cvtColor(frame_test,cv2.COLOR_BGR2RGB)
    T = 500
    tic = time.time()
    for j in range(T):
        # _ = SE_module.refine(frame_test_RGB, gt[idx],VOT=False,use_selector=True)
        _ = SE_module.refine(frame_test_RGB, gt[idx],VOT=False,use_selector=False,mode='mask')
    toc = time.time()
    print('%f FPS'%(T/(toc-tic)))




if __name__ == '__main__':
    main()
