# coding=utf-8
import os

#project_path = '/home/space/Documents/experiment/VOT20RGBD/src/'
project_path = '/home/yangjinyu/rgbd_tracker/CLGSD/'

siammask_cfg = os.path.join(project_path, 'models/siammask_r50_l3/config.yaml')
siammask_snap = os.path.join(project_path, 'models/siammask_r50_l3/model.pth')

rtmd_model = os.path.join(project_path, 'models/rt-mdnet.pth')

cdet_model = os.path.join(project_path, 'models/ctdet_coco_resdcn18.pth')
# cdet_res = os.path.join(project_path, 'models/ctdet_res')
cdet_res = os.path.join(project_path,'models/ctdet_res_all')

refineA = os.path.join(project_path, 'models/SEbcmnet_ep0040.pth.tar')
refineB = os.path.join(project_path, 'models/Branch_Selector_ep0030.pth.tar')

flow_model = os.path.join(project_path, 'models/FlowNet2-CS_checkpoint.pth.tar')
