import os
from re import M
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging
import scipy.io
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/depthQ_dataset.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')


'''
    depth quality 
'''
mat_det = scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/det.mat')
mat_cdtb = scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/cdtb_25.mat')
mat_cdtblist = [scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/cdtb_resize{}.mat'.format(i)) for i in [10,20,30,40,50,60,70,80]]
# mat_cdtb = [scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/cdtb_{}.mat'.format(i) for i in [10,20,30,40,50,60,70,80])]
mat_stc = scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/stc_all.mat') 
det = mat_det['S'][0]
cdtb = mat_cdtb['A'][0]
stc = mat_stc['S'][0]
sequence_listQ = []
depthQlist = []
seqQlist = []

# det and stc
# for matfile in [det,stc]:
'''
    three dataset
    return depthQlist[det,stc,cdtb] 
'''
for i, matfile in enumerate([det, stc]):
    depthQ = [] 
    sequencelistQ = [] 
    for result in matfile:
        sequence_name = str(result[0]).split("'")[1]
        sequence_depthQ =  result[1][0]
        sequencelistQ.append(sequence_name)
        depthQ.append(sequence_depthQ)
    depthQlist.append(depthQ)
    seqQlist.append(sequencelistQ)

#cdtb
# depthQ = []
# sequencelistQ = []
# for i, matcdtb in enumerate(mat_cdtblist):
#     matfile = matcdtb['S'][0]
#     for result in matfile:
#         sequence_name = str(result[0]).split("'")[1]
#         sequence_depthQ =  result[1][0]
#         sequencelistQ.append(sequence_name)
#         depthQ.append(sequence_depthQ)
# depthQlist.append(depthQ)
# seqQlist.append(sequencelistQ)
'''
    depth quality
'''
seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')

all_sequence = [Sequence_t(sequence_listQ[i]) for i in range(len(sequence_listQ))]

# dataset select
# dataset = ['Depthtrack', 'STC', 'CDTB']
dataset = ['Depthtrack', 'STC' ]
for j in range(len(seqQlist)):
    seqlist = seqQlist[j]
    depthvalue = depthQlist[j]
    sum_depthQ = 0
    frame_all = 0
    for i, value in enumerate(depthvalue):
        frame_all+=len(value)
        sumvalue = np.sum(value)
        sum_depthQ += sumvalue
    # sum_depthQ: each dataset depth quality (bpr) sum
    average_score = sum_depthQ / frame_all
    
    print('Dataset: {} quality: {}  frame: {} average quality: {}' .format(dataset[j], sum_depthQ, frame_all, average_score))
    logging.info('Dataset: {} quality: {}  frame: {} average quality: {}' .format(dataset[j], sum_depthQ, frame_all, average_score))


# %% drop rate
import logging
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/depthQ_drop.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
depthQ_result = [
'DAL        &0.351& 0.419 & 0.444',
'CA3DMS     &0.166 & 0.205 & 0.208 ',
'CSR_RGBD++     &0.091 & 0.099 & 0.126 ',
'DSKCF     &0.0277 &0.027 & 0.0291 ',
'DSKCF_shape &0.0216 &0.0245 &0.0247',
'DRefine     & 0.394 & 0.452 & 0.465 ',
'SLMD      &0.373 & 0.459 & 0.479 ',
'TALGD  & 0.516 & 0.510 & 0.517',
'CLGSD & 0.309 & 0.398 & 0.428  ',
'DDiMP & 0.416 & 0.474 & 0.488  ',
'Siam_LTD & 0.324 & 0.418 & 0.420  ',
'LTDSEd & 0.370 & 0.434 &  0.451',
'Siam_LTD &0.276  & 0.353 &0.370 ',
'SiamDW\_D &  0.377 & 0.427 & 0.450 ',
'DeT   &  0.490  &  0.506 & 0.502 ',
'TSDM   & 0.296 & 0.346 & 0.370 ',
'ATCAIS & 0.473 & 0.485 & 0.494 ',
'STARK_RGBD & 0.534 & 0.552 & 0.557 ',
'sttc_rgbd & 0.427 & 0.455 & 0.477']
trackername = []
highQ = []
mediumQ = []
lowQ = []
for i, result in enumerate(depthQ_result):
    line = result.split("&")
    trackername.append(line[0])
    lowQ.append(float(line[1]))
    mediumQ.append(float(line[2]))
    highQ.append(float(line[3]))

def droprate(trackname, low,medium,high):
    average_low = 0
    average_medium = 0
    average_high = 0
    average_lowdrop = 0
    average_mediumdrop =0
    for i in range(len(low)):
        average_low += low[i]
        average_medium += medium[i]
        average_high += high[i]
        low_drop = (high[i] - low[i]) / high[i]
        medium_drop = (high[i] - medium[i]) / high[i]
        average_lowdrop += low_drop
        average_mediumdrop += medium_drop
        print('{} droprate low: {:.2f}%, medium: {:.2f}%'.format(trackname[i], low_drop*100, medium_drop*100 ))
        # logging.info('{} droprate low: {}, medium: {}'.format(trackname[i], low_drop, medium_drop ))
    print('average low:{} lowdrop:{}, average medium:{} medium drop {}, average high: {}'.format(average_low/len(low), average_lowdrop/len(low),average_medium/len(low),average_mediumdrop/len(low),average_high/len(low)))
droprate(trackername, lowQ, mediumQ, highQ)
# %%
