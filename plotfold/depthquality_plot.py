import os
import numpy as np
import sys
import cv2
sys.path.append('/home/lz/TMM2022/experiment_metric/metric')
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
print('ok')

'''
    depthQ: depth quality value for sequence np.array()
    return: quality index [True, False,True...]
'''
def divide_quality(sequence:Sequence_t, depthQ):
    gtvalue = sequence.gt
    depthvalue = depthQ    
    highquality = depthvalue < 0.4
    lowquality = depthvalue > 0.8
    set1 = depthvalue < 0.8
    set2 = depthvalue > 0.4
    mediumquality = set1 == set2
    return gtvalue, highquality, mediumquality, lowquality

def trackingresult(trajectory: Tracking, sequence: Sequence_t):
    
    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.prebox_conf(sequence.name)
    state = True
    if not len(prebbox) == len(sequence.gt) - 1:
        state =  False


    return prebbox, state
def plot_quality(sequence:Sequence_t, trackers_result,trackerlist, gtbox, depthQ):
    trackerslist = trackerlist
    gt_bbox = gtbox
    trackers_bbox = trackers_result
    #highquality = high_list
    #mediumquality = medium_list
    #lowquality = low_list
    #qualityvalue = depthQ 

    colorpath = os.path.join(sequence._path, 'color')
    depthpath = os.path.join(sequence._path, 'depth')
    colorimageindex = os.listdir(colorpath)
    colorimageindex.sort()
    colorimagepath = [os.path.join(colorpath, filename) for i, filename in enumerate(colorimageindex)]

    # line_color
    red = (0, 0, 255)
    green = (0, 255, 0)
    grey = (128, 128, 128)
    yellow = (10, 215, 255)
    blue = (255, 0, 0)
    puple = (128, 0, 128)
    orange = (0, 69, 255)
    line_color = [red, grey, yellow, blue, puple, orange, green]

    save_path = os.path.join(savepath, sequence.name, 'color')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(1,len(gt_bbox)):
        if depthQ[i] < 0.4:
            savename = os.path.join(save_path, '{}_high_{}'.format(depthQ[i], os.path.basename(colorimageindex[i])))
        elif depthQ[i] > 0.8:
            savename = os.path.join(save_path, '{}_low_{}'.format(depthQ[i], os.path.basename(colorimageindex[i])))
        else:
            savename = os.path.join(save_path, '{}_medium_{}'.format(depthQ[i], os.path.basename(colorimageindex[i])))

        originimage = cv2.imread(colorimagepath[i])
        # gt bbox plot
        try:
            cv2.line(originimage, (int(gt_bbox[i][0]),int(gt_bbox[i][1])), (int(gt_bbox[i][0]+gt_bbox[i][2]), int(gt_bbox[i][1])), line_color[-1])
            cv2.line(originimage, (int(gt_bbox[i][0]),int(gt_bbox[i][1])), (int(gt_bbox[i][0]), int(gt_bbox[i][1]+gt_bbox[i][3])), line_color[-1])
            cv2.line(originimage, (int(gt_bbox[i][0] + gt_bbox[i][2]),int(gt_bbox[i][1]+gt_bbox[i][3])), (int(gt_bbox[i][0]), int(gt_bbox[i][1]+gt_bbox[i][3])), line_color[-1])
            cv2.line(originimage, (int(gt_bbox[i][0] + gt_bbox[i][2]),int(gt_bbox[i][1]+gt_bbox[i][3])), (int(gt_bbox[i][0]+gt_bbox[i][2]), int(gt_bbox[i][1])), line_color[-1])
        except:
            savename = savename.split(".")[0]+ '.' + savename.split(".")[1] + '_gtloss' + '.' + savename.split(".")[2]
            

        for j in range(len(trackerslist)):
            left_x = trackers_bbox[j][i-1][0]
            left_y = trackers_bbox[j][i-1][1]
            width =  trackers_bbox[j][i-1][2]
            height = trackers_bbox[j][i-1][3]
            try:
                cv2.line(originimage, (int(left_x),int(left_y)), (int(left_x+width), int(left_y)), line_color[j])
                cv2.line(originimage, (int(left_x),int(left_y)), (int(left_x), int(left_y+height)), line_color[j])
                cv2.line(originimage, (int(left_x+width),int(left_y+height)), (int(left_x+width), int(left_y)), line_color[j])
                cv2.line(originimage, (int(left_x+width),int(left_y+height)), (int(left_x), int(left_y+height)), line_color[j])
            except:
                savename = savename.split(".")[0]+ '.' + savename.split('.')[1] + '_loss_{}'.format(trackerslist[j]) + '.' +savename.split(".")[2]
                continue    
        try:
            cv2.imwrite(savename, originimage)
        except Exception as e:
            print(e)

'''
    depth quality 
'''
import scipy.io
mat_det = scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/det.mat')
mat_cdtb = scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/cdtb_25.mat')
mat_stc = scipy.io.loadmat('/data1/yjy/rgbd_benchmark/all_benchmark/depthquality/stc_all.mat') 
det = mat_det['S'][0]
cdtb = mat_cdtb['A'][0]
stc = mat_stc['S'][0]
sequence_listQ = []
depthQ = []
for matfile in [det,stc]:
    for result in matfile:
        sequence_name = str(result[0]).split("'")[1]
        sequence_depthQ =  result[1][0]
        sequence_listQ.append(sequence_name)
        depthQ.append(sequence_depthQ)

for i in range(len(cdtb)-1):
    sequence_name = str(cdtb[i][0]).split("'")[1]
    sequence_depthQ = cdtb[i][1][0]
    sequence_listQ.append(sequence_name)
    depthQ.append(sequence_depthQ)
'''
    depth quality
'''
savepath= '/ssd3/lz/TMM2022/visualization/depthquality/'
seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')
# all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
trackers_list = ['iiau_rgbd','TALGD','DAL','DeT','CA3DMS','CSRKCF']
# trackers_list = ['CSRKCF']
all_trackers = [Tracking(tracker) for tracker in trackers_list]
all_sequence = [Sequence_t(sequence_listQ[i]) for i in range(len(sequence_listQ))]
seq_ok = [
'adapter01_indoor',
'backpack_indoor',
'bag01_indoor',
'bag02_indoor',
'ball01_wild',
'ball06_indoor',
'ball10_wild',
'ball11_wild',
'ball15_wild',
'ball18_indoor',
'ball20_indoor',
'bandlight_indoor'
]
count_seq = 0
for index, sequence in enumerate(all_sequence):
    skip_flag = 0
    print('{} {} start'.format(count_seq,sequence.name))
    if sequence.name in seq_ok:
        continue
    gt_box, high_list, medium_list, low_list = divide_quality(sequence, np.array(depthQ[index]))
    trackers_box = []
    for i, trackers in enumerate(all_trackers):
        
        if sequence.name in trackers._seqlist:
            prebox, state = trackingresult(trackers, sequence)
            if not state:
                skip_flag = 1
                break
            trackers_box.append(prebox)
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            skip_flag = 1
            print('{} lack {}'.format(trackers.name, sequence.name))
            continue
    if skip_flag == 1:
        continue
    plot_quality(sequence,trackers_box,trackers_list,gt_box, depthQ[index])
    count_seq += 1
    print('{} {} finished'.format(count_seq,sequence.name))