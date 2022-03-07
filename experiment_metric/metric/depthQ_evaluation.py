import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging
import scipy.io
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/depthQ.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
def compute_DQ(trajectory: Tracking, sequence: Sequence_t, all_prre: PrRe, depthQ):

    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.prebox_conf(sequence.name)
    gt = sequence.gt 

    # firstframe in each sequence
    overlaps = np.concatenate(([1], np.array([estimateIOU(prebbox[i], gt[i+1] ) for i in range(len(prebbox))])))
    overlaps[np.isnan(overlaps)]=0
    confidence = np.concatenate(([1], np.array(confidence)))


    #n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])
    # sequence.invisible (full-occlusion tag) if invisible= 1 full-occlusion invisible
    visible = np.array(sequence.invisible) < 1
    visible = visible + 0
    try:
        assert len(overlaps) == len(visible) == len(confidence) == len(depthQ)
    except:
        print("assert not equal")    
    all_prre.add_list_iou(overlaps)
    all_prre.add_visible(visible)
    all_prre.add_confidence(confidence)
    all_prre.add_depthquality(depthQ) 



'''
    depth quality 
'''
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
seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')
# all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
all_trackers = [Tracking(tracker) for tracker in ['DeT']]
all_sequence = [Sequence_t(sequence_listQ[i]) for i in range(len(sequence_listQ))]

for trackers in all_trackers:
    print(trackers.name)
    for i in range(len(all_sequence)):
        
        if all_sequence[i].name in trackers._seqlist:
            compute_DQ(trackers, all_sequence[i], trackers._prre, depthQ[i])
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            trackers.lack(all_sequence[i].name)
            continue
    
    result = trackers.prre.fscore_DQ
    list_quality = ['high quality', 'medium quality', 'low quality']
    for qualityID in range(len(result)):

        pr,re,fscore = result[qualityID]
        print('Trackers: {} quality level: {}   pr: {}  re: {}  fscore: {}'.format(trackers.name,list_quality[qualityID],pr, re, fscore))
        logging.info('Trackers: {}  quality level: {}  pr: {}  re: {}  fscore: {}'.format(trackers.name, list_quality[qualityID], pr, re, fscore))

