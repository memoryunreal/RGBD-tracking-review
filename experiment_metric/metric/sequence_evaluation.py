import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging

log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/sequence_overall.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')

def compute_tpr_curves(trajectory: Tracking, sequence: Sequence_t, all_prre: PrRe):
    
    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.vot_prebox_conf(sequence.name)
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
        assert len(overlaps) == len(visible) == len(confidence)
    except:
        print("assert not equal")    
    all_prre.add_list_iou(overlaps)
    all_prre.add_visible(visible)
    all_prre.add_confidence(confidence) 


seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')
# all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
all_trackers = [Tracking(tracker,path='/data1/yjy/rgbd_benchmark/all_benchmark/normal/results') for tracker in ['iiau_rgbd']]
# all_trackers = [Tracking(tracker) for tracker in ['LTDSEd']]
all_sequence = [Sequence_t(seq) for seq in seq_list]
plotfile =  open('sequence_overall.txt', 'w')
for i, trackers in enumerate(all_trackers):
    print(trackers.name)
    for sequence in all_sequence:
        
        if sequence.name in trackers._votseqlist:
            compute_tpr_curves(trackers, sequence, trackers._prre)
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            trackers.lack(sequence.name)
            continue
        pr_list, re_list = trackers._prre.value
        pr,re,fscore = trackers._prre.fscore
        frame_num = trackers._prre.count
        trackers._prre.reset()
        print('Trackers: {}  Sequence: {} frame_num: {}  pr: {}  re: {}  fscore: {}'.format(trackers.name, sequence.name, frame_num, pr, re, fscore))
        logging.info('Trackers: {}  Seq_num: {} frame_num: {}  pr: {}  re: {}  fscore: {}'.format(trackers.name, sequence.name, frame_num, pr, re, fscore))
        # plotfile.writelines(trackers.name)
        # plotfile.writelines('\n')
    
        # plotfile.writelines('Pr_list: ')
        # plotfile.writelines(str(pr_list))
        # plotfile.writelines('\n')
    
        # plotfile.writelines('Re_list: ')
        # plotfile.writelines(str(re_list))
        # plotfile.writelines('\n')

plotfile.close()