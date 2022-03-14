import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/longterm.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
def compute_LT_curves(trajectory: Tracking, sequence: Sequence_t, all_prre: PrRe):
    
    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.prebox_conf(sequence.name)
    # prebbox, confidence = trajectory.vot_prebox_conf(sequence.name)
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
    first_invisible = np.argwhere(visible == 0)
    all_prre.add_LT(first_invisible, len(overlaps))    
    all_prre.add_list_iou(overlaps)
    all_prre.add_visible(visible)
    all_prre.add_confidence(confidence) 


seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')
# all_trackers = [Tracking(tracker, path='/data1/gaoshang/workspace_votRGBD2019/results/') for tracker in ['prompt-cdtb']]
all_trackers = [Tracking(tracker) for tracker in ['DAL']]
# all_trackers = [Tracking(tracker) for tracker in ['CADMS']]
all_sequence = [Sequence_t(seq) for seq in seq_list]

for trackers in all_trackers:
    print(trackers.name)
    for sequence in all_sequence:
        
        if sequence.name in trackers._seqlist:
        # if sequence.name in trackers.vot_list():
            compute_LT_curves(trackers, sequence, trackers._prre)
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            trackers.lack(sequence.name)
            continue
    
    result = trackers._prre.fscore_LT
    before = result[0]
    after = result[1]
    print('Trackers: {}  {}  fscore: {}'.format(trackers.name, 'before invisible', before))
    print('Trackers: {}  {}  fscore: {}'.format(trackers.name, 'after invisible', after))
    logging.info('Trackers: {}  {}  fscore: {}'.format(trackers.name, 'before invisible', before))
    logging.info('Trackers: {}  {}  fscore: {}'.format(trackers.name, 'after invisible', after))