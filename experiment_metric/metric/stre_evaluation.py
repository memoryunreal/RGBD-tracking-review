import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/stre.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')
def compute_SRE_curves(trajectory: Tracking, sequence: Sequence_t, all_prre: PrRe, tre=1):
    
    #overlaps = np.array(calculate_overlaps(trajectory, sequence.groundtruth(), (sequence.size) if bounded else None))
    prebbox, confidence = trajectory.vot_prebox_conf(sequence.name)
    gt = sequence.gt 

    # firstframe in each sequence
    try:
        overlaps = np.concatenate(([1], np.array([estimateIOU(prebbox[i], gt[i+tre] ) for i in range(len(prebbox))])))
        overlaps[np.isnan(overlaps)]=0
        confidence = np.concatenate(([1], np.array(confidence)))
    except:
        print("Error tre: {}  sequence {} sequence frame {} result frame {}".format(tre, sequence.name, len(sequence.gt), len(prebbox)))
        return 0

    #n_visible = len([region for region in sequence.groundtruth() if region.type is not RegionType.SPECIAL])
    # sequence.invisible (full-occlusion tag) if invisible= 1 full-occlusion invisible
    visible = np.array(sequence.invisible) < 1
    visible = visible + 0
    '''
        temporal evaluation
    '''

    visible = visible[tre-1:]
    '''
        temporal evaluation
    '''
    try:
        assert len(overlaps) == len(visible) == len(confidence)
    except:
        print("assert not equal")
        print("Error tre: {} Tracker: {} sequence {} sequence frame {} result frame {}".format(tre,trajectory.name, sequence.name, len(sequence.gt), len(prebbox)))
        return False    

    all_prre.add_list_iou(overlaps)
    all_prre.add_visible(visible)
    all_prre.add_confidence(confidence) 


seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')

SRE_workspace = '/data1/yjy/rgbd_benchmark/all_benchmark/SRE_workspace'
# sre_list = os.listdir(SRE_workspace)
sre_list = ['tre10']
# sre_tracker = ['DAL', 'CSRDCF', 'TSDM', 'DeT', 'iiau_rgbd', 'sttc_rgbd']
# sre_tracker = [ 'TSDM','CSRDCF','DAL', 'DeT', 'iiau_rgbd', 'sttc_rgbd']
sre_tracker = [ 'DSKCF_shape']
sre_average_fscore = [[] for i in range(len(sre_tracker))]
sre_average_pr = [[] for i in range(len(sre_tracker))]
sre_average_re = [[] for i in range(len(sre_tracker))]
tre_average_fscore = [[] for i in range(len(sre_tracker))]
tre_average_re = [[] for i in range(len(sre_tracker))]
tre_average_pr = [[] for i in range(len(sre_tracker))]
for sre in sre_list:
    if sre=='tre10' or sre =='tre20' or sre == 'tre30' or sre=='tre40' or sre=='tre50':
    

        # all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
        # all_trackers = [Tracking(tracker, path=os.path.join(SRE_workspace, sre, 'results')) for i,tracker in enumerate(['TSDM', 'CSRDCF'])]
        all_trackers = [Tracking(tracker, path=os.path.join(SRE_workspace, sre, 'results')) for i,tracker in enumerate(sre_tracker)]
        # all_trackers = [Tracking(tracker) for tracker in ['TSDM']]
        all_sequence = [Sequence_t(seq) for seq in seq_list]

        for id, trackers in enumerate(all_trackers):
            # if sre == 'tre20' or sre == 'tre30':
            #     if trackers.name == 'TSDM':
            #         continue
            print(trackers.name)
            
            for sequence in all_sequence:
          
                if not os.path.exists(os.path.join(trackers._votpath, sequence.name, '{}_001.txt'.format(sequence.name))):
                    continue
                if sequence.name in trackers._votseqlist:
                    compute_SRE_curves(trackers, sequence, trackers._prre, tre=int(sre.split('tre')[1]))
                    #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
                else:
                    trackers.lack(sequence.name)
                    continue
                
            pr,re,fscore = trackers._prre.fscore
            if trackers.name == sre_tracker[id]:
                tre_average_fscore[id].append(fscore)
                tre_average_pr[id].append(pr)
                tre_average_re[id].append(re)
            print('Trackers: {} sre: {}  pr: {} re: {} fscore: {}'.format(trackers.name, sre, pr, re, fscore))
    else:
    # all_trackers = [Tracking(tracker) for tracker in os.listdir('/data1/yjy/rgbd_benchmark/all_benchmark/results/')]
        all_trackers = [Tracking(tracker, path=os.path.join(SRE_workspace, sre, 'results')) for i,tracker in enumerate(sre_tracker)]
        # all_trackers = [Tracking(tracker) for tracker in ['CADMS']]
        all_sequence = [Sequence_t(seq) for seq in seq_list]

        for id, trackers in enumerate(all_trackers):
            print(trackers.name)
            for sequence in all_sequence:
                if not os.path.exists(os.path.join(trackers._votpath, sequence.name, '{}_001.txt'.format(sequence.name))):
                    continue
                if sequence.name in trackers._votseqlist:
                    compute_SRE_curves(trackers, sequence, trackers._prre)
                    #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
                else:
                    trackers.lack(sequence.name)
                    continue
                
            pr,re,fscore = trackers._prre.fscore
            if trackers.name == sre_tracker[id]:
                sre_average_fscore[id].append(fscore)
                sre_average_pr[id].append(pr)
                sre_average_re[id].append(re)
            print('Trackers: {} sre: {}  pr: {} re: {} fscore: {}'.format(trackers.name, sre, pr, re, fscore))
            logging.info('Trackers: {}  Seq_num: {} frame_num: {}  pr: {}  re: {}  fscore: {}'.format(trackers.name, trackers._numseq, trackers._prre.count, pr, re, fscore))

for i, sre in enumerate(sre_average_fscore):
    trackerID = sre_tracker[i]
    srefscore = np.mean(np.array(sre))
    srepr = np.mean(np.array(sre_average_pr[i]))
    srere = np.mean(np.array(sre_average_re[i]))
    print('Trackers: {} sre pr:{} re: {} fscore: {}'.format(trackerID, srepr, srere, srefscore))

    logging.info('Trackers: {} sre pr:{} re: {} fscore: {}'.format(trackerID, srepr, srere, srefscore))
for i, tre in enumerate(tre_average_fscore):
    trackerID = sre_tracker[i]
    trefscore = np.mean(np.array(tre))
    print('Trackers: {} tre fscore: {}'.format(trackerID, trefscore))
    logging.info('Trackers: {} tre fscore: {}'.format(trackerID, trefscore))

print('ok')