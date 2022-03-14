import os
import numpy as np
from Tracker import Tracking
from Sequence import Sequence_t
from PrRe import PrRe
from Iou import estimateIOU
import logging

log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../log/speed.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')

def compute_speed(trajectory: Tracking, sequence: Sequence_t):
    
    speed_value, flag = trajectory.prebox_speed(sequence.name)
    if flag:
        return 0, 0
    else:
        time_consume = np.sum(speed_value)
        lengthframe= sequence.num_frame - 1
    return time_consume, lengthframe


seq_list = os.listdir('/data1/yjy/rgbd_benchmark/alldata')
seq_list.remove('list.txt')

all_trackers = [Tracking(tracker) for tracker in ['DeT', 'DAL', 'TSDM']]
all_sequence = [Sequence_t(seq) for seq in seq_list]

for i, trackers in enumerate(all_trackers):
    print(trackers.name)
    tracker_speed = 0
    tracker_frame = 0
    for sequence in all_sequence:
        
        if sequence.name in trackers._seqlist:
            seqtime, seqframe = compute_speed(trackers, sequence)
            tracker_frame += seqframe
            tracker_speed += seqtime
            #print('{}: length of iou {} '.format(trackers.name, trackers._prre.count))
        else:
            trackers.lack(sequence.name)
            continue
    trackerfps = tracker_frame / tracker_speed
    print('Trackers: {}  Seq_num: {} frame_num: {}  fps: {}'.format(trackers.name, trackers._numseq, tracker_frame, trackerfps))
    logging.info('Trackers: {}  Seq_num: {} frame_num: {}  fps: {}'.format(trackers.name, trackers._numseq, tracker_frame, trackerfps))
