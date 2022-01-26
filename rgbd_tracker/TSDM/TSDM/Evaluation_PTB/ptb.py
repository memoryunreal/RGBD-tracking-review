from PTB_track import ptb_track

with open('/media/guo/DataUbuntu/PTB_dataset/all_init_bbox.txt', 'r') as f1:
    lines = f1.readlines()

sequence_name = lines[0:190:2]
sequence_gt = lines[1:190:2]


for sequence in range(0,95):
    name = sequence_name[sequence].strip('\n')
    print('Now is the sequence of {}'.format(name))
    gt_bbox_ = [int(i) for i in sequence_gt[sequence].strip('\n').split(',')]
    mTrack = ptb_track()
    mTrack.track(name, gt_bbox_.copy())
    del mTrack
