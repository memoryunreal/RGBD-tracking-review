import numpy as np
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def estimateIOU(box_a, box_b):
    if str(box_b[0]) == 'nan':
        return 0
    areaA = box_a[2] * box_a[3]
    areaB = box_b[2] * box_b[3]

    area_sum = areaA + areaB
    max_a_x = box_a[0] + box_a[2]
    max_a_y = box_a[1] + box_a[3]

    max_b_x = box_b[0] + box_b[2]
    max_b_y = box_b[1] + box_b[3]

    inter_x_max = min(max_a_x, max_b_x)
    inter_y_max = min(max_a_y, max_b_y)
    inter_x_min = max(box_a[0], box_b[0])
    inter_y_min = max(box_a[1], box_b[1])
    inter_w = inter_x_max - inter_x_min
    inter_h = inter_y_max - inter_y_min
    if inter_w<=0 or inter_h<=0:
        inter_area = 0 
    else:
        inter_area = inter_w * inter_h

    overlap = inter_area  / (area_sum - inter_area)
    return overlap





class Success(object):
    """Computes and stores the Success"""

    def __init__(self, n=21, max_overlap=1):
        self.max_overlap = max_overlap
        self.Xaxis = np.linspace(0, self.max_overlap, n)
        self.reset()

    def reset(self):
        self.overlaps = []

    def add_overlap(self, val):
        self.overlaps.append(val)

    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):
        # succ = [
        #     np.sum(i >= thres
        #            for i in self.overlaps).astype(float) / self.count
        #     for thres in self.Xaxis
        # ]
        succ = [
            np.sum(np.fromiter((i >= thres
                   for i in self.overlaps), dtype=float)) / self.count
            for thres in self.Xaxis
        ]
        
    
        return np.array(succ)

    @property
    def average(self):
        if len(self.overlaps) == 0:
            return 0
        return np.trapz(self.value, x=self.Xaxis) * 100 / self.max_overlap