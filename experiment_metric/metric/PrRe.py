import numpy as np
class PrRe(object):
    """Computes and stores the Success"""

    def __init__(self):
        self.reset()
        self.thresholds = np.linspace(1, 0, 100)
    def reset(self):
        self.overlaps = []
        self.confidence = []
        self.visible = []
        self.depthQ = []
        self.attribute_name = []
        self.attribute_value = [[] for i in range(15)]
        self.lt = []
        self.start_frame = 0
        
    def add_overlap(self, val):
        self.overlaps.append(val)

    def add_list_iou(self, overlap:list):
        self.overlaps = np.concatenate((self.overlaps, overlap))
    
    def add_confidence(self, confidence:list):
        self.confidence = np.concatenate((self.confidence, confidence))
    def add_visible(self, visible:list):
        self.visible = np.concatenate((self.visible, visible))

    def add_depthquality(self, depthQ:list):
        self.depthQ = np.concatenate((self.depthQ, depthQ))

    def add_attribute(self, allattribute:dict):
        attributename = []
        for i, key in enumerate(allattribute):
            attributename.append(key) 
            self.attribute_value[i] = np.concatenate((self.attribute_value[i], allattribute.get(key)))
        self.attribute_name = attributename
        
    def add_LT(self, firstindex, length_frame):
        try:
            start_frame = self.start_frame
            first_invisible = start_frame + firstindex[0][0]
            end_frame = start_frame + length_frame - 1
            self.lt.append([start_frame,first_invisible, end_frame])
            self.start_frame += length_frame
        except:
            start_frame = self.start_frame
            end_frame = start_frame + length_frame - 1
            self.lt.append([start_frame,0, end_frame])
            self.start_frame += length_frame


    @property
    def count(self):
        return len(self.overlaps)

    @property
    def value(self):

        # succ = [
        #     np.sum(np.fromiter((i >= thres
        #            for i in self.overlaps), dtype=float)) / self.count
        #     for thres in self.Xaxis
        # ]
        # return np.array(succ)
        
        n_visible = len([vis for vis in self.visible if vis == True])
        precision = len(self.thresholds) * [float(0)]
        recall = len(self.thresholds) * [float(0)]

        for i, threshold in enumerate(self.thresholds):

            subset = self.confidence >= threshold
            
            if np.sum(subset) == 0:
                precision[i] = 1
                recall[i] = 0
            else:
                try:
                    # precision[i] = np.mean(self.overlaps[subset])
                    precision[i] = np.mean(self.overlaps[subset])
                    recall[i] = np.sum(self.overlaps[subset]) / n_visible
                except:
                    print('exception')
                if precision[i] == np.nan or recall[i] == np.nan:
                    print('nan')
        return precision, recall
    @property
    def value_DQ(self):

        '''
            quality level: low < 0.4   medium >= 0.4 <=0.8 high > 0.8
            all_precision: [high medium low]
            all_reacall: [high medium low]
        '''
        try:
            set1 = self.depthQ <=0.8
            set2 = self.depthQ >= 0.4
            lowQ_set = self.depthQ > 0.8
            mediumQ_set = set1 == set2
            highQ_set = self.depthQ < 0.4
        except:
            print('depth quality error') 
        all_set = [highQ_set,mediumQ_set, lowQ_set]
        all_precision =[]
        all_recall = []
        for qualityset in range(len(all_set)):
            confidence = self.confidence[all_set[qualityset]]
            overlaps = self.overlaps[all_set[qualityset]]
            visible = self.visible[all_set[qualityset]]
            n_visible = len([vis for vis in visible if vis == True])
            precision = len(self.thresholds) * [float(0)]
            recall = len(self.thresholds) * [float(0)]

            for i, threshold in enumerate(self.thresholds):

                subset = confidence >= threshold
                
                if np.sum(subset) == 0:
                    precision[i] = 1
                    recall[i] = 0
                else:
                    try:
                        # precision[i] = np.mean(self.overlaps[subset])
                        precision[i] = np.mean(overlaps[subset])
                        recall[i] = np.sum(overlaps[subset]) / n_visible
                    except:
                        print('exception')
                    if precision[i] == np.nan or recall[i] == np.nan:
                        print('nan')
            all_precision.append(precision)
            all_recall.append(recall)
        return all_precision, all_recall

    '''
        attribute 
    '''
    @property
    def value_AT(self):
        '''
            Attribute calculate
        '''
        all_precision = []
        all_recall = []
        all_tag = self.attribute_name
        all_value = self.attribute_value
        for id in range(len(all_tag)):
            index = all_value[id] ==1
            confidence = self.confidence[index]
            overlaps = self.overlaps[index]
            visible = self.visible[index]
            n_visible = len([vis for vis in visible if vis == True])
            precision = len(self.thresholds) * [float(0)]
            recall = len(self.thresholds) * [float(0)]

            for i, threshold in enumerate(self.thresholds):

                subset = confidence >= threshold
                
                if np.sum(subset) == 0:
                    precision[i] = 1
                    recall[i] = 0
                else:
                    try:
                        # precision[i] = np.mean(self.overlaps[subset])
                        precision[i] = np.mean(overlaps[subset])
                        recall[i] = np.sum(overlaps[subset]) / n_visible
                    except:
                        print('exception')
                    if precision[i] == np.nan or recall[i] == np.nan:
                        print('nan')
            #recall[np.isnan(recall)]=0
            all_precision.append(precision)
            all_recall.append(recall)
        
        return all_tag, all_precision, all_recall

    @property
    def value_LT(self):

        '''
            first invisible frame
            performanc: [0: 1st invisible] [1st invisible: end]
        '''
        n_visible = len([vis for vis in self.visible if vis == True])
        precision = len(self.thresholds) * [float(0)]
        recall = len(self.thresholds) * [float(0)]
        lt_Framelist = self.lt
        all_result = [[],[]]
        before_invisible = len(self.overlaps) * [float(0)]
        after_invisible = len(self.overlaps) * [float(0)]
        for index, framelist in enumerate(lt_Framelist):
            start = framelist[0]
            invisible = framelist[1]
            end = framelist[2]
            if invisible == 0:
                continue

            before_invisible[start:invisible] = (invisible-start) * [float(1)]
            after_invisible[invisible:end] = (end-invisible) * [float(1)]



        before_invisible = np.array(before_invisible)
        after_invisible = np.array(after_invisible)
        for id, frame in enumerate([before_invisible, after_invisible]):
            try:
                x_frame = frame == 1
                confidence = self.confidence[x_frame]
            except:
                print('ok')
            overlaps = self.overlaps[x_frame]
            visible = self.visible[x_frame]
            n_visible = len([vis for vis in visible if vis == True])

            precision = len(self.thresholds) * [float(0)]
            recall = len(self.thresholds) * [float(0)]

            for i, threshold in enumerate(self.thresholds):

                subset = confidence >= threshold
                
                if np.sum(subset) == 0:
                    precision[i] = 1
                    recall[i] = 0
                else:
                    try:
                        # precision[i] = np.mean(self.overlaps[subset])
                        precision[i] = np.mean(overlaps[subset])
                        recall[i] = np.sum(overlaps[subset]) / n_visible
                    except:
                        print('exception')
                    if precision[i] == np.nan or recall[i] == np.nan:
                        print('nan')
          
            all_result[id] = [precision, recall]
        return all_result




    @property
    def fscore(self):
        pr, re = self.value

        # pr_score = abs(np.trapz(pr, self.thresholds))
        pr_score = np.mean(pr)
        # re_score = abs(np.trapz(re, self.thresholds))
        re_score = np.max(re)
        # pr_score = np.sum(pr)/100
        # re_score = np.sum(re)/100
        fmeasure = [2*pr[i]*re[i]/(pr[i]+re[i]) for i in range(len(pr))]
        fscore = max(fmeasure)
        return pr_score, re_score, fscore

    '''
        depth quality evaluation
        all_result:[high medium low], all_result[high] = [pr,re,f]
    '''
    @property
    def fscore_DQ(self):
        all_pr, all_re = self.value_DQ
        all_result = []
        for index in range(len(all_pr)):
            pr = all_pr[index]
            re = all_re[index]
            pr_score = np.mean(pr)
            # re_score = abs(np.trapz(re, self.thresholds))
            re_score = np.max(re)
            # pr_score = np.sum(pr)/100
            # re_score = np.sum(re)/100
            fmeasure = [2*pr[i]*re[i]/(pr[i]+re[i]) for i in range(len(pr))]
            fscore = max(fmeasure)
            all_result.append(np.array([pr_score, re_score, fscore]))
        return all_result
    '''
        attribute
    '''
    @property
    def fscore_AT(self):
        all_tag, all_pr, all_re = self.value_AT
        all_result = dict()
        all_fscore = []
        tag_list = []
        for index in range(len(all_pr)):
            pr = all_pr[index]
            re = all_re[index]
            pr_score = np.max(pr)
            # re_score = abs(np.trapz(re, self.thresholds))
            re_score = np.max(re)
            # pr_score = np.sum(pr)/100
            # re_score = np.sum(re)/100
            fmeasure = [2*pr[i]*re[i]/(pr[i]+re[i]) for i in range(len(pr))]
            fscore = max(fmeasure)
            all_fscore.append(np.array([fscore]))
            tag_list.append(all_tag[index])
        all_result = dict(zip(tag_list,all_fscore))
        return all_result
    '''
        long term evaluation
        first invisible
    '''
    @property
    def fscore_LT(self):
        '''
            before result: result[0]
            after result: result[1]
        '''
        all_result = self.value_LT
        result = []
        for id in range(len(all_result)):
            pr = all_result[id][0]
            re = all_result[id][1]
            pr_score = np.max(pr)
            # re_score = abs(np.trapz(re, self.thresholds))
            re_score = np.max(re)
            # pr_score = np.sum(pr)/100
            # re_score = np.sum(re)/100
            fmeasure = [2*pr[i]*re[i]/(pr[i]+re[i]) for i in range(len(pr))]
            fmeasure = np.nan_to_num(np.array(fmeasure))
            fscore = np.max(fmeasure)
            result.append(fscore)
        return result
    '''
    long term evaluation
    first invisible
    '''

class Recall(object):
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
