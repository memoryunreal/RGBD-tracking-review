import numpy as np
import torch
import torch.nn.functional as F
import cv2

from TSDM.models.SiamRPN.SiamNetMob import MySiamRPNMob
from TSDM.tools.anchor import Anchors
from TSDM.tools.bbox import get_axis_aligned_bbox
from TSDM.tools.model_load import load_pretrain

PENALTY_K = 0.05
WINDOW_INFLUENCE = 0.28
LR = 0.22
EXEMPLAR_SIZE = 127
INSTANCE_SIZE = 255
LOST_INSTANCE_SIZE = 831
BASE_SIZE = 8
CONTEXT_AMOUNT = 0.5
CONFIDENCE_LOW = 0.8
CONFIDENCE_HIGH = 0.98

ANCHOR_STRIDE = 8
ANCHOR_RATIOS = [0.33, 0.5, 1, 2, 3]
ANCHOR_SCALES = [8]
ANCHOR_NUM = 5

class MySiamRPNMobTracker():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.longterm_state = False
        self.anchor_num = len(ANCHOR_RATIOS) * len(ANCHOR_SCALES)
        self.interval = 0

    def generate_anchor(self, score_size):
        anchors = Anchors(ANCHOR_STRIDE,
                          ANCHOR_RATIOS,
                          ANCHOR_SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def update_state(self, region):
        self.center_pos = np.array([region[0]+region[2]/2, region[1]+region[3]/2])
        self.size = np.array([region[2], region[3]])

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]
 
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def init(self, img, bbox):
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        self.cx16 = np.array([self.center_pos[0]])
        self.cy16 = np.array([self.center_pos[1]])
        self.width16 = np.array([self.size[0]])
        self.height16 = np.array([self.size[1]])

        # calculate z crop size
        w_z = self.size[0] + CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    EXEMPLAR_SIZE, s_z, 
                                    self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        w_z = self.size[0] + CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = LOST_INSTANCE_SIZE
        else:
            instance_size = INSTANCE_SIZE

        score_size = (instance_size - EXEMPLAR_SIZE) // ANCHOR_STRIDE + 1 + BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)

        s_x = s_z * (instance_size / EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    instance_size,
                                    round(s_x), self.channel_average)
        
        outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * PENALTY_K)
        pscore = penalty * score

        # window penalty
        if not self.longterm_state:
            pscore = pscore * (1 - WINDOW_INFLUENCE) + \
                    window * WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001

        # get 'best_score' and the most important 'scores' and 'boxes' and 'lr'
        best_idx = np.argmax(pscore)
        best_score = pscore[best_idx]

        best_idx16 = np.argsort(pscore)[::-1][:16] 
        best_idx16 = best_idx16[pscore[best_idx16] > pscore[best_idx]*0.95].tolist()
  
        bbox = pred_bbox[:, best_idx16] / scale_z
        lr = penalty[best_idx16] * score[best_idx16] * LR
        
        # get position and size
        if best_score >= CONFIDENCE_LOW:
            cx = bbox[0,0] + self.center_pos[0]
            cy = bbox[1,0] + self.center_pos[1]
            width = self.size[0] * (1 - lr[0]) + bbox[2,0] * lr[0]
            height = self.size[1] * (1 - lr[0]) + bbox[3,0] * lr[0]

            self.cx16 = bbox[0,:] + self.center_pos[0]
            self.cy16 = bbox[1,:] + self.center_pos[1]
            self.width16 = self.size[0] * (1 - lr) + bbox[2,:] * lr
            self.height16 = self.size[1] * (1 - lr) + bbox[3,:] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]
            width = self.size[0]
            height = self.size[1]

            self.cx16 = np.array([cx])
            self.cy16 = np.array([cy])
            self.width16 = np.array([width])
            self.height16 = np.array([height])

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        
        bbox16 = [self.cx16 - self.width16 / 2,
                  self.cy16 - self.height16 / 2,
                  self.width16,
                  self.height16]
        
        if best_score < CONFIDENCE_LOW:
            self.interval += 1
            self.longterm_state = False
            if self.interval >= 10:
                self.longterm_state = True
                self.interval = 0
        elif best_score > CONFIDENCE_HIGH:
            self.longterm_state = False
            self.interval = 0

        return {
                'bbox': bbox,
                'bbox16': bbox16,
                'best_score': best_score,
               }


if __name__ == '__main__':

    model = MySiamRPN()
    model_load_path = '/home/guo/zpy/Mypysot/mypysot/dataset/weight/model.pth'
    model = load_pretrain(model, model_load_path).cuda().eval()
    track = MySiamRPNTracker(model)

    img1 = cv2.imread('/home/guo/zpy/vot-toolkit-master/sequences/backpack_blue/color/00000001.jpg')
    img2 = cv2.imread('/home/guo/zpy/vot-toolkit-master/sequences/backpack_blue/color/00000002.jpg')
    imgd1 = Image.open('/home/guo/zpy/vot-toolkit-master/sequences/backpack_blue/depth/00000002.png')
    imgd2 = Image.open('/home/guo/zpy/vot-toolkit-master/sequences/backpack_blue/depth/00000002.png')

    imgd1 = np.array(imgd1)
    imgd1 = imgd1[98:98+216, 278:278+167]
    depth = np.mean(imgd1)

    gt_bbox = [278.33,98.0,167.66,216.67]
    gt_bbox = [gt_bbox[0], gt_bbox[1],
    gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
    gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
    

    track.init(img1, gt_bbox_)
    state = track.track(img2,imgd2,1,depth)

    boxes = np.array(state['bbox16'].copy())
    for i in range(boxes.shape[1]):
        box = [int(i) for i in boxes[:,i]]
        x1, y1, x2, y2 = box[0], box[1], box[0]+box[2], box[1]+box[3]
        cv2.line(img2,(x1,y1),(x1,y2),(255,0,0),1)
        cv2.line(img2,(x2,y1),(x2,y2),(255,0,0),1)
        cv2.line(img2,(x1,y1),(x2,y1),(255,0,0),1)
        cv2.line(img2,(x1,y2),(x2,y2),(255,0,0),1)
    cv2.imwrite('result.jpg', img2)
    print(state['best_score'])






