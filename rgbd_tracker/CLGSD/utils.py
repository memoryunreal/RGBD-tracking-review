# coding=utf-8

import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import torch


def overlap_ratio2(r1, r2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,x,y] or
            2d array of N x [x,y,x,y]
    '''
    rect1 = r1.copy()
    rect2 = r2.copy()
    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    rect1[:, 2:] = rect1[:, 2:] - rect1[:, :2] + 1
    rect2[:, 2:] = rect2[:, 2:] - rect2[:, :2] + 1

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


# siamrpn_tracker - clip boundary
def UOF_bbox_clip(cx, cy, width, height, boundary):
    cx = np.clip(cx, 0, boundary[1])
    cy = np.clip(cy, 0, boundary[0])
    width = np.clip(width, 10, boundary[1])
    height = np.clip(height, 10, boundary[0])
    return cx, cy, width, height

# ==================================================================================================================== #


class get_tracker():
    def __init__(self, tracker_name=None, select_vid='ballet', vot_res_path=None, vot_seq_path=None):

        self.name = tracker_name

        select_gt_path = os.path.join(vot_seq_path, select_vid, 'groundtruth.txt')
        select_box_path = os.path.join(vot_res_path, tracker_name, 'longterm', select_vid, '%s_001.txt' % select_vid)
        select_score_path = os.path.join(vot_res_path, tracker_name, 'longterm', select_vid,
                                         '%s_001_confidence.value' % select_vid)
        self.select_frame_path = os.path.join(vot_seq_path, select_vid, 'color')

        self.gt = self.get_gt(select_gt_path)
        self.boxes = self.get_box(select_box_path)
        self.scores = self.get_scores(select_score_path)

    def get_box(self, path):
        with open(path, 'r') as f:
            boxes = f.readlines()[1:]

        tmp_boxes = []
        for i in range(boxes.__len__()):
            tmp_boxes.append(np.array([float(item) for item in boxes[i].split(',')[:4]]).reshape(-1, 4))
        boxes = np.concatenate(tmp_boxes, 0)

        return boxes

    def get_scores(self, path):
        with open(path, 'r') as f:
            scores = f.readlines()[1:]

        tmp_scores = []
        for i in range(scores.__len__()):
            tmp_scores.append(float(scores[i]))
        scores = np.array(tmp_scores)

        return scores

    def get_frame(self, idx):
        im_path = os.path.join(self.select_frame_path, '%08d.jpg' % idx)
        im = cv2.imread(im_path)[:, :, ::-1]  # RGB
        # H, W, _ = im.shape
        #
        # S = 480.0 / H
        #
        # im = cv2.resize(im, (int(W * S), int(H * S)))
        # im[:35, :160, 2] = 220
        # im[:35, :160, 1] = im[:35, :160, 1] // 2
        # im[:35, :160, 0] = im[:35, :160, 1] // 2

        return im

    def get_gt(self, path):
        gt = np.loadtxt(path, delimiter=',')[1:, :]

        return gt

    def __len__(self):
        return self.gt.shape[0]+1


def box_to_xyxy(box):
    tmp_box = box.copy()
    tmp_box = tmp_box.reshape(-1,4)
    tmp_box[:,2:] = tmp_box[:,2:] + tmp_box[:,:2]
    return tmp_box


def box_to_xywh(box):
    tmp_box = box.copy()
    tmp_box = tmp_box.reshape(-1,4)
    tmp_box[:,2:] = tmp_box[:,2:] - tmp_box[:,:2]
    return tmp_box


def overlap_ratio(r1, r2, mode='xyxy'):
    """
    computing IoU
    :param r1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param r2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    rec1 = r1.copy().reshape(-1, 4)
    rec2 = r2.copy().reshape(-1, 4)

    if mode == 'xywh':
        # to [x1 y1 x2 y2]
        rec1[:, 2:] = rec1[:, 2:] + rec1[:, :2] - 1
        rec2[:, 2:] = rec2[:, 2:] + rec2[:, :2] - 1

    # computing area of each rectangles
    s_rec1 = (rec1[:, 2] - rec1[:, 0]) * (rec1[:, 3] - rec1[:, 1])
    s_rec2 = (rec2[:, 2] - rec2[:, 0]) * (rec2[:, 3] - rec2[:, 1])

    # find the each edge of intersect rectangle
    left_line = np.maximum(rec1[:, 0], rec2[:, 0])
    right_line = np.minimum(rec1[:, 2], rec2[:, 2])
    top_line = np.maximum(rec1[:, 1], rec2[:, 1])
    bottom_line = np.minimum(rec1[:, 3], rec2[:, 3])

    intersect = np.maximum(0, right_line - left_line) * np.maximum(0, bottom_line - top_line)
    union = s_rec1 + s_rec2 - intersect
    iou = np.clip(intersect / union, 0, 1)

    return iou


def save_vot(
        vid_name, tracker_name='0', save_path='./',
        box_list=[], confidence_list=[], time_list=[], tag='unsupervised'   # longterm and unsupervised
):
    vid_path = os.path.join(save_path, tracker_name, tag, '{:s}'.format(vid_name))
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    # save box
    if len(box_list) > 0:
        with open(os.path.join(vid_path, '{:s}_001.txt'.format(vid_name)), 'w') as f:
            # f.write('1\n')
            f.writelines(box_list)
    # save confidence
    if len(confidence_list) > 0:
        with open(os.path.join(vid_path, '{:s}_001_confidence.value'.format(vid_name)), 'w') as f:
            f.write('\n')
            f.writelines(confidence_list)
    # save time
    if len(time_list) > 0:
        with open(os.path.join(vid_path, '{:s}_time.txt'.format(vid_name)), 'w') as f:
            # f.write(time_list[0])
            f.writelines(time_list)


def save_lasot(
        vid_name, tracker_name='0', save_path='./', box_list=[]  # longterm and unsupervised
):
    vid_path = os.path.join(save_path, tracker_name)
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    # save box
    if len(box_list) > 0:
        with open(os.path.join(vid_path, '{:s}.txt'.format(vid_name)), 'w') as f:
            f.writelines(box_list)


def save_tlp(
        vid_name, tracker_name='0', save_path='./', box_list=[]  # longterm and unsupervised
):
    vid_path = os.path.join(save_path, tracker_name)
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    # save box
    if len(box_list) > 0:
        with open(os.path.join(vid_path, '{:s}.txt'.format(vid_name)), 'w') as f:
            f.writelines(box_list)


def save_got10k(
        vid_name, tracker_name='0', save_path='./', box_list=[], time_list=[]  # longterm and unsupervised
):
    vid_path = os.path.join(save_path, tracker_name, '{:s}'.format(vid_name))
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)

    # save box
    if len(box_list) > 0:
        with open(os.path.join(vid_path, '{:s}_001.txt'.format(vid_name)), 'w') as f:
            f.writelines(box_list)

    if len(time_list) > 0:
        with open(os.path.join(vid_path, '{:s}_time.txt'.format(vid_name)), 'w') as f:
            f.write(time_list[0])
            f.writelines(time_list)


def show(
        img, frame_idx,
        predict_box=None,  # [x1 y1 x2 y2]
        predict_score=None,
        siam_score=None,
        verifier_box=None,
        verifier_score=None,
        rpn_score=None,
        mask=None,
        gt_box=None,  # [x y w h]
        boxes1=None,  # [x1 y1 x2 y2]
        boxes2=None,  # [x1 y1 x2 y2]
        boxes3=None,  # [x1 y1 x2 y2]
        tracker_name=''
):

    if mask is not None:
        img[:, :, 1] = (mask > 0) * 255 + (mask == 0) * img[:, :, 1] # BGR

    im = Image.fromarray(img[:, :, ::-1])  # RGB
    draw = ImageDraw.Draw(im)

    if gt_box is not None:  # [x y w h]
        draw.rectangle((gt_box[0],
                        gt_box[1],
                        gt_box[0] + gt_box[2],
                        gt_box[1] + gt_box[3]), outline=(255, 0, 0), width=3)

    if predict_box is not None:  # [x y w h]
        draw.rectangle((predict_box[0],
                        predict_box[1],
                        predict_box[2],
                        predict_box[3]), outline=(0, 255, 0), width=3)

    if verifier_box is not None:  # [x y w h]
        draw.rectangle((verifier_box[0],
                        verifier_box[1],
                        verifier_box[0] + verifier_box[2],
                        verifier_box[1] + verifier_box[3]), outline=(0, 0, 255), width=3)

    if boxes1 is not None:  # [x1 y1 x2 y2]
        for i in range(boxes1.shape[0]):
            draw.rectangle((boxes1[i, 0], boxes1[i, 1],
                            boxes1[i, 2], boxes1[i, 3]), outline=(255, 0, 0), width=1)
    if boxes2 is not None:  # [x1 y1 x2 y2]
        for i in range(boxes2.shape[0]):
            draw.rectangle((boxes2[i, 0], boxes2[i, 1],
                            boxes2[i, 2], boxes2[i, 3]), outline=(0, 0, 255), width=1)
    if boxes3 is not None:  # [x1 y1 x2 y2]
        for i in range(boxes3.shape[0]):
            draw.rectangle((boxes3[i, 0], boxes3[i, 1],
                            boxes3[i, 2], boxes3[i, 3]), outline=(255, 255, 0), width=2)

    img = np.array(im)[:, :, ::-1]
    img = cv2.resize(img, (640 * 2, 360 * 2))
    img[:155, :150, 0] = 220
    img[:155, :150, 1] = img[:155, :150, 1]//2
    img[:155, :150, 2] = img[:155, :150, 1]//2

    im = Image.fromarray(img[:, :, ::-1])  # RGB
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", 30)
    draw.text((5, 5), '# {:d}'.format(frame_idx), fill=(255, 255, 255), font=font)

    if predict_score is not None:
        draw.text((5, 40), 'S: {:.2f}'.format(predict_score), fill=(255, 255, 255), font=font)
    if siam_score is not None:
        draw.text((5, 75), 'S: {:.2f}'.format(siam_score), fill=(255, 255, 255), font=font)
    if verifier_score is not None:
        draw.text((5, 110), 'V: {:.2f}'.format(verifier_score), fill=(255, 255, 255), font=font)
    if rpn_score is not None:
        draw.text((5, 145), 'R: {:.2f}'.format(rpn_score), fill=(255, 255, 255), font=font)

    # draw.text((5, 680), 'Mask', fill=(0, 255, 0), font=font)
    # draw.text((95, 680), 'GT', fill=(255, 0, 0), font=font)

    img = np.array(im)[:, :, ::-1]  # BGR
    cv2.imshow(tracker_name, img)
    cv2.waitKey(1)


# ==================================================================================================================== #


def get_subwindow_t(img, center_pos, size):
    """
    args:
        img(np.ndarray): BGR image
        bbox: (x, y, w, h) bbox
    """
    CONTEXT_AMOUNT = 0.5
    EXEMPLAR_SIZE = 127

    # calculate z crop size
    w_z = size[0] + CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + CONTEXT_AMOUNT * np.sum(size)
    s_z = round(np.sqrt(w_z * h_z))

    # calculate channle average
    channel_average = np.mean(img, axis=(0, 1))

    # get crop
    z_crop, _ = get_subwindow(img, center_pos,
                           EXEMPLAR_SIZE, s_z, channel_average)
    return z_crop


def get_subwindow_s(img, center_pos, size):
    """
    args:
        img(np.ndarray): BGR image
        bbox: (x, y, w, h) bbox
    """
    CONTEXT_AMOUNT = 0.5
    EXEMPLAR_SIZE = 127
    INSTANCE_SIZE = 255

    # calculate z crop size
    w_z = size[0] + CONTEXT_AMOUNT * np.sum(size)
    h_z = size[1] + CONTEXT_AMOUNT * np.sum(size)
    s_z = np.sqrt(w_z * h_z)

    scale_z = EXEMPLAR_SIZE / s_z
    s_x = s_z * (INSTANCE_SIZE / EXEMPLAR_SIZE)

    # calculate channle average
    channel_average = np.mean(img, axis=(0, 1))

    x_crop, resize_scale = get_subwindow(img, center_pos, INSTANCE_SIZE,
                           round(s_x), channel_average)

    return x_crop, scale_z, resize_scale


def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
    """
    args:
        im: bgr based image
        pos: center position
        model_sz: exemplar size
        s_z: original size
        avg_chans: channel average
    """
    CUDA = True

    if isinstance(pos, float):
        pos = [pos, pos]
    sz = original_sz
    im_sz = im.shape
    c = (original_sz + 1) / 2
    # context_xmin = round(pos[0] - c) # py2 and py3 round
    context_xmin = np.floor(pos[0] - c + 0.5)
    context_xmax = context_xmin + sz - 1
    # context_ymin = round(pos[1] - c)
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
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k), np.uint8)
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
        resize_scale = model_sz/original_sz
    else:
        resize_scale = 1

    if np.random.random() > 0.5:
        im_patch = im_patch[:, ::-1, :]

    im_patch = im_patch.transpose(2, 0, 1)
    im_patch = im_patch[np.newaxis, :, :, :]
    im_patch = im_patch.astype(np.float32)

    return im_patch, resize_scale
