# coding=utf-8
import numpy as np

import torch
import torch.optim as optim

from rtmdnet.options import opts
from rtmdnet.img_cropper import imgCropper
from rtmdnet.model import MDNet, BinaryLoss
from rtmdnet.utils import samples2maskroi
from rtmdnet.roi_align.modules.roi_align import RoIAlignAdaMax
from rtmdnet.sample_generator import SampleGenerator, gen_samples

from utils import overlap_ratio
import cv2 as cv

class RTMD(object):
    def __init__(self, model_path):  # img [BGR]
        self.opts = opts
        
        # Init model
        self.model = MDNet(model_path)
        if self.opts['adaptive_align']:
            self.model.roi_align_model = RoIAlignAdaMax(3, 3, 1. / 8)
        if self.opts['use_gpu']:
            self.model = self.model.cuda()

        self.model.set_learnable_params(self.opts['ft_layers'])

        # Init image crop model
        self.img_crop_model = imgCropper(1.)
        if self.opts['use_gpu']:
            self.img_crop_model.gpuEnable()

        # Init criterion and optimizer
        self.criterion = BinaryLoss()
        self.init_optimizer = self.set_optimizer(self.opts['lr_init'])
        self.update_optimizer = self.set_optimizer(self.opts['lr_update'])

        self.trans_f = opts['trans_f']

        self.pos_feats_all = []
        self.neg_feats_all = []
        self.pos_feats = None
        self.neg_feats = None
        # self.target_bbox = None
        self.feat_dim = None

        self.frame_idx = 0

    def init(self, img, init_bbox):  # img [BGR] [x y w h]
        print("rtmd_local init come in")
        cur_image = img[:, :, ::-1]  # BGR --> RGB
        # Init bbox
        target_bbox = np.array(init_bbox)  # [x y w h]

        # Draw pos/neg samples  [x y w h]
        ishape = cur_image.shape

        neg_examples = gen_samples(
            SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
            self.opts['n_neg_init'],
            self.opts['overlap_neg_init'])
        pos_examples = gen_samples(
            SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
            self.opts['n_pos_init'],
            self.opts['overlap_pos_init'])

        neg_examples = np.random.permutation(neg_examples)

        # neg_examples[:, 2:] = neg_examples[:, 2:] + neg_examples[:, :2] - 1
        # boxes = neg_examples.astype(int)
        # for k in range(boxes.shape[0]):
        #     tmp_box = boxes[k]
        #     cv.rectangle(img, tuple(tmp_box[:2]), tuple(tmp_box[2:]), (0,255,0), 2)
        # pos_examples[:, 2:] = pos_examples[:, 2:] + pos_examples[:, :2] - 1
        # boxes = pos_examples.astype(int)
        # for k in range(boxes.shape[0]):
        #     tmp_box = boxes[k]
        #     cv.rectangle(img, tuple(tmp_box[:2]), tuple(tmp_box[2:]), (0,0,255), 2)
        # cv.imshow('', img)
        # cv.waitKey(0)

        # compute padded query box
        padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (self.opts['padding'] - 1.) / 2.).min()
        padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (self.opts['padding'] - 1.) / 2.).min()
        padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (self.opts['padding'] + 1.) / 2.).max()
        padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (self.opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.array([padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1]).reshape(1, 4)
        
        scene_boxes = padded_scene_box.copy()
        if opts['jitter']:
            # horizontal shift
            jittered_scene_box_horizon = padded_scene_box.copy()
            jittered_scene_box_horizon[0, 0] -= 4.
            jitter_scale_horizon = 1.

            # vertical shift
            jittered_scene_box_vertical = padded_scene_box.copy()
            jittered_scene_box_vertical[0, 1] -= 4.
            jitter_scale_vertical = 1.

            jittered_scene_box_reduce1 = padded_scene_box.copy()
            jitter_scale_reduce1 = 1.1 ** -1

            # vertical shift
            jittered_scene_box_enlarge1 = padded_scene_box.copy()
            jitter_scale_enlarge1 = 1.1 ** 1

            # scale reduction
            jittered_scene_box_reduce2 = padded_scene_box.copy()
            jitter_scale_reduce2 = 1.1 ** -2

            # scale enlarge
            jittered_scene_box_enlarge2 = padded_scene_box.copy()
            jitter_scale_enlarge2 = 1.1 ** 2

            scene_boxes = np.concatenate(
                [scene_boxes, jittered_scene_box_horizon, jittered_scene_box_vertical, jittered_scene_box_reduce1,
                 jittered_scene_box_enlarge1, jittered_scene_box_reduce2, jittered_scene_box_enlarge2], axis=0)
            jitter_scale = [1., jitter_scale_horizon, jitter_scale_vertical, jitter_scale_reduce1,
                            jitter_scale_enlarge1, jitter_scale_reduce2, jitter_scale_enlarge2]
        else:
            jitter_scale = [1.]

        # ------------------------------------------------------------------------------------------------------
       
        self.model.eval()
        for bidx in range(0, scene_boxes.shape[0]):
            # crop and resize query patch
            crop_img_size = \
                (scene_boxes[bidx, 2:4] * ((self.opts['crop_size'], self.opts['crop_size']) / target_bbox[2:4])).astype(int) \
                * jitter_scale[bidx]
            crop_img_size = np.maximum(crop_img_size, [80, 80])
            crop_img_size = np.minimum(crop_img_size, [3700, 3700])

            cropped_image, cur_image_var = \
                self.img_crop_model.crop_image(cur_image, scene_boxes[bidx].reshape(1, 4), crop_img_size)
            cropped_image = cropped_image - 128.

            # query feature
            feat_map = self.model(cropped_image, out_layer='conv3')

            rel_target_bbox = target_bbox.copy()
            rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

            scaled_obj_size = float(self.opts['crop_size']) * jitter_scale[bidx]
            # --------------------------------------------- pos samples ---------------------------------------------
            # box in query patch
            cur_pos_rois = pos_examples.copy()
            cur_pos_rois[:, 0:2] -= np.repeat(scene_boxes[bidx, 0:2].reshape(1, 2), cur_pos_rois.shape[0], axis=0)
            cur_pos_rois = samples2maskroi(cur_pos_rois, self.model.receptive_field,
                                           (scaled_obj_size, scaled_obj_size), target_bbox[2:4])

            # roi pooling
            batch_idx = np.zeros((pos_examples.shape[0], 1))
            cur_pos_rois = np.concatenate((batch_idx, cur_pos_rois), axis=1)
            cur_pos_rois = torch.from_numpy(cur_pos_rois.astype(np.float32)).cuda()
            cur_pos_feats = self.model.roi_align_model(feat_map, cur_pos_rois)
            # if np.random.rand() > 0.5:
            #     cur_pos_feats = torch.rot90(cur_pos_feats, 1, [2,3]).contiguous()
            cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

            # --------------------------------------------- neg samples ---------------------------------------------
            # box in query patch
            cur_neg_rois = neg_examples.copy()
            cur_neg_rois[:, 0:2] -= np.repeat(scene_boxes[bidx, 0:2].reshape(1, 2), cur_neg_rois.shape[0], axis=0)
            cur_neg_rois = samples2maskroi(cur_neg_rois, self.model.receptive_field,
                                           (scaled_obj_size, scaled_obj_size), target_bbox[2:4])

            # roi pooling
            batch_idx = np.zeros((neg_examples.shape[0], 1))
            cur_neg_rois = np.concatenate((batch_idx, cur_neg_rois), axis=1)
            cur_neg_rois = torch.from_numpy(cur_neg_rois.astype(np.float32)).cuda()
            cur_neg_feats = self.model.roi_align_model(feat_map, cur_neg_rois)
            cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

            feat_dim = cur_pos_feats.size(-1)

            if bidx == 0:
                pos_feats = cur_pos_feats  # 500*4608
                neg_feats = cur_neg_feats  # 5000*4608
            else:
                pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)
        
        # ==========================================================
        if pos_feats.size(0) > self.opts['n_pos_init']:
            pos_idx = np.asarray(range(pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            pos_feats = pos_feats[pos_idx[0:self.opts['n_pos_init']], :]

        if neg_feats.size(0) > self.opts['n_neg_init']:
            neg_idx = np.asarray(range(neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            neg_feats = neg_feats[neg_idx[0:self.opts['n_neg_init']], :]

        # concatenate extra features to original_features
        extra_obj_size = np.array((self.opts['crop_size'], self.opts['crop_size']))
        extra_crop_img_size = extra_obj_size * (self.opts['padding'] + 0.6)  # padding=1.2
        replicate_num = 100
        
        for iidx in range(replicate_num):
            extra_target_bbox = target_bbox.copy()

            extra_scene_box = np.copy(extra_target_bbox)
            extra_scene_box_center = extra_scene_box[0:2] + extra_scene_box[2:4] / 2.
            extra_scene_box_size = extra_scene_box[2:4] * (self.opts['padding'] + 0.6)
            extra_scene_box[0:2] = extra_scene_box_center - extra_scene_box_size / 2.
            extra_scene_box[2:4] = extra_scene_box_size

            extra_shift_offset = np.clip(2. * np.random.randn(2), -4, 4)
            cur_extra_scale = 1.1 ** np.clip(np.random.randn(1), -2, 2)

            extra_scene_box[0] += extra_shift_offset[0]
            extra_scene_box[1] += extra_shift_offset[1]
            extra_scene_box[2:4] *= cur_extra_scale[0]

            scaled_obj_size = float(self.opts['crop_size']) / cur_extra_scale[0]

            cur_extra_cropped_image, _ = self.img_crop_model.crop_image(
                cur_image, np.reshape(extra_scene_box, (1, 4)), extra_crop_img_size)
            cur_extra_cropped_image = cur_extra_cropped_image.detach()

            cur_extra_pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                                                 extra_target_bbox, self.opts['n_pos_init'] / replicate_num,
                                                 self.opts['overlap_pos_init'])
            cur_extra_neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 2, 1.1),
                                                 extra_target_bbox, self.opts['n_neg_init'] / replicate_num / 4,
                                                 self.opts['overlap_neg_init'])

            #
            batch_num = iidx * np.ones((cur_extra_pos_examples.shape[0], 1))
            cur_extra_pos_rois = cur_extra_pos_examples.copy()
            cur_extra_pos_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                    cur_extra_pos_rois.shape[0], axis=0)
            cur_extra_pos_rois = samples2maskroi(cur_extra_pos_rois, self.model.receptive_field,
                                                 (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                                 self.opts['padding'])
            cur_extra_pos_rois = np.concatenate((batch_num, cur_extra_pos_rois), axis=1)

            #
            batch_num = iidx * np.ones((cur_extra_neg_examples.shape[0], 1))
            cur_extra_neg_rois = np.copy(cur_extra_neg_examples)
            cur_extra_neg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                    cur_extra_neg_rois.shape[0], axis=0)
            cur_extra_neg_rois = samples2maskroi(cur_extra_neg_rois, self.model.receptive_field,
                                                 (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                                 self.opts['padding'])
            cur_extra_neg_rois = np.concatenate((batch_num, cur_extra_neg_rois), axis=1)

            if iidx == 0:
                extra_cropped_image = cur_extra_cropped_image

                extra_pos_rois = cur_extra_pos_rois.copy()
                extra_neg_rois = cur_extra_neg_rois.copy()
            else:
                extra_cropped_image = torch.cat((extra_cropped_image, cur_extra_cropped_image), dim=0)

                extra_pos_rois = np.concatenate((extra_pos_rois, cur_extra_pos_rois.copy()), axis=0)
                extra_neg_rois = np.concatenate((extra_neg_rois, cur_extra_neg_rois.copy()), axis=0)
        
        extra_pos_rois = torch.from_numpy(extra_pos_rois.astype('float32')).cuda()  # pos*500
        extra_neg_rois = torch.from_numpy(extra_neg_rois.astype('float32')).cuda()  # neg*1200

        extra_cropped_image -= 128.
        extra_feat_maps = self.model(extra_cropped_image, out_layer='conv3')
        
        extra_pos_feats = self.model.roi_align_model(extra_feat_maps, extra_pos_rois)
        extra_pos_feats = extra_pos_feats.view(extra_pos_feats.size(0), -1).data.clone()

        extra_neg_feats = self.model.roi_align_model(extra_feat_maps, extra_neg_rois)
        extra_neg_feats = extra_neg_feats.view(extra_neg_feats.size(0), -1).data.clone()

        pos_feats = torch.cat((pos_feats, extra_pos_feats), dim=0)  # pos*1000
        neg_feats = torch.cat((neg_feats, extra_neg_feats), dim=0)  # neg*6200
       
        # -------------------------------------------------------------------------------------------------------

        torch.cuda.empty_cache()
        self.model.zero_grad()
     
        # Initial training
        self.train(self.init_optimizer, pos_feats, neg_feats, self.opts['maxiter_init'])
    
        # add all samples  pos*1000 neg*6200 只保留 pos*50 neg*200
        if pos_feats.size(0) > self.opts['n_pos_update']:
            pos_idx = np.arange(pos_feats.size(0))
            np.random.shuffle(pos_idx)
            self.pos_feats_all = [pos_feats[pos_idx[0:self.opts['n_pos_update']]]]

        if neg_feats.size(0) > self.opts['n_neg_update']:
            neg_idx = np.arange(neg_feats.size(0))
            np.random.shuffle(neg_idx)
            self.neg_feats_all = [neg_feats[neg_idx[0:self.opts['n_neg_update']]]]

        self.pos_feats_1st = [pos_feats[pos_idx[0:self.opts['n_pos_update']]]]
        self.neg_feats_1st = [neg_feats[neg_idx[0:self.opts['n_neg_update']]]]

    def inference(self, img, target_bbox, proposals=None, thres=None):  # [x y w h] [x y x y]
        if thres is not None:
            tmp_thres = thres
        else:
            tmp_thres = 0
        cur_image = img[:, :, ::-1]  # BGR --> RGB

        # TODO Estimate target bbox
        ishape = cur_image.shape
        if proposals is not None and proposals.size > 0:
            proposals = proposals.reshape(-1, 4)
            samples = np.array(proposals)  # [x y x y]
            samples[:, 2:] = samples[:, 2:] - samples[:, :2] + 1  # [x y x y] to [x y w h]

            # samples2 = gen_samples(  # [x y w h]
            #     SampleGenerator('gaussian', (ishape[1], ishape[0]), self.trans_f, self.opts['scale_f'], valid=True),
            #     target_bbox, int(self.opts['n_samples']))
            # samples = np.concatenate((samples, samples2), axis=0)
        else:
            samples = gen_samples(  # [x y w h]
                SampleGenerator('gaussian', (ishape[1], ishape[0]), self.trans_f, self.opts['scale_f'], valid=True),
                target_bbox, int(self.opts['n_samples']))

        padded_x1 = (samples[:, 0] - samples[:, 2] * (self.opts['padding']-1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (self.opts['padding']-1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (self.opts['padding']+1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (self.opts['padding']+1.) / 2.).max()
        scene_boxes = np.array([padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1])

        if scene_boxes[0] > cur_image.shape[1]:
            scene_boxes[0] = cur_image.shape[1]-1
        if scene_boxes[1] > cur_image.shape[0]:
            scene_boxes[1] = cur_image.shape[0]-1
        if scene_boxes[0] + scene_boxes[2] < 0:
            scene_boxes[2] = -scene_boxes[0]+1
        if scene_boxes[1] + scene_boxes[3] < 0:
            scene_boxes[3] = -scene_boxes[1]+1

        # crop and resize query patch
        cur_resize_ratio = np.array([self.opts['crop_size'], self.opts['crop_size']]) / target_bbox[2:4]
        crop_img_size = (scene_boxes[2:4] * cur_resize_ratio).astype(int)
        crop_img_size = np.maximum(crop_img_size, [80, 80])
        crop_img_size = np.minimum(crop_img_size, [3700, 3700])

        cropped_image, cur_image_var = \
            self.img_crop_model.crop_image(cur_image, scene_boxes.reshape(1, 4), crop_img_size)
        cropped_image = cropped_image - 128.

        self.model.eval()
        feat_map = self.model(cropped_image, out_layer='conv3')

        # --------------------------------------------- samples ---------------------------------------------
        # box in query patch
        sample_rois = samples.copy()
        sample_rois[:, 0:2] -= scene_boxes[:2]
        sample_rois = samples2maskroi(sample_rois, self.model.receptive_field,
                                      (self.opts['crop_size'], self.opts['crop_size']), target_bbox[2:4])

        # roi pooling
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = torch.from_numpy(sample_rois.astype(np.float32)).cuda()
        sample_feats = self.model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        
        # -------------------------------------------------------------------------------------------------------

        # scores
        sample_scores = self.model(sample_feats, in_layer='fc4')

        if samples.shape[0] > 1:
            top_scores, top_idx = sample_scores[:, 1].topk(5)
            top_idx = top_idx.data.cpu().numpy()
            target_score = top_scores.data.mean().item()
            target_bbox = samples[top_idx].mean(axis=0)
        else:
            top_scores, top_idx = sample_scores[:, 1].topk(1)
            top_idx = top_idx.data.cpu().numpy()
            target_score = top_scores.data.mean().item()
            target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > tmp_thres

        # # Expand search area at failure
        if success:
            self.trans_f = opts['trans_f']
        else:
            self.trans_f = opts['trans_f_expand']

        target_bbox = target_bbox.reshape(-1)
        target_bbox[2:] = target_bbox[2:] + target_bbox[:2] - 1  # [x y w h] to [x y x y]
        return target_bbox, target_score, success  # [x y x y]

    def eval(self, img, target_bbox, proposals, thres=None):  # [x y w h] [x y x y]
        if thres is not None:
            tmp_thres = thres
        else:
            tmp_thres = 0
        cur_image = img[:, :, ::-1]  # BGR --> RGB

        # TODO Estimate target bbox
        ishape = cur_image.shape

        proposals = proposals.reshape(-1, 4)
        samples = np.array(proposals)  # [x y x y]
        samples[:, 2:] = samples[:, 2:] - samples[:, :2] + 1  # [x y x y] to [x y w h]

        padded_x1 = (samples[:, 0] - samples[:, 2] * (self.opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (self.opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (self.opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (self.opts['padding'] + 1.) / 2.).max()
        scene_boxes = np.array([padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1])

        if scene_boxes[0] > cur_image.shape[1]:
            scene_boxes[0] = cur_image.shape[1] - 1
        if scene_boxes[1] > cur_image.shape[0]:
            scene_boxes[1] = cur_image.shape[0] - 1
        if scene_boxes[0] + scene_boxes[2] < 0:
            scene_boxes[2] = -scene_boxes[0] + 1
        if scene_boxes[1] + scene_boxes[3] < 0:
            scene_boxes[3] = -scene_boxes[1] + 1

        # crop and resize query patch
        cur_resize_ratio = np.array([self.opts['crop_size'], self.opts['crop_size']]) / target_bbox[2:4]
        crop_img_size = (scene_boxes[2:4] * cur_resize_ratio).astype(int)
        crop_img_size = np.maximum(crop_img_size, [80, 80])
        crop_img_size = np.minimum(crop_img_size, [3700, 3700])

        cropped_image, cur_image_var = \
            self.img_crop_model.crop_image(cur_image, scene_boxes.reshape(1, 4), crop_img_size)
        cropped_image = cropped_image - 128.

        self.model.eval()
        feat_map = self.model(cropped_image, out_layer='conv3')

        # --------------------------------------------- samples ---------------------------------------------
        # box in query patch
        sample_rois = samples.copy()
        sample_rois[:, 0:2] -= scene_boxes[:2]
        sample_rois = samples2maskroi(sample_rois, self.model.receptive_field,
                                      (self.opts['crop_size'], self.opts['crop_size']), target_bbox[2:4])

        # roi pooling
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = torch.from_numpy(sample_rois.astype(np.float32)).cuda()
        sample_feats = self.model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()

        # -------------------------------------------------------------------------------------------------------

        # scores
        sample_scores = self.model(sample_feats, in_layer='fc4')

        top_scores = sample_scores[:, 1]
        target_score = top_scores.detach().cpu().numpy()

        sort_idx = np.argsort(-target_score)
        target_bbox = samples[sort_idx]
        target_score = target_score[sort_idx]
        success = target_score.max() > tmp_thres

        # # Expand search area at failure
        if success:
            self.trans_f = opts['trans_f']
        else:
            self.trans_f = opts['trans_f_expand']

        target_bbox = target_bbox.reshape(-1, 4)
        target_bbox[:, 2:] = target_bbox[:, 2:] + target_bbox[:, :2] - 1  # [x y w h] to [x y x y]
        return target_bbox, target_score, success  # [x y x y]

    def collect(self, img, target_box, succ_flag, neg_s):  # [x y x y] [x y x y]
        cur_image = img[:, :, ::-1]  # BGR --> RGB
        target_bbox = np.array(target_box)  # [x y x y]
        target_bbox[2:] = target_bbox[2:] - target_bbox[:2] + 1

        # Data collect
        # TODO Draw pos/neg samples
        if succ_flag:

            ishape = cur_image.shape
            pos_examples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
                self.opts['n_pos_update'],
                self.opts['overlap_pos_update'])

            neg_examples = np.array(neg_s)  # [x y x y]
            neg_examples[:, 2:] = neg_examples[:, 2:] - neg_examples[:, :2] + 1  # [x y x y] to [x y w h]

            neg_examples2 = gen_samples(
                SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
                int(self.opts['n_neg_update']-neg_examples.shape[0]),
                self.opts['overlap_neg_update'])
            neg_examples = np.concatenate((neg_examples2, neg_examples), axis=0)

            neg_examples = np.random.permutation(neg_examples)

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (self.opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (self.opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (self.opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (self.opts['padding'] + 1.) / 2.).max()
            padded_scene_box = \
                np.array([padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1]).reshape(1, 4)
            scene_boxes = padded_scene_box.copy()

            # crop and resize query patch
            # enlarge
            cur_resize_ratio = np.array([self.opts['crop_size'], self.opts['crop_size']]) / target_bbox[2:4]
            crop_img_size = (scene_boxes[0, 2:4] * cur_resize_ratio).astype(int)
            crop_img_size = np.maximum(crop_img_size, [80, 80])
            crop_img_size = np.minimum(crop_img_size, [3700, 3700])

            cropped_image, cur_image_var = \
                self.img_crop_model.crop_image(cur_image, scene_boxes.reshape(1, 4), crop_img_size)
            cropped_image = cropped_image - 128.

            # query feature
            feat_map = self.model(cropped_image, out_layer='conv3')

            # --------------------------------------------- pos samples ---------------------------------------------
            # box in query patch
            cur_pos_rois = pos_examples.copy()
            cur_pos_rois[:, 0:2] -= scene_boxes[0, :2]
            cur_pos_rois = samples2maskroi(cur_pos_rois, self.model.receptive_field,
                                           (self.opts['crop_size'], self.opts['crop_size']), target_bbox[2:4])

            # roi pooling
            batch_idx = np.zeros((pos_examples.shape[0], 1))
            cur_pos_rois = np.concatenate((batch_idx, cur_pos_rois), axis=1)
            cur_pos_rois = torch.from_numpy(cur_pos_rois.astype(np.float32)).cuda()
            cur_pos_feats = self.model.roi_align_model(feat_map, cur_pos_rois)
            # if np.random.rand() > 0.5:
            #     cur_pos_feats = torch.rot90(cur_pos_feats, 1, [2,3]).contiguous()
            cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

            # --------------------------------------------- neg samples ---------------------------------------------
            # box in query patch
            cur_neg_rois = neg_examples.copy()
            cur_neg_rois[:, 0:2] -= scene_boxes[0, :2]
            cur_neg_rois = samples2maskroi(cur_neg_rois, self.model.receptive_field,
                                           (self.opts['crop_size'], self.opts['crop_size']), target_bbox[2:4])

            # roi pooling
            batch_idx = np.zeros((neg_examples.shape[0], 1))
            cur_neg_rois = np.concatenate((batch_idx, cur_neg_rois), axis=1)
            cur_neg_rois = torch.from_numpy(cur_neg_rois.astype(np.float32)).cuda()
            cur_neg_feats = self.model.roi_align_model(feat_map, cur_neg_rois)
            cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

            # -------------------------------------------------------------------------------------------------------

            self.feat_dim = cur_pos_feats.size(-1)

            pos_feats = cur_pos_feats  # index select
            neg_feats = cur_neg_feats

            if pos_feats.size(0) > self.opts['n_pos_update']:  # 50
                pos_idx = np.arange(pos_feats.size(0))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats[pos_idx[0:self.opts['n_pos_update']]]

            if neg_feats.size(0) > self.opts['n_neg_update']:  # 200
                neg_idx = np.arange(neg_feats.size(0))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats[neg_idx[0:self.opts['n_pos_update']]]

            self.pos_feats_all.append(pos_feats)
            self.neg_feats_all.append(neg_feats)

            if len(self.pos_feats_all) > self.opts['n_frames_long']:  # 100 frames
                self.pos_feats_all.pop(0)
            if len(self.neg_feats_all) > self.opts['n_frames_short']:  # 20 frames
                self.neg_feats_all.pop(0)

    def update(self, succ_flag, frame_idx, interval=None):
        if interval is None:
            interval = self.opts['long_interval']

        # Short term update
        if not succ_flag:
            nframes = min(self.opts['n_frames_short'], len(self.pos_feats_all))
            pos_data = torch.cat(self.pos_feats_all[-nframes:], dim=0)
            neg_data = torch.cat(self.neg_feats_all, dim=0)
            self.train(self.update_optimizer, pos_data, neg_data, self.opts['maxiter_update'])

        # Long term update
        elif frame_idx % interval == 0:
            pos_data = torch.cat(self.pos_feats_all[:], dim=0)
            neg_data = torch.cat(self.neg_feats_all, dim=0)
            self.train(self.update_optimizer, pos_data, neg_data, self.opts['maxiter_update'])

    def update_all(self, succ_flag):

        # Short term update
        if succ_flag:
            pos_data = torch.cat(self.pos_feats_all[:], dim=0)
            neg_data = torch.cat(self.neg_feats_all, dim=0)
            self.train(self.update_optimizer, pos_data, neg_data, self.opts['maxiter_update'])

    def set_optimizer(self, lr_base):
        lr = lr_base
        lr_mult = self.opts['lr_mult']

        params = self.model.get_learnable_params()
        param_list = []
        for k, p in params.items():
            lr = lr_base
            for l, m in lr_mult.items():
                if k.startswith(l):
                    lr = lr_base * m
            param_list.append({'params': [p], 'lr': lr})

        optimizer = optim.SGD(param_list, lr=lr, momentum=self.opts['momentum'], weight_decay=self.opts['w_decay'])
        return optimizer

    def train(self, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
        self.model.train()

        batch_pos = self.opts['batch_pos']
        batch_neg = self.opts['batch_neg']
        batch_test = self.opts['batch_test']
        batch_neg_cand = max(self.opts['batch_neg_cand'], batch_neg)

        pos_idx = np.random.permutation(pos_feats.size(0))
        neg_idx = np.random.permutation(neg_feats.size(0))

        while len(pos_idx) < batch_pos * maxiter:
            pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
        while len(neg_idx) < batch_neg_cand * maxiter:
            neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
        pos_pointer = 0
        neg_pointer = 0

        for iters in range(maxiter):

            # select pos idx
            pos_next = pos_pointer+batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            # pos_cur_idx = torch.from_numpy(pos_cur_idx).long().cuda()
            pos_pointer = pos_next

            # select neg idx
            neg_next = neg_pointer+batch_neg_cand
            neg_cur_idx = neg_idx[neg_pointer:neg_next]
            # neg_cur_idx = torch.from_numpy(neg_cur_idx).long().cuda()
            neg_pointer = neg_next

            # create batch
            batch_pos_feats = pos_feats[pos_cur_idx, :]
            batch_neg_feats = neg_feats[neg_cur_idx, :]

            # hard negative mining
            if batch_neg_cand > batch_neg:
                self.model.eval()  # model transfer into evaluation mode
                neg_cand_score = None
                for start in range(0, batch_neg_cand, batch_test):
                    end = min(start+batch_test, batch_neg_cand)
                    score = self.model(batch_neg_feats[start:end], in_layer=in_layer)
                    if start == 0:
                        neg_cand_score = score.data[:, 1].clone()
                    else:
                        neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

                _, top_idx = neg_cand_score.topk(batch_neg)
                batch_neg_feats = batch_neg_feats.index_select(0, top_idx)
                self.model.train()  # model transfer into train mode

            # forward
            pos_score = self.model(batch_pos_feats, in_layer=in_layer)
            neg_score = self.model(batch_neg_feats, in_layer=in_layer)

            # optimize
            loss = self.criterion(pos_score, neg_score)
            self.model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts['grad_clip'])
            optimizer.step()

            # print("Iter %d, Loss %.4f" % (iters, loss.item()))
