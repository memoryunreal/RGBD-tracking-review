# coding=utf-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from rtmdnet.options import opts
from rtmdnet.bbreg import BBRegressor
from rtmdnet.img_cropper import imgCropper
from rtmdnet.model import MDNet, BinaryLoss
from rtmdnet.utils import samples2maskroi
from rtmdnet.roi_align.modules.roi_align import RoIAlignAdaMax
from rtmdnet.sample_generator import SampleGenerator, gen_samples


class RTMD(object):
    def __init__(self, opt):
        self.opts = opt
        
        # Init model
        self.model = MDNet(self.opts['model_path'])
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

        self.pos_feats_all = None
        self.neg_feats_all = None
        self.pos_feats = None
        self.neg_feats = None
        self.feat_dim = None
        self.target_bbox = None

        self.frame_idx = 0

    def init(self, img, init_bbox):  # img [BGR]
        cur_image = img[:, :, ::-1]  # RGB
        # Init bbox
        self.target_bbox = np.array(init_bbox)  # [x y w h]

        # TODO Draw pos/neg samples
        ishape = cur_image.shape
        pos_examples = None  # [x y w h]
        neg_examples = None  # [x y w h]

        pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                                   self.target_bbox, self.opts['n_pos_init'], self.opts['overlap_pos_init'])
        neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 1, 2, 1.1),
                                   self.target_bbox, self.opts['n_neg_init'], self.opts['overlap_neg_init'])
        neg_examples = np.random.permutation(neg_examples)

        self.model.eval()

        # compute padded query box
        padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (self.opts['padding'] - 1.) / 2.).min()
        padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (self.opts['padding'] - 1.) / 2.).min()
        padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (self.opts['padding'] + 1.) / 2.).max()
        padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (self.opts['padding'] + 1.) / 2.).max()
        scene_boxes = np.array([padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1])

        # crop and resize query patch
        cur_resize_ratio = np.array([self.opts['crop_size'], self.opts['crop_size']]) / self.target_bbox[2:4]  # enlarge
        crop_img_size = (scene_boxes[2:4] * cur_resize_ratio).astype(int)
        cropped_image, cur_image_var = \
            self.img_crop_model.crop_image(cur_image, scene_boxes.reshape(1, 4), crop_img_size)
        cropped_image = cropped_image - 128.

        # query feature
        feat_map = self.model(cropped_image, out_layer='conv3')

        # --------------------------------------------- pos samples ---------------------------------------------
        # box in query patch
        cur_pos_rois = pos_examples.copy()
        cur_pos_rois[:, 0:2] -= scene_boxes[0:2]
        cur_pos_rois = samples2maskroi(cur_pos_rois, self.model.receptive_field,
                                       (self.opts['crop_size'], self.opts['crop_size']), self.target_bbox[2:4])

        # roi pooling
        batch_idx = np.zeros((pos_examples.shape[0], 1))
        cur_pos_rois = np.concatenate((batch_idx, cur_pos_rois), axis=1)
        cur_pos_rois = torch.from_numpy(cur_pos_rois.astype(np.float32)).cuda()
        cur_pos_feats = self.model.roi_align_model(feat_map, cur_pos_rois)
        cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

        # --------------------------------------------- neg samples ---------------------------------------------
        # box in query patch
        cur_neg_rois = neg_examples.copy()
        cur_neg_rois[:, 0:2] -= scene_boxes[0:2]
        cur_neg_rois = samples2maskroi(cur_neg_rois, self.model.receptive_field, 
                                       (self.opts['crop_size'], self.opts['crop_size']), self.target_bbox[2:4])

        # roi pooling
        batch_idx = np.zeros((neg_examples.shape[0], 1))
        cur_neg_rois = np.concatenate((batch_idx, cur_neg_rois), axis=1)
        cur_neg_rois = torch.from_numpy(cur_neg_rois.astype(np.float32)).cuda()
        cur_neg_feats = self.model.roi_align_model(feat_map, cur_neg_rois)
        cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

        # ------------------------------------------------------------------------------------------------------

        self.feat_dim = cur_pos_feats.size(-1)
        self.pos_feats = cur_pos_feats
        self.neg_feats = cur_neg_feats

        if self.pos_feats.size(0) > self.opts['n_pos_init']:
            pos_idx = np.array(range(self.pos_feats.size(0)))
            np.random.shuffle(pos_idx)
            self.pos_feats = self.pos_feats[pos_idx[0:self.opts['n_pos_init']], :]

        if self.neg_feats.size(0) > self.opts['n_neg_init']:
            neg_idx = np.array(range(self.neg_feats.size(0)))
            np.random.shuffle(neg_idx)
            self.neg_feats = self.neg_feats[neg_idx[0:self.opts['n_neg_init']], :]

        # TODO concatenate extra features to original_features
        # self.pos_feats = torch.cat((self.pos_feats, extra_pos_feats),dim=0)
        # self.neg_feats = torch.cat((self.neg_feats, extra_neg_feats), dim=0)

        torch.cuda.empty_cache()
        self.model.zero_grad()
    
        # Initial training
        self.train(self.init_optimizer, self.pos_feats, self.neg_feats, self.opts['maxiter_init'])

        # add all samples
        if self.pos_feats.size(0) > self.opts['n_pos_update']:
            pos_idx = np.arange(self.pos_feats.size(0))
            np.random.shuffle(pos_idx)
            self.pos_feats_all = [self.pos_feats[pos_idx[0:self.opts['n_pos_update']]]]

        if self.neg_feats.size(0) > self.opts['n_neg_update']:
            neg_idx = np.arange(self.neg_feats.size(0))
            np.random.shuffle(neg_idx)
            self.neg_feats_all = [self.neg_feats[neg_idx[0:self.opts['n_neg_update']]]]

    def track(self, img):
        self.frame_idx += 1
        cur_image = img[:, :, ::-1]  # BGR

        # TODO Estimate target bbox
        samples = None  # [x y w h] (256, 4)

        ishape = cur_image.shape
        samples = gen_samples(
            SampleGenerator('gaussian', (ishape[1], ishape[0]), self.trans_f, self.opts['scale_f'], valid=True),
            self.target_bbox, self.opts['n_samples'])

        padded_x1 = (samples[:, 0] - samples[:, 2] * (self.opts['padding']-1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (self.opts['padding']-1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (self.opts['padding']+1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (self.opts['padding']+1.) / 2.).max()
        scene_boxes = np.array([padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1])

        # crop and resize query patch
        cur_resize_ratio = np.array([self.opts['crop_size'], self.opts['crop_size']]) / self.target_bbox[2:4]
        crop_img_size = (scene_boxes[2:4] * cur_resize_ratio).astype(int)

        cropped_image, cur_image_var = \
            self.img_crop_model.crop_image(cur_image, scene_boxes.reshape(1, 4), crop_img_size)
        cropped_image = cropped_image - 128.

        self.model.eval()
        feat_map = self.model(cropped_image, out_layer='conv3')

        # --------------------------------------------- samples ---------------------------------------------
        # box in query patch
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= scene_boxes[0:2]
        sample_rois = samples2maskroi(sample_rois, self.model.receptive_field,
                                      (self.opts['crop_size'], self.opts['crop_size']), self.target_bbox[2:4])

        # roi pooling
        batch_idx = np.zeros((samples.shape[0], 1))
        sample_rois = np.concatenate((batch_idx, sample_rois), axis=1)
        sample_rois = torch.from_numpy(sample_rois.astype(np.float32)).cuda()
        sample_feats = self.model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()

        # ---------------------------------------------------------------------------------------------------

        # scores
        sample_scores = self.model(sample_feats, in_layer='fc4')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.detach().cpu().numpy()
        target_score = top_scores.detach().mean()
        self.target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > self.opts['success_thr']

        # Expand search area at failure
        if success:
            self.trans_f = self.opts['trans_f']
        else:
            self.trans_f = self.opts['trans_f_expand']

        bbox = self.target_bbox
        score = target_score

        # Data collect
        if success:

            # TODO Draw pos/neg samples
            pos_examples = None
            neg_examples = None

            # iou = UOF_overlap_ratio(bbreg_bbox, samples, mode='xywh')
            # pos_examples = samples[iou > 0.7, :]
            # neg_examples = samples[iou < 0.5, :]

            pos_examples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), self.target_bbox,
                self.opts['n_pos_update'],
                self.opts['overlap_pos_update'])
            neg_examples = gen_samples(
                SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), self.target_bbox,
                self.opts['n_neg_update'],
                self.opts['overlap_neg_update'])

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (self.opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (self.opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (self.opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (self.opts['padding'] + 1.) / 2.).max()
            scene_boxes = np.array([padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1])

            # crop and resize query patch
            # enlarge
            cur_resize_ratio = np.array([self.opts['crop_size'], self.opts['crop_size']]) / self.target_bbox[2:4]
            crop_img_size = (scene_boxes[2:4] * cur_resize_ratio).astype(int)
            cropped_image, cur_image_var = \
                self.img_crop_model.crop_image(cur_image, scene_boxes.reshape(1, 4), crop_img_size)
            cropped_image = cropped_image - 128.

            # query feature
            feat_map = self.model(cropped_image, out_layer='conv3')

            # --------------------------------------------- pos samples ---------------------------------------------
            # box in query patch
            cur_pos_rois = pos_examples
            cur_pos_rois[:, 0:2] -= scene_boxes[:2]
            cur_pos_rois = samples2maskroi(cur_pos_rois, self.model.receptive_field,
                                           (self.opts['crop_size'], self.opts['crop_size']),
                                           self.target_bbox[2:4])  # enlarge

            # roi pooling
            batch_idx = np.zeros((pos_examples.shape[0], 1))
            cur_pos_rois = np.concatenate((batch_idx, cur_pos_rois), axis=1)
            cur_pos_rois = torch.from_numpy(cur_pos_rois.astype(np.float32)).cuda()
            cur_pos_feats = self.model.roi_align_model(feat_map, cur_pos_rois)
            cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

            # --------------------------------------------- neg samples ---------------------------------------------
            # box in query patch
            cur_neg_rois = neg_examples
            cur_neg_rois[:, 0:2] -= scene_boxes[:2]
            cur_neg_rois = samples2maskroi(cur_neg_rois, self.model.receptive_field,
                                           (self.opts['crop_size'], self.opts['crop_size']),
                                           self.target_bbox[2:4])  # enlarge

            # roi pooling
            batch_idx = np.zeros((neg_examples.shape[0], 1))
            cur_neg_rois = np.concatenate((batch_idx, cur_neg_rois), axis=1)
            cur_neg_rois = torch.from_numpy(cur_neg_rois.astype(np.float32)).cuda()
            cur_neg_feats = self.model.roi_align_model(feat_map, cur_neg_rois)
            cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

            # -------------------------------------------------------------------------------------------------------

            self.feat_dim = cur_pos_feats.size(-1)

            self.pos_feats = cur_pos_feats  # index select
            self.neg_feats = cur_neg_feats

            if self.pos_feats.size(0) > self.opts['n_pos_update']:
                pos_idx = np.arange(self.pos_feats.size(0))
                np.random.shuffle(pos_idx)
                self.pos_feats = self.pos_feats[pos_idx[0:self.opts['n_pos_update']]]

            if self.neg_feats.size(0) > self.opts['n_neg_update']:
                neg_idx = np.arange(self.neg_feats.size(0))
                np.random.shuffle(neg_idx)
                self.neg_feats = self.neg_feats[neg_idx[0:self.opts['n_pos_update']]]

            self.pos_feats_all.append(self.pos_feats)
            self.neg_feats_all.append(self.neg_feats)

            if len(self.pos_feats_all) > self.opts['n_frames_long']:
                del self.pos_feats_all[0]
            if len(self.neg_feats_all) > self.opts['n_frames_short']:
                del self.neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(self.opts['n_frames_short'], len(self.pos_feats_all))
            pos_data = torch.cat(self.pos_feats_all[-nframes:], dim=0)
            neg_data = torch.cat(self.neg_feats_all, dim=0)
            self.train(self.update_optimizer, pos_data, neg_data, self.opts['maxiter_update'])

        # Long term update
        elif self.frame_idx % self.opts['long_interval'] == 0:
            pos_data = torch.cat(self.pos_feats_all, dim=0)
            neg_data = torch.cat(self.neg_feats_all, dim=0)
            self.train(self.update_optimizer, pos_data, neg_data, self.opts['maxiter_update'])

        return bbox, score

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

            if self.opts['visual_log']:
                print("Iter %d, Loss %.4f" % (iters, loss.item()))
