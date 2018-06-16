# -*- coding:utf-8 -*-
from __future__ import print_function

import os
import json
import random
import math
import numpy as np
import cv2

import torch
import torch.utils.data as data

from utils import *

class MPII(data.Dataset):
    def __init__(self, opt, split = 'train'):
        self.img_folder = os.path.join(opt.data_path, 'images')
        self.is_train = (split == 'train')       # training set or validation set
        self.opt = opt
        self.num_joints = 15
        jsonfile = os.path.join(opt.data_path, 'mpii_.json')

        # create train/val split
        with open(jsonfile) as anno_file:
            annolist = json.load(anno_file)['MPII']

        self.anno = []
        # img_list = []
        # sigma = opt.sigma

        # MPII R arms: 12(shoulder), 11(elbow), 10(wrist)
        #      L arms: 13(shoulder), 14(elbow), 15(wrist)
        #      R leg: 2(hip), 1(knee), 0(ankle)
        #      L leg: 3(hip), 4(knee), 5(ankle)
        #      6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top
        # MPII R arms: 2(shoulder), 3(elbow), 4(wrist)
        #      L arms: 5(shoulder), 6(elbow), 7(wrist)
        #      R leg: 8(hip), 9(knee), 10(ankle)
        #      L leg: 11(hip), 12(knee), 13(ankle)
        #      14 - pelvis-thorax, 1 - upper neck, 0 - head top

        for idx, anno in enumerate(annolist):
            if (self.is_train and anno['isValidation'] == 0) or (not self.is_train and anno['isValidation'] == 1):
                self.anno.append(anno)

        self.mean = [110.3341, 113.2268, 112.3129]

        # self.mean, self.std = self._compute_mean()


    def _compute_mean(self):
        meanstd_file = 'mpii_mean.pth.tar'
        if os.path.exists(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for anno in self.anno:
                img_path = os.path.join(self.img_folder, anno['img_name'])
                img = cv2.imread(img_path)
                img = torch.from_numpy(img.transpose((2,0,1)))
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.anno)
            std /= len(self.anno)
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']


    def __getitem__(self, index):
        anno = self.anno[index]

        # load image
        img_path = os.path.join(self.img_folder, anno['img_name'])
        img = cv2.imread(img_path)
        img = mean_sub(img, np.array(self.mean)).astype(np.float32)
        # img = torch.from_numpy(img.transpose((2,0,1))).float().div(255) # CxHxW

        # preprocess
        # all joints in the image
        joints = torch.Tensor(anno['joint_self'])#.view(self.num_joints,3)
        headrect = torch.Tensor(anno['headrect_self'])
        ref_scale = compute_head_norm(headrect)

        joints[:,:,0:2].sub_(1) # Convert pts to zero based

        h = int(anno['img_height'])
        w = int(anno['img_width'])

        scale = anno['scale'] * 200
        center = anno['objpos']
        #center = [center[0]-1, center[1]-1]
        rot = 0
        factor = 1
        out_h = self.opt.input_res[0]/self.opt.stride
        out_w = self.opt.input_res[1]/self.opt.stride
        if self.is_train:
            # transform (scale, rotate, crop_pad)
            random.seed()
            if random.random() < self.opt.rotate_prob:
                rot = (random.random()*2-1)*self.opt.rotate_degree_max
            if random.random() < self.opt.scale_prob:
                factor = random.random()*(self.opt.scale_max-self.opt.scale_min)+self.opt.scale_min
            img = transform_image(img, center, scale, self.opt.input_res, factor, rot)
            # joints transform
            jo = joints[:,0:2].numpy()
            jo = transform_point(jo, center, scale, self.opt.input_res, 0, factor, rot)
            joints[:,0:2] = torch.from_numpy(jo/self.opt.stride)

            out_h = self.opt.input_res[0]/self.opt.stride
            out_w = self.opt.input_res[1]/self.opt.stride

            # flip
            if random.random() < self.opt.flip_prob:
                img = fliplr(img)
                joints = swaplr_joint(joints, out_w, self.opt.dataset)

        else:
            img = transform_image(img, center, scale, self.opt.input_res, factor, rot)
            # joints transform
            jo = joints[:,0:2].numpy()
            jo = transform_point(jo, center, scale, self.opt.input_res, 0, factor, rot)
            joints[:,0:2] = torch.from_numpy(jo/self.opt.stride)

            sf = float(self.opt.input_res[0])/scale
            ref_scale.mul_(sf*sf/self.opt.stride/self.opt.stride)
            meta = {'joints':joints, 'ref_scale':ref_scale}

        # generate heatmaps
        sigma = self.opt.sigma
        joint_hm = torch.zeros(self.num_joints, out_h, out_w)
        # joint_hm
        for j in range(self.num_joints):        # 15
            if joints[j,2] != 2:
                joint_hm[j] = draw_gaussian(joint_hm[j], [joints[j,0], joints[j,1]], sigma)
        heatmaps = joint_hm
        if self.opt.with_bg:
            bg_hm = torch.max(joint_hm, dim=0, keepdim=True)[0].neg_().add_(1)
            heatmaps = torch.cat((heatmaps, bg_hm))
            

        # if self.is_train:
        #     # Color Jitter
        #     img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        if self.is_train:
            return img, heatmaps
        else:
            return img, heatmaps, meta

    def __len__(self):
        return len(self.anno)
