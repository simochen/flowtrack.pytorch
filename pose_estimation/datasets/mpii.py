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
        if opt.num_limbs == 14:
            self.limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                          [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]
        elif opt.num_limbs == 17:
            self.limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                          [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13],
                          [8,11],[2,8],[5,11]]
        elif opt.num_limbs == 18:
            self.limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                          [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13],
                          [2,5],[8,11],[2,8],[5,11]]

        for idx, anno in enumerate(annolist):
            if (self.is_train and anno['isValidation'] == 0) or (not self.is_train and anno['isValidation'] == 1):
                self.anno.append(anno)

        self.mean = [110.3341, 113.2268, 112.3129]

        # self.mean, self.std = 0.5 * torch.ones(3), 0.5 * torch.ones(3)
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
        joints = torch.Tensor(anno['joint_self']).view(1,self.num_joints,3)
        headrect = torch.Tensor(anno['headrect_self']).view(1,4)
        ref_scale = compute_head_norm(headrect)
        if anno['numOtherPeople'] > 0:
            joint_dict = anno['joint_other']
            joint_other = torch.Tensor(joint_dict['_ArrayData_']).view(joint_dict['_ArraySize_'][::-1]).transpose(0,2)
            joints = torch.cat((joints, joint_other))
            headrect = torch.Tensor(anno['headrect_other']).view(-1,4)
            ref_scale = torch.cat((ref_scale, compute_head_norm(headrect)))
        joints[:,:,0:2].sub_(1) # Convert pts to zero based

        h = int(anno['img_height'])
        w = int(anno['img_width'])
        if self.is_train:
            # transform (scale, rotate, crop_pad)
            random.seed()
            scale = anno['scale_self'] * 200
            c = anno['objpos_self']
            #c = [c[0]-1, c[1]-1]   # 与后面transform处相抵消，不用-1
            rot = 0
            delta = [0, 0]
            # factor = scale / h
            # center = [c[0], h/2]
            factor = 1
            center = c
            #factor = scale / max(anno['img_height'], anno['img_width'])
            if random.random() < self.opt.rotate_prob:
                rot = (random.random()*2-1)*self.opt.rotate_degree_max
            delta[0] = round((random.random()*2-1)*self.opt.center_move_max[0])
            delta[1] = round((random.random()*2-1)*self.opt.center_move_max[1])
            if random.random() < self.opt.scale_prob:
                factor = random.random()*(self.opt.scale_max-self.opt.scale_min)+self.opt.scale_min
                center = c
            img = transform_image(img, center, scale, self.opt.input_res, factor, delta, rot)
            # joints transform
            jo = joints[:,:,0:2].numpy()
            jo = transform_point(jo, center, scale, self.opt.input_res, 0, factor, delta, rot)
            joints[:,:,0:2] = torch.from_numpy(jo/self.opt.stride)

            out_h = self.opt.input_res[0]/self.opt.stride
            out_w = self.opt.input_res[1]/self.opt.stride

            # flip
            if random.random() < self.opt.flip_prob:
                img = fliplr(img)
                joints = swaplr_joint(joints, out_w, self.opt.dataset)

        else:
            sf = max(float(self.opt.input_res[0])/h,float(self.opt.input_res[1])/w)
            img = cv2.resize(img, (0,0), fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
            img = torch.from_numpy(img.transpose((2,0,1)))
            # max_stride = int(self.opt.stride * math.pow(2, self.opt.depth))
            max_stride = self.opt.stride
            if img.size(1) % max_stride != 0:
                pad_h = max_stride - img.size(1) % max_stride
                # img = torch.cat((img, torch.FloatTensor(self.mean).view(-1,1,1).expand(img.size(0),pad_h,img.size(2))), dim=1)
                img = torch.cat((img, torch.zeros(img.size(0), pad_h, img.size(2))), dim=1)
            if img.size(2) % max_stride != 0:
                pad_w = max_stride - img.size(2) % max_stride
                # img = torch.cat((img, torch.FloatTensor(self.mean).view(-1,1,1).expand(img.size(0),img.size(1),pad_w)), dim=2)
                img = torch.cat((img, torch.zeros(img.size(0), img.size(1), pad_w)), dim=2)
            out_h = img.size(1)/self.opt.stride
            out_w = img.size(2)/self.opt.stride
            ref_scale.mul_(sf/self.opt.stride)
            joints[:,:,:2].mul_(sf/self.opt.stride)
            meta = {'joints':joints, 'ref_scale':ref_scale}

        # generate heatmaps
        sigma = self.opt.sigma
        joint_hm = torch.zeros(self.num_joints, out_h, out_w)
        limb_hm = torch.zeros(self.opt.num_limbs, out_h, out_w)
        for i in range(joints.size(0)):    # number of people
            # joint_hm
            for j in range(self.num_joints):        # 16
                if joints[i,j,2] != 2:
                    joint_hm[j] = draw_gaussian(joint_hm[j], [joints[i,j,0], joints[i,j,1]], sigma)
            # limb_hm
            for j in range(self.opt.num_limbs):
                pt1 = joints[i,self.limbs[j][0]]
                pt2 = joints[i,self.limbs[j][1]]
                if pt1[2] != 2 and pt2[2] != 2:
                    limb_hm[j] = draw_line_gaussian(limb_hm[j], [[pt1[0],pt1[1]],[pt2[0],pt2[1]]], sigma)

        heatmaps = torch.cat((joint_hm, limb_hm))
        if self.opt.with_bg:
            bg_hm = torch.max(joint_hm, dim=0, keepdim=True)[0].neg_().add_(1)
            heatmaps = torch.cat((heatmaps, bg_hm))
        #joint_hm = torch.cat((joint_hm, bg_hm))
        # joint_hm = joint_hm.gt(0.5).float()
        # limb_hm = limb_hm.gt(0.5).float()
        # heatmaps = heatmaps.gt(0.5).float()

        # img = color_normalize(img, self.mean, self.std)
        # Joints

        # if self.is_train:
        #     # Color Jitter?
        #     inp[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     inp[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     inp[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        if self.is_train:
            return img, heatmaps
        else:
            return img, heatmaps, meta

    def __len__(self):
        return len(self.anno)
