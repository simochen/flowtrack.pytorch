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

class COCO(data.Dataset):
    def __init__(self, opt, split = 'train'):
        self.opt = opt
        self.num_joints = 17
        self.split = split
        if self.split == 'train':
            self.img_folder = os.path.join(opt.data_path,'images','train2017')
            jsonfile = os.path.join(opt.data_path,'json','coco_train.json')
        elif self.split == 'valid':
            self.img_folder = os.path.join(opt.data_path,'images','val2017')
            jsonfile = os.path.join(opt.data_path,'json','coco_val.json')
        if self.opt.with_mask:
            self.mask_folder = os.path.join(opt.data_path,'images','mask2017')

        # create train/val split
        with open(jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        # COCO R arm: 6(shoulder), 8(elbow), 10(wrist)
        #      L arm: 5(shoulder), 7(elbow), 9(wrist)
        #      R leg: 12(hip), 14(knee), 16(ankle)
        #      L leg: 11(hip), 13(knee), 15(ankle)
        #       face: 0(nose), 1(l-eye), 2(r-eye), 3(l-ear), 4(r-ear)

        self.mean = [103.9438, 113.9438, 119.8627]    # COCO
        # self.mean = [103.939, 116.779, 123.68]      # pretrained

        # self.mean, self.std = self._compute_mean()


    def _compute_mean(self):
        meanstd_file = 'coco_mean.pth.tar'
        if os.path.exists(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = np.zeros(3)
            std = np.zeros(3)
            with open('coco/annotations/person_keypoints_train2017.json') as f:
                anno = json.load(f)
            for image in anno['images']:
                img_path = os.path.join(self.img_folder, image['file_name'])
                img = cv2.imread(img_path)
                mean += img.reshape(-1, img.shape[-1]).mean(0)
                std += img.reshape(-1, img.shape[-1]).std(0)
            mean /= len(anno['images'])
            std /= len(anno['images'])
            meanstd = {
                'mean': mean,
                'std': std,
                }
            torch.save(meanstd, meanstd_file)
        if self.split == 'train':
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']


    def __getitem__(self, index):
        anno = self.anno[index]

        # load image
        img_path = os.path.join(self.img_folder, '{}.jpg'.format(anno['img_name']))
        img = cv2.imread(img_path)
        img = mean_sub(img, np.array(self.mean)).astype(np.float32)
        # img = torch.from_numpy(img.transpose((2,0,1))).float().div(255) # CxHxW

        if self.opt.with_mask:
            # load mask
            mask_path = os.path.join(self.mask_folder, 'mask_miss_{}.png'.format(anno['img_name']))
            mask = cv2.imread(mask_path, 0)[:,:,np.newaxis]
            img = np.concatenate((img, mask), axis=2)


        # preprocess
        # all joints in the image
        joints = torch.Tensor(anno['joint_self'])#.view(self.num_joints,3)
        # bbox = torch.Tensor(anno['bbox']).view(1,4) # [x,y,w,h]
        ref_scale = torch.Tensor([anno['segment_area']])

        h = img.shape[0]
        w = img.shape[1]

        scale = anno['scale']
        center = anno['objpos']
        rot = 0
        factor = 1
        out_h = self.opt.input_res[0]/self.opt.stride
        out_w = self.opt.input_res[1]/self.opt.stride
        if self.split == 'train':
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
        for j in range(self.num_joints):        # 17
            if joints[j,2] != 2:
                joint_hm[j] = draw_gaussian(joint_hm[j], [joints[j,0], joints[j,1]], sigma)
        heatmaps = joint_hm
        if self.opt.with_bg:
            bg_hm = torch.max(joint_hm, dim=0, keepdim=True)[0].neg_().add_(1)
            heatmaps = torch.cat((heatmaps, bg_hm))

        # if self.split == 'train':
        #     # Color Jitter
        #     img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        if self.split == 'train':
            return img, heatmaps
        else:
            return img, heatmaps, meta

    def __len__(self):
        return len(self.anno)
