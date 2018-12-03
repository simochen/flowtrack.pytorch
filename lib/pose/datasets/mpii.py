# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import random
import math
import numpy as np
import cv2

import torch
import torch.utils.data as data
from collections import OrderedDict
from scipy.io import loadmat, savemat

from utils import *

class MPII(data.Dataset):
    def __init__(self, opt, split = 'train'):
        self.img_folder = os.path.join(opt.data_path, 'images')
        self.split = split       # training set or validation set
        self.opt = opt
        self.num_joints = 16
        jsonfile = os.path.join(opt.data_path, 'annotations', 'mpii_anno.json')

        # create train/val split
        with open(jsonfile) as f:
            annolist = json.load(f)

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
            if (self.split == 'train' and anno['isValidation'] == 0) or (self.split != 'train' and anno['isValidation'] == 1):
                self.anno.append(anno)

        # self.mean = [110.3341, 113.2268, 112.3129]
        # ImageNet BGR order
        self.mean = [0.406, 0.456, 0.485]
        self.std = [0.225, 0.224, 0.229]
        # self.mean = [103.939, 116.779, 123.68]

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
                img = cv2.imread(img_path)# / 255.0
                # img = img[:,:,::-1]
                mean += img.reshape(-1, img.shape[-1]).mean(0)
                std += img.reshape(-1, img.shape[-1]).std(0)
            mean /= len(self.anno)
            std /= len(self.anno)
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
        img_path = os.path.join(self.img_folder, anno['img_name'])
        img = cv2.imread(img_path)#.astype(np.float32)
        # img = img[:,:,::-1]
        # img = torch.from_numpy(img.transpose((2,0,1))).float().div(255) # CxHxW

        # preprocess
        # all joints in the image
        joints = np.array(anno['joint_self'])#.view(self.num_joints,3)
        # headrect = torch.Tensor(anno['headrect_self'])
        # ref_scale = compute_head_norm(headrect)
        ref_scale = torch.Tensor([anno['headsize']])

        keypoints = joints.copy()
        joints[:,:2] -= 1 # Convert pts to zero based

        h = img.shape[0]
        w = img.shape[1]

        scale = anno['scale'] * 200 * 1.25
        center = np.array(anno['objpos']) - 1
        if center[0] != -1:
            center[1] += 0.06 * scale
        rot = 0
        factor = 1
        out_h = self.opt.input_res[0]//self.opt.stride
        out_w = self.opt.input_res[1]//self.opt.stride
        if self.split == 'train':
            # flip
            if random.random() < self.opt.flip_prob:
                img = fliplr(img)
                joints = swaplr_joint(joints, w, self.opt.dataset)
                center[0] = w - 1 - center[0]
                if self.opt.debug:
                    print("Flip")

            # transform (scale, rotate, crop_pad)
            if random.random() < self.opt.rotate_prob:
                # rot = (random.random()*2-1)*self.opt.rotate_degree_max
                degree = self.opt.rotate_degree_max
                rot = np.clip(np.random.randn()*degree, -degree*2, degree*2)
            if random.random() < self.opt.scale_prob:
                # factor = random.random()*(self.opt.scale_max-self.opt.scale_min)+self.opt.scale_min
                sc = max(self.opt.scale_max-1, 1-self.opt.scale_min)
                factor = np.clip(np.random.randn()*sc + 1, self.opt.scale_min, self.opt.scale_max)
            if self.opt.debug:
                print("Scale:", factor)
                print("Rotate:", rot)
        else:
            meta = {
                'image': anno['img_name'],
                'joints': keypoints,
                'center': center,
                'scale': scale,
                'ref_scale':ref_scale
            }

        img = transform_image(img, center, scale, self.opt.input_res, factor, rot)

        img = normalize(img.float().div(255.0), self.mean, self.std)
        # img = mean_sub(img, self.mean)

        # joints transform
        joints[:,0:2] = transform_point(joints[:,0:2], center, scale, (out_h, out_w), 0, factor, rot)
        joints = torch.from_numpy(joints)

        # generate heatmaps
        sigma = self.opt.sigma
        joint_hm = torch.zeros(self.num_joints, out_h, out_w)
        # joint_hm
        for j in range(self.num_joints):        # 16
            if joints[j,2] > 0:
                joint_hm[j] = draw_gaussian(joint_hm[j], [joints[j,0], joints[j,1]], sigma)
        heatmaps = joint_hm
        if self.opt.with_bg:
            bg_hm = torch.max(joint_hm, dim=0, keepdim=True)[0].neg_().add_(1)
            heatmaps = torch.cat((heatmaps, bg_hm))
        target_weight = joints[:,2].gt(0).float()

        # if self.is_train:
        #     # Color Jitter
        #     img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        if self.split == 'train':
            return img, heatmaps, target_weight
        else:
            return img, heatmaps, target_weight, meta

    def __len__(self):
        return len(self.anno)

    def evaluate(self, preds, meta, out_dir):
        preds = preds[:, :, :2] + 1.0   # Nx16x2

        annos = meta['joints'][:,:,:2]  # Nx16x2
        is_visible = meta['joints'][:,:,2] > 0  # Nx16
        headsize = meta['ref_scale']    # N
        pred_file = os.path.join(out_dir, 'pred.mat')
        savemat(pred_file, mdict={'preds': preds})

        threshold = 0.5

        dists = np.linalg.norm(preds - annos, axis=2)   # Nx16
        dists = dists / headsize    # Nx16
        correct = (dists <= threshold) * is_visible
        joint_count = np.sum(is_visible, axis=0)    # 16
        PCKh = np.sum(correct, axis=0) * 100. / joint_count  # 16

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        joint_count = np.ma.array(joint_count, mask=False)
        joint_count.mask[6:8] = True
        joint_ratio = joint_count / np.sum(joint_count).astype(np.float64)

        results = OrderedDict([
            ('Head', PCKh[9]),
            ('Neck', PCKh[8]),
            ('Shoulder', 0.5 * (PCKh[12] + PCKh[13])),
            ('Elbow', 0.5 * (PCKh[11] + PCKh[14])),
            ('Wrist', 0.5 * (PCKh[10] + PCKh[15])),
            ('Hip', 0.5 * (PCKh[2] + PCKh[3])),
            ('Knee', 0.5 * (PCKh[1] + PCKh[4])),
            ('Ankle', 0.5 * (PCKh[0] + PCKh[5])),
            ('Mean', np.sum(PCKh * joint_ratio))
        ])

        return results, results['Mean']
