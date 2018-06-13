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

class AIC(data.Dataset):
    def __init__(self, opt, split='train'):
        self.split = split
        self.opt = opt
        self.num_joints = 14
        if self.split == 'train':
            self.img_folder = os.path.join(opt.data_path,'train','keypoint_train_images_20170902')
            # jsonfile = os.path.join(opt.data_path,'train','keypoint_train_annotations_20170909.json')
            jsonfile = os.path.join(opt.data_path, 'train_person.json')
        elif self.split == 'valid' or self.split == 'test_val':
            self.img_folder = os.path.join(opt.data_path,'val','keypoint_validation_images_20170911')
            jsonfile = os.path.join(opt.data_path,'val','keypoint_validation_annotations_20170911.json')
        elif self.split == 'test_a':
            self.img_folder = os.path.join(opt.data_path,'test','keypoint_test_a_images_20170923')
            jsonfile = os.path.join(opt.data_path,'test_a.json')

        with open(jsonfile) as anno_file:
            self.anno = json.load(anno_file)

        # AIC  R arms: 0(shoulder), 1(elbow), 2(wrist)
        #      L arms: 3(shoulder), 4(elbow), 5(wrist)
        #      R leg: 6(hip), 7(knee), 8(ankle)
        #      L leg: 9(hip), 10(knee), 11(ankle)
        #      12 - head top,  13 - neck
        if opt.num_limbs == 14:
            self.limbs = [[12,13],
                          [13,0],[0,1],[1,2],[13,3],[3,4],[4,5],
                          [0,6],[6,7],[7,8],[3,9],[9,10],[10,11],
                          [6,9]]

        self.mean, self.std = 0.5 * torch.ones(3), 0.5 * torch.ones(3)
        # self.mean, self.std = self._compute_mean()


    def _compute_mean(self):
        meanstd_file = 'aic_mean.pth.tar'
        if os.path.exists(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for anno in self.anno:
                img_path = os.path.join(self.img_folder, anno['image_id']+'.jpg')
                img = cv2.imread(img_path) / np.float32(255)
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
        img_path = os.path.join(self.img_folder, anno['image_id']+'.jpg')
        img = cv2.imread(img_path) / np.float32(255)

        h = img.shape[0]
        w = img.shape[1]

        # preprocess
        # all joints in the image
        if 'test' not in self.split:
            human_list = sorted(anno['human_annotations'].keys())
            bbox = torch.Tensor(0)
            for k in human_list:
                if len(bbox.size()) == 0:
                    bbox = torch.Tensor(anno['human_annotations'][k]).view(1,4)
                    joints = torch.Tensor(anno['keypoint_annotations'][k]).view(1,self.num_joints,3)
                else:
                    bbox = torch.cat((bbox, torch.Tensor(anno['human_annotations'][k]).view(1,4)))
                    joints = torch.cat((joints,torch.Tensor(anno['keypoint_annotations'][k]).view(1,self.num_joints,3)))
            ref_scale = compute_bbox_area(bbox)
            joints[:,:,0:2].sub_(1) # Convert pts to zero based

        if self.split == 'train':
            random.seed()

            # single-person bbox
            bbox_self = anno['human_annotations'][anno['person_index']]
            c = [(bbox_self[0]+bbox_self[2])/2, (bbox_self[1]+bbox_self[3])/2]
            scale = bbox_self[3] - bbox_self[1]
            sw = bbox_self[2] - bbox_self[0]

            # # multi-person bbox
            # # transform (scale, rotate, crop_pad)
            # # bbox.sub_(1)
            # ul = bbox[:,:2].min(dim=0)[0].squeeze()
            # br = bbox[:,2:].max(dim=0)[0].squeeze()
            # # ul = [max(0, ul[0]), max(0, ul[1])]
            # # br = [min(w, br[0]), min(h, br[1])]
            # c = [(ul[0]+br[0])/2, (ul[1]+br[1])/2]
            # scale = br[1] - ul[1]
            # sw = br[0] - ul[0]

            if scale == 0:
                print(anno['image_id'])
                print(bbox)
                scale = h
            #c = [c[0]-1, c[1]-1]   # 与后面transform处相抵消，不用-1

            rot = 0
            delta = [0, 0]
            # factor = scale / h
            # center = [c[0], h/2]
            factor = scale / max(scale, sw)
            center = c
            #factor = 200 * scale / max(anno['img_height'], anno['img_width'])
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
            img = cv2.resize(img, (0,0), fx=sf, fy=sf)
            img = torch.from_numpy(img.transpose((2,0,1)))
            # max_stride = int(self.opt.stride * math.pow(2, self.opt.depth))
            max_stride = self.opt.stride
            if img.size(1) % max_stride != 0:
                pad_h = max_stride - img.size(1) % max_stride
                img = torch.cat((img, torch.ones(img.size(0), pad_h, img.size(2)).mul_(0.5)), dim=1)
            if img.size(2) % max_stride != 0:
                pad_w = max_stride - img.size(2) % max_stride
                img = torch.cat((img, torch.ones(img.size(0), img.size(1), pad_w).mul_(0.5)), dim=2)
            if 'test' in self.split:
                return img, anno['image_id'], sf
            # validation
            out_h = img.size(1)/self.opt.stride
            out_w = img.size(2)/self.opt.stride
            ref_scale.mul_(sf*sf/self.opt.stride/self.opt.stride)
            joints[:,:,:2].mul_(sf/self.opt.stride)
            meta = {'joints':joints, 'ref_scale':ref_scale}

        # generate heatmaps
        sigma = self.opt.sigma
        joint_hm = torch.zeros(self.num_joints, out_h, out_w)
        limb_hm = torch.zeros(self.opt.num_limbs, out_h, out_w)
        for i in range(joints.size(0)):    # number of people
            # joint_hm
            for j in range(self.num_joints):        # 16
                if joints[i,j,2] == 1:
                    joint_hm[j] = draw_gaussian(joint_hm[j], [joints[i,j,0], joints[i,j,1]], sigma)
            # limb_hm
            for j in range(self.opt.num_limbs):
                pt1 = joints[i,self.limbs[j][0]]
                pt2 = joints[i,self.limbs[j][1]]
                if pt1[2] == 1 and pt2[2] == 1:
                    limb_hm[j] = draw_line_gaussian(limb_hm[j], [[pt1[0],pt1[1]],[pt2[0],pt2[1]]], sigma)
        bg_hm = torch.max(joint_hm, dim=0, keepdim=True)[0].neg_().add_(1)
        joint_hm = torch.cat((joint_hm, bg_hm))
        #heatmaps = torch.cat((joint_hm, bg_hm, limb_hm))

        # # =============== visualize ============
        # cv2.imshow('joint', joint_hm[0].numpy())
        # cv2.imshow('image', img.numpy().transpose((1,2,0)))
        # print(joint_hm[-1].max(), joint_hm[-1].min())
        # print(img.max(), img.min())
        # cv2.waitKey(0)
        # cv2.destroyWindow('image')
        # cv2.destroyWindow('joint')

        img = color_normalize(img, self.mean, self.std)
        # Joints

        # if self.is_train:
        #     # Color Jitter?
        #     inp[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     inp[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        #     inp[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        if self.split == 'train':
            return img, joint_hm, limb_hm
        elif self.split == 'valid':
            return img, joint_hm, limb_hm, meta

    def __len__(self):
        return len(self.anno)
