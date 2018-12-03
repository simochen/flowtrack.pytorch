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

from pycocotools.coco import COCO as COCOtool
from pycocotools.cocoeval import COCOeval
from collections import defaultdict, OrderedDict
from utils import *

class COCO(data.Dataset):
    def __init__(self, opt, split = 'train'):
        self.opt = opt
        self.num_joints = 17
        self.split = split
        # self.split = 'train' / 'val'
        self.img_folder = os.path.join(opt.data_path,'images','%s2017' % self.split)
        jsonfile = os.path.join(opt.data_path,'json','coco_%s_single.json' % self.split)
        annofile = os.path.join(opt.data_path,'annotations','person_keypoints_%s2017.json' % self.split)
        if self.opt.with_mask:
            self.mask_folder = os.path.join(opt.data_path,'images','mask2017')

        # create train/val split
        with open(jsonfile) as f:
            self.anno = json.load(f)['coco']

        self.coco = COCOtool(annofile)

        # COCO R arm: 6(shoulder), 8(elbow), 10(wrist)
        #      L arm: 5(shoulder), 7(elbow), 9(wrist)
        #      R leg: 12(hip), 14(knee), 16(ankle)
        #      L leg: 11(hip), 13(knee), 15(ankle)
        #       face: 0(nose), 1(l-eye), 2(r-eye), 3(l-ear), 4(r-ear)

        # self.mean = [103.9438, 113.9438, 119.8627]    # COCO
        # ImageNet BGR order
        # self.mean = [0.406, 0.456, 0.485]
        self.std = [0.225, 0.224, 0.229]
        self.mean = [103.939, 116.779, 123.68]

        # self.mean, self.std = self._compute_mean()


    def _compute_mean(self):
        meanstd_file = 'coco_mean.pth.tar'
        if os.path.exists(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = np.zeros(3)
            std = np.zeros(3)
            with open(os.path.join(self.opt.data_path,'annotations','person_keypoints_train2017.json')) as f:
                anno = json.load(f)
            for image in anno['images']:
                img_path = os.path.join(self.img_folder, image['file_name'])
                img = cv2.imread(img_path)# / 255.0
                # img = img[:,:,::-1]
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
        img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # img = img[:,:,::-1]
        # img = torch.from_numpy(img.transpose((2,0,1))).float().div(255) # CxHxW

        # load mask
        if self.opt.with_mask:
            mask_path = os.path.join(self.mask_folder, 'mask_miss_{}.png'.format(anno['img_name']))
            mask = cv2.imread(mask_path, 0)[:,:,np.newaxis]
            img = np.concatenate((img, mask), axis=2)
            self.mean.append(0)
            self.std.append(1)

        # preprocess
        # all joints in the image
        joints = np.array(anno['joint_self'])#.view(self.num_joints,3)
        # bbox = torch.Tensor(anno['bbox']).view(1,4) # [x,y,w,h]
        ref_scale = torch.Tensor([anno['segment_area']])

        h = img.shape[0]
        w = img.shape[1]

        # scale = anno['scale'] * 1.25
        pheight = anno['bbox'][3]
        pwidth = anno['bbox'][2]
        scale = max(pheight, pwidth*self.opt.input_res[0]/self.opt.input_res[1]) * 1.25
        center = np.array(anno['objpos'])
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
                'image': int(anno['img_name']),
                #'joints': joints,
                'center': center,
                'scale': scale,
                'score': 1,
                'area': pheight*pwidth,
                'ref_scale':ref_scale
            }

        img = transform_image(img, center, scale, self.opt.input_res, factor, rot)

        img = normalize(img.float().div(255.0), self.mean, self.std)
        # img = mean_sub(img, np.array(self.mean))

        # joints transform
        joints[:,0:2] = transform_point(joints[:,0:2], center, scale, (out_h, out_w), 0, factor, rot)
        joints = torch.from_numpy(joints)

        # integer coords
        joints = joints.round().int()

        # generate heatmaps
        sigma = self.opt.sigma
        joint_hm = torch.zeros(self.num_joints, out_h, out_w)
        # joint_hm
        for j in range(self.num_joints):        # 17
            if joints[j,2] != 0:
                joint_hm[j] = draw_gaussian(joint_hm[j], [joints[j,0], joints[j,1]], sigma)
        heatmaps = joint_hm
        if self.opt.with_bg:
            bg_hm = torch.max(joint_hm, dim=0, keepdim=True)[0].neg_().add_(1)
            heatmaps = torch.cat((heatmaps, bg_hm))

        if self.opt.debug:
            maps = torch.sum(heatmaps, dim=0, keepdim=False).numpy()
            target = cv2.resize(maps, (self.opt.input_res[1], self.opt.input_res[0]))
            cv2.imshow('image', get_img(img, np.array(self.mean), np.array(self.std)) + target[:,:,np.newaxis])
            cv2.waitKey(0)
            cv2.destroyWindow('image')

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

    def _write_results(self, keypoints, filename):
        results = []
        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue
            kpts = np.array([img_kpts[i]['joints'] for i in range(len(img_kpts))])
            kpts = kpts.reshape((kpts.shape[0], -1))
            result = [{'image_id': img_kpts[i]['image'],
                        'category_id': 1,
                        'keypoints': kpts[i].tolist(),
                        'score': img_kpts[i]['score']
                       } for i in range(len(img_kpts))]
            results.extend(result)
        with open(filename, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(filename))
        except Exception:
            content = []
            with open(filename, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(filename, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_eval(self, res_file):
        coco_dt = self.coco.loadRes(res_file)
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        keys = ["AP", "AP@.5", "AP@.75", "AP@M", "AP@L", "AR", "AR@.5", "AR@.75", "AR@M", "AR@L"]
        return dict(zip(keys, coco_eval.stats))

    def evaluate(self, preds, meta, out_dir):
        delta = 2 * np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        res_folder = os.path.join(out_dir, 'results')
        if not os.path.exists(res_folder):
            os.makedirs(res_folder)
        res_file = os.path.join(
            res_folder, 'keypoints_%s_results.json' % self.split)

        _keypoints = []
        for i in range(len(preds)):
            _keypoints.append({
                'joints': preds[i],
                'area': meta['area'][i],
                'score': meta['score'][i],
                'image': meta['image'][i]
            })
        keypoints = defaultdict(list)
        for kpt in _keypoints:
            keypoints[kpt['image']].append(kpt)
        nms_keypoints = []
        for img in keypoints.keys():
            img_kpts = keypoints[img]
            for kpt in img_kpts:
                box_score = kpt['score']
                kpt_score = 0
                num_kpts = 0
                for j in range(self.num_joints):
                    score = kpt['joints'][j,2]
                    if score > self.opt.kpt_threshold:
                        kpt_score += score
                        num_kpts += 1
                if num_kpts > 0:
                    kpt_score /= num_kpts
                # rescoring
                kpt['score'] = box_score * kpt_score
            keep = nms_oks(img_kpts, self.opt.oks_threshold, delta)
            if len(keep) == 0:
                nms_keypoints.append(img_kpts)
            else:
                nms_keypoints.append([img_kpts[i] for i in keep])

        self._write_results(nms_keypoints, res_file)

        info = self._coco_eval(res_file)
        result = OrderedDict(info)
        return result, result['AP']
