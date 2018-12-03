from __future__ import absolute_import, division

import os
import math
import numpy as np
import torch
import cv2
# import scipy.misc

def compute_head_norm(rect):
    # rect [N,4] (x1, y1, x2, y2)
    return rect[:,2].sub(rect[:,0]).pow_(2).add(rect[:,3].sub(rect[:,1]).pow_(2)).sqrt_().mul_(0.6)

def compute_bbox_area(bbox):
    # bbox [N,4] (x1, y1, x2, y2)
    return bbox[:,2].sub(bbox[:,0]).mul(bbox[:,3].sub(bbox[:,1]))

# def compute_bbox_area_coco(bbox):
#     # bbox [N,4] (x, y, w, h)
#     return bbox[:,2].mul(bbox[:,3])

def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, x.size(1), x.size(2))
    return x.sub(mean.view(3, 1, 1).expand_as(x)).div(std.view(3, 1, 1).expand_as(x))

def mean_sub(x, mean):
    if isinstance(mean, list):
        if isinstance(x, np.ndarray):
            mean = np.array(mean)
        else:
            mean = torch.Tensor(mean)
    if isinstance(x, np.ndarray):
        y = x - mean.reshape((1,1,-1))
        y = y.astype(np.float32)
    else:
        y = x - mean.view((-1,1,1))
    return y

def normalize(x, mean, std):
    if isinstance(mean, list):
        if isinstance(x, np.ndarray):
            mean = np.array(mean)
            std = np.array(std)
        else:
            mean = torch.Tensor(mean)
            std = torch.Tensor(std)
    if isinstance(x, np.ndarray):
        y = (x - mean.reshape((1,1,-1))) / std.reshape((1,1,-1))
        y = y.astype(np.float32)
    else:
        y = (x - mean.view((-1,1,1))) / std.view((-1,1,1))
    return y

def get_img(img, mean, std):
    return img.numpy().transpose((1,2,0)) * std.reshape((1,1,-1)) + mean.reshape((1,1,-1))

NUM_JOINTS = {'mpii':16, 'aic':14, 'coco':17}

def get_pairs(dataset='coco'):
    """
    get matched joint pairs
    """
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
    if dataset ==  'mpii':
        joint_right = [0, 1, 2, 10, 11, 12]
        joint_left =  [5, 4, 3, 15, 14, 13]
    # AIC  R arms: 0(shoulder), 1(elbow), 2(wrist)
    #      L arms: 3(shoulder), 4(elbow), 5(wrist)
    #      R leg: 6(hip), 7(knee), 8(ankle)
    #      L leg: 9(hip), 10(knee), 11(ankle)
    #      12 - head top,  13 - neck
    elif dataset == 'aic':
        joint_right = [0, 1, 2, 6, 7, 8]
        joint_left  = [3, 4, 5, 9,10,11]
    # COCO R arm: 6(shoulder), 8(elbow), 10(wrist)
    #      L arm: 5(shoulder), 7(elbow), 9(wrist)
    #      R leg: 12(hip), 14(knee), 16(ankle)
    #      L leg: 11(hip), 13(knee), 15(ankle)
    #       face: 0(nose), 1(l-eye), 2(r-eye), 3(l-ear), 4(r-ear)
    elif dataset == 'coco':
        joint_right = [2, 4, 6, 8,10, 12, 14, 16]
        joint_left  = [1, 3, 5, 7, 9, 11, 13, 15]
    else:
        print('Not supported dataset: ' + dataset)

    return joint_right, joint_left

def swaplr_joint(joints, width, dataset='coco'):
    """
    flip coords
    Input: joints - (n x)16 x 3 (x, y, is_visible)
    """

    right, left = get_pairs(dataset)
    njoints = NUM_JOINTS[dataset]
    nsym = len(right)

    # Flip x coords
    j = joints.reshape((-1, 3))
    j[:, 0] = width - 1 - j[:, 0]
    # Swap left-right joints
    if joints.ndim > 2 and joints.shape[0] > 1:
        n = joints.shape[0]
        right = np.tile(right,n) + (np.arange(n)*njoints).repeat(nsym)
        left = np.tile(left,n) + (np.arange(n)*njoints).repeat(nsym)
    tmp = j[right, :].copy()
    j[right, :] = j[left, :]
    j[left, :] = tmp

    return joints


def fliplr(img):
    if img.ndim == 3:   # H x W x C
        img = np.flip(img, 1).copy()
    elif img.ndim == 4: # B x H x W x C
        img = np.flip(img, 2).copy()
    return img

def fliplr_tensor(img):
    """
    flip image: Tensor
    """
    if img.dim() == 3:      # C x H x W
        img = torch.from_numpy(np.flip(img.numpy(), 2).copy())
    elif img.dim() == 4:    # B x C x H x W
        img = torch.from_numpy(np.flip(img.numpy(), 3).copy())
    return img

def swap(img, pairs):
    """
    swap images according to channel index
    Input: img - Tensor[(batch_size,) num_joints, height, width]
    """
    if img.ndim == 3:
        img = img[:, :, ::-1]
        tmp = img[pairs[0], :].copy()
        img[pairs[0], :] = img[pairs[1], :]
        img[pairs[1], :] = tmp
    elif img.ndim == 4:
        img = img[:, :, :, ::-1]
        tmp = img[:, pairs[0]].copy()
        img[:, pairs[0]] = img[:, pairs[1]]
        img[:, pairs[1]] = tmp
    return img

def swaplr_image(img, dataset='coco'):
    """
    swap images according to channel index
    Input: img - Tensor[(batch_size,) num_joints, height, width]
    """
    joint_right, joint_left = get_pairs(dataset)

    # pairs = np.array([joint_right, joint_left]).astype(int)
    # img = swap(img, pairs)
    img = swap(img, [joint_right, joint_left])

    return img

"""
General image processing functions
"""
def get_transform(center, scale, res, factor=1, rot=0, delta=(0,0)):
    # Generate transformation matrix
    t = np.eye(3)
    if factor == 0:
        t[0, 2] = -center[0] + 0.5 * res[1] + delta[0]
        t[1, 2] = -center[1] + 0.5 * res[0] + delta[1]
    else:
        res_ = res[0] * factor  # H * factor
        t[0, 0] = res_ / scale
        t[1, 1] = res_ / scale
        t[0, 2] = -res_ * center[0] / scale + 0.5 * res[1] + delta[0]
        t[1, 2] = -res_ * center[1] / scale + 0.5 * res[0] + delta[1]
    if rot != 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        s, c = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [c, -s]
        rot_mat[1,:2] = [s, c]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1] * 0.5 - delta[0]
        t_mat[1,2] = -res[0] * 0.5 - delta[1]
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def do_transfrom(pt, t):
    pt_shape = pt.shape
    if len(pt_shape) >= 3:
        pt = pt.reshape((-1,2))
    pt_ = np.concatenate((pt, np.ones((pt.shape[0],1))), axis=1)
    new_pt = np.dot(t, pt_.T)[:2].T
    if len(pt_shape) >= 3:
        new_pt = new_pt.reshape(pt_shape)
    return new_pt #.round().astype(int)

# pt: ([M *] N, 2) np.array
def transform_point(pt, center, scale, res, invert=0, factor=1, rot=0, delta=(0,0)):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, factor=factor, rot=rot, delta=delta)
    if invert:
        t = np.linalg.inv(t)
    return do_transfrom(pt, t)


def transform_preds(coords, center, scale, res):
    # coords: [batch_size, num_joints, 2] (x, y)
    # res: (h, w)
    for i in range(coords.shape[0]):
        coords[i] = transform_point(coords[i], center[i], scale[i], res, invert=1)
    return coords

# img[INPUT] :    tensor [C x H x W]
#              or numpy  [H x W x C]
# img[OUTPUT]:    tensor [C x H x W]
def transform_image(img, center, scale, res, factor=1, rot=0, delta=(0,0)): # pad_value=(0,0,0)
    if torch.is_tensor(img):
        img = img.numpy()
        if img.ndim > 2:
            img = img.transpose((1,2,0))
    t = get_transform(center, scale, res, factor=factor, rot=rot, delta=delta)
    trans_img = cv2.warpAffine(img, t[:2], (res[1],res[0])) # borderValue=pad_value
    if img.ndim > 2:
        trans_img = trans_img.transpose((2,0,1))
    return torch.from_numpy(trans_img)
