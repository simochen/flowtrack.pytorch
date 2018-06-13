from __future__ import absolute_import

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
    return x - mean.reshape((1,1,-1))


def swaplr_joint(joints, width, dataset='mpii'):
    """
    flip coords
    Input: joints - (n x)16 x 3 (x, y, is_visible)
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
        right = [2, 3, 4, 8, 9, 10]
        left =  [5, 6, 7, 11, 12, 13]
        njoints = 15
        nsym = 6
    # AIC  R arms: 0(shoulder), 1(elbow), 2(wrist)
    #      L arms: 3(shoulder), 4(elbow), 5(wrist)
    #      R leg: 6(hip), 7(knee), 8(ankle)
    #      L leg: 9(hip), 10(knee), 11(ankle)
    #      12 - head top,  13 - neck
    elif dataset == 'aic':
        right = [0, 1, 2, 6, 7, 8]
        left  = [3, 4, 5, 9,10,11]
        njoints = 14
        nsym = 6
    # COCO R arm: 6(shoulder), 8(elbow), 10(wrist)
    #      L arm: 5(shoulder), 7(elbow), 9(wrist)
    #      R leg: 12(hip), 14(knee), 16(ankle)
    #      L leg: 11(hip), 13(knee), 15(ankle)
    #       face: 0(nose), 1(l-eye), 2(r-eye), 3(l-ear), 4(r-ear)
    elif dataset == 'coco':
        right = [2, 4, 6, 8,10, 12, 14, 16]
        left  = [1, 3, 5, 7, 9, 11, 13, 15]
        njoints = 17
        nsym = 8
    else:
        print('Not supported dataset: ' + dataset)

    # Flip x coords
    j = joints.view(-1, 3)
    j[:, 0].neg_().add_(width-1)
    # Swap left-right joints
    if joints.dim() > 2 and joints.size(0) > 1:
        n = joints.size(0)
        right = np.tile(right,n) + (np.arange(n)*njoints).repeat(nsym)
        left = np.tile(left,n) + (np.arange(n)*njoints).repeat(nsym)
    tmp = j[right, :].clone()
    j[right, :] = j[left, :]
    j[left, :] = tmp

    return joints


def fliplr_(img, meta):
    """
    flip image
    """
    if img.dim() == 3:
        img = torch.from_numpy(np.flip(img.numpy(), 2).copy())
        width = img.size(2)
        # center person
        meta['objpos_self'][0] = width - 1 - meta['objpos_self'][0]
        meta['joint_self'] = swaplr_joint(meta['joint_self'], width)
        # other people
        for i in range(meta['numOtherPeople']):
            meta['objpos_other'][i][0] = width - 1 - meta['objpos_other'][i][0]
        if meta['numOtherPeople'] > 0:
            meta['joint_others'] = swaplr_joint(meta['joint_others'], width)
    elif img.dim() == 4:
        for i in range(img.size(0)):
            img[i], meta[i] = fliplr_(img[i], meta[i])
    return img, meta


def fliplr(img):
    """
    flip image
    """
    if img.dim() == 3:      # C x H x W
        img = torch.from_numpy(np.flip(img.numpy(), 2).copy())
    elif img.dim() == 4:    # B x C x H x W
        img = torch.from_numpy(np.flip(img.numpy(), 3).copy())
    return img

def swap(img, pairs):
    if img.dim() == 3:
        tmp = img[pairs[0], :].clone()
        img[pairs[0], :] = img[pairs[1], :]
        img[pairs[1], :] = tmp
    elif img.dim() == 4:
        tmp = img[:, pairs[0]].clone()
        img[:, pairs[0]] = img[:, pairs[1]]
        img[:, pairs[1]] = tmp
    return img

def swaplr_image(img, offset_joint, num_limbs=0, offset_limb=0, dataset='mpii'):
    """
    swap images according to channel index
    Input: joints - (n x)16 x 3 (x, y, is_visible)
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
    limb_right = []
    limb_left = []
    if dataset ==  'mpii':
        joint_right = [2, 3, 4, 8, 9, 10]
        joint_left =  [5, 6, 7, 11, 12, 13]
        if num_limbs == 14:
            limb_right = [1, 2, 3, 8, 9, 10]
            limb_left = [4, 5, 6, 11, 12, 13]
        elif num_limbs == 17:
            limb_right = [1, 2, 3, 8, 9, 10, 15]
            limb_left = [4, 5, 6, 11, 12, 13, 16]
        elif num_limbs == 18:
            limb_right = [1, 2, 3, 8, 9, 10, 16]
            limb_left = [4, 5, 6, 11, 12, 13, 17]
    # AIC  R arms: 0(shoulder), 1(elbow), 2(wrist)
    #      L arms: 3(shoulder), 4(elbow), 5(wrist)
    #      R leg: 6(hip), 7(knee), 8(ankle)
    #      L leg: 9(hip), 10(knee), 11(ankle)
    #      12 - head top,  13 - neck
    elif dataset == 'aic':
        joint_right = [0, 1, 2, 6, 7, 8]
        joint_left  = [3, 4, 5, 9,10,11]
        if num_limbs == 14:
            limb_right = [1, 2, 3, 7, 8, 9]
            limb_left = [4, 5, 6, 10, 11, 12]
    # COCO R arm: 6(shoulder), 8(elbow), 10(wrist)
    #      L arm: 5(shoulder), 7(elbow), 9(wrist)
    #      R leg: 12(hip), 14(knee), 16(ankle)
    #      L leg: 11(hip), 13(knee), 15(ankle)
    #       face: 0(nose), 1(l-eye), 2(r-eye), 3(l-ear), 4(r-ear)
    elif dataset == 'coco':
        joint_right = [2, 4, 6, 8,10, 12, 14, 16]
        joint_left  = [1, 3, 5, 7, 9, 11, 13, 15]
        if num_limbs == 17:
            limb_right = [0, 1, 4, 5, 6, 10, 11, 12]
            limb_left =  [2, 3, 7, 8, 9, 13, 14, 15]
        elif num_limbs == 19:
            limb_right = [2, 3, 6, 9,11, 14, 16, 18]
            limb_left =  [0, 1, 5, 8,10, 13, 15, 17]
    else:
        print('Not supported dataset: ' + dataset)

    pairs = np.concatenate((np.array([joint_right, joint_left])+offset_joint,
                            np.array([limb_right, limb_left])+offset_limb)).astype(int)
    img = swap(img, pairs)
    return img

"""
General image processing functions
"""
def get_transform(center, scale, res, factor=0, delta=(0,0), rot=0):
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
def transform_point(pt, center, scale, res, invert=0, factor=0, delta=(0,0), rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, factor=factor, delta=delta, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    # if pt.ndim >= 3:
    #     pt_shape = pt.shape
    #     pt = pt.reshape((-1,2))
    # pt_ = np.concatenate((pt, np.ones((pt.shape[0],1))), axis=1)
    # new_pt = np.dot(t, pt_.T)[:2].T
    # if pt.ndim >= 3:
    #     new_pt = new_pt.reshape(pt_shape)
    # return new_pt #.round().astype(int)
    return do_transfrom(pt, t)


def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = torch.from_numpy(transform(coords[p, 0:2], center, scale, res, 1))
    return coords

# def crop(img, center, scale, res, factor=0, delta=(0,0), rot=0):
#     t = get_transform(center, scale, res, factor=factor, delta=delta, rot=rot)
#     t = np.linalg.inv(t)
#     # Upper left, Upper right, Bottom left, Bottom right
#     points = t.dot(np.array([[0,0,1],[res[1],0,1],[0,res[0],1],[res[1],res[0],1]]).T)[:2].astype(int).T
#     # # Upper left point
#     # ul = t.dot(np.array([[0,0,1]]))[:2].astype(int)
#     # # Upper right point
#     # ur = t.dot(np.array([[res[1],0,1]]))[:2].astype(int)
#     # # Bottom left point
#     # bl = t.dot(np.array([[0,res[0],1]]))[:2].astype(int)
#     # # Bottom right point
#     # br = t.dot(np.array([[res[1],res[0],1]]))[:2].astype(int)
#
#     # bounding rectangle
#     # ul_bound = [min(ul[0],bl[0]), min(ul[1],ur[1])]
#     # br_bound = [max(ur[0],br[0]), max(bl[1],br[1])]
#     ul_bound = [min(points[0][0],points[2][0]), min(points[0][1],points[1][1])]
#     br_bound = [max(points[1][0],points[3][0]), max(points[2][1],points[3][1])]
#
#     # relative ul and br
#     # ul_r = [ul[0]-ul_bound[0], ul[1]-ul_bound[1]]
#     # br_r = [br[0]-ul_bound[0], br[1]-ul_bound[1]]
#     ul_r = [points[0][0]-ul_bound[0], points[0][1]-ul_bound[1]]
#     br_r = [points[3][0]-ul_bound[0], points[3][1]-ul_bound[1]]
#
#     crop_shape = [br_bound[1] - ul_bound[1], br_bound[0] - ul_bound[0]]
#     if img.dim() > 2:
#         crop_shape += [img.size(0)]
#     crop_img = np.zeros(crop_shape)
#
#     # Range to fill new array
#     crop_x = [max(0, -ul_bound[0]), min(br_bound[0], img.size(-1)) - ul_bound[0]]
#     crop_y = [max(0, -ul_bound[1]), min(br_bound[1], img.size(-2)) - ul_bound[1]]
#     # Range to sample from original image
#     img_x = [max(0, ul_bound[0]), min(img.size(-1), br_bound[0])]
#     img_y = [max(0, ul_bound[1]), min(img.size(-2), br_bound[1])]
#
#     img = img.numpy()
#     if img.ndim > 2:
#         img = img.transpose((1,2,0))
#     crop_img[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]] = img[img_y[0]:img_y[1], img_x[0]:img_x[1]]
#
#     if rot != 0:
#         # Remove padding
#         crop_img = scipy.misc.imrotate(crop_img, rot, interp='bicubic')
#         crop_img = crop_img[ul_r[1]:br_r[1], ul_r[0]:ul_r[1]]
#
#     crop_img = scipy.misc.imresize(crop_img, res, interp='bicubic')
#     if img.ndim > 2:
#         crop_img = crop_img.transpose((2,0,1))
#
#     # scipy.misc.imrotate, scipy.misc.imresize return numpy with uint8 dtype
#     return torch.from_numpy(crop_img / 255.0)

# img[INPUT] :    tensor [C x H x W]
#              or numpy  [H x W x C]
# img[OUTPUT]:    tensor [C x H x W]
def transform_image(img, center, scale, res, factor=0, delta=(0,0), rot=0): # pad_value=(0,0,0)
    if torch.is_tensor(img):
        img = img.numpy()
        if img.ndim > 2:
            img = img.transpose((1,2,0))
    t = get_transform(center, scale, res, factor=factor, delta=delta, rot=rot)
    trans_img = cv2.warpAffine(img, t[:2], (res[1],res[0])) # borderValue=pad_value
    if img.ndim > 2:
        trans_img = trans_img.transpose((2,0,1))
    return torch.from_numpy(trans_img)
