from __future__ import absolute_import, division

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *

def max_preds(heatmap, threshold=0):
    n, c, h, w = heatmap.size()
    scores, indexes = torch.max(heatmap.view(n, c, -1), -1)
    x = indexes.fmod(w).float()
    y = indexes.div(w).float()
    coords = torch.stack((x, y), dim=2)
    scores = scores.view(n, c, 1)
    mask = scores.gt(threshold).float()
    coords = coords.mul(mask)
    return coords.cpu().numpy(), scores.cpu().numpy()

def final_preds(heatmap, center, scale, opt):
    coords, scores = max_preds(heatmap)
    n, c, h, w = heatmap.size()
    if opt.adjust_coords:
        for i in range(n):
            for j in range(c):
                hm = heatmap[i,j]
                x = int(coords[i,j,0])
                y = int(coords[i,j,1])
                if 0 < x < w-1 and 0 < y < h-1:
                    diff = np.array([hm[y,x+1]-hm[y,x-1], hm[y+1,x]-hm[y-1,x]])
                    coords[i,j] += np.sign(diff) * 0.25
    coords = transform_preds(coords, center, scale, (h, w))
    return coords, scores

def nms_heatmap(heatmap, threshold = 0, window_size=3):
    dim =  heatmap.dim()    # = 4
    pad = (window_size - 1) // 2
    if torch.is_tensor(heatmap):
        heatmap = Variable(heatmap)
    max_map = F.max_pool2d(heatmap, kernel_size=window_size, stride=1, padding=pad)
    heatmap = heatmap.data
    max_map = max_map.data
    mask = torch.eq(heatmap, max_map)
    if threshold > 0:
        mask = mask.min(heatmap.ge(threshold))
    n, c, h, w = heatmap.size()
    heatmap = heatmap.mul(mask.float()).view(n, c,-1)
    scores, indexes = torch.max(heatmap, -1)
    x = indexes.fmod(w).float()
    y = indexes.div(w).float()
    pred = torch.stack((x,y,scores),dim=2)  # [batch, num_peaks, 3]

    return pred.cpu().numpy()


# anno - joints [(N x) num_joints x 3]  (x, y, is_visible)
# pred - joints [N x num_joints x 3]    (x, y, score)
# ref_scale - [N]
def compute_oks(pred, anno, ref_scale, delta, ground_truth=True, threshold=0):
    """Compute oks array."""
    batch_size = pred.shape[0]
    if anno.ndim < 3:
        anno = np.tile(anno[np.newaxis,:,:], (batch_size,1,1))
    oks = np.zeros((batch_size))
    # for every human keypoint annotation
    for i in range(batch_size):
        pred_joints = pred[i]     # [num_joints, 3]
        anno_joints = anno[i]     # [num_joints, 3]
        scale = ref_scale[i]    # Scalar
        if ground_truth:
            is_count = anno_joints[:, 2] != 2   # [cnt]
        else:
            is_count = np.logical_and(anno_joints[:,2] >= threshold, pred_joints[:,2] >= threshold)
        if is_count.sum() != 0:
            dist = pow((anno_joints[is_count,:2]-pred_joints[is_count,:2]),2).sum(1)   # [cnt]
            d = pow(delta[is_count],2)      # [cnt]
            oks[i] = np.exp(-dist/2/d/(scale+np.spacing(1))).mean()
        # else:
        #     pass
    return oks

def nms_oks(pred, oks_thresh, delta, kpt_thresh=0):
    num_preds = len(pred)
    scores = np.array([pred[i]['score'] for i in range(num_preds)])
    joints = np.array([pred[i]['joints'] for i in range(num_preds)])
    areas = np.array([pred[i]['area'] for i in range(num_preds)])

    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ref_scale = (areas[i] + areas[order[1:]]) / 2
        oks = compute_oks(joints[order[1:]], joints[i], ref_scale,
                          delta, ground_truth=False, threshold=kpt_thresh)
        ind = np.where(oks <= oks_thresh)[0]
        order = order[ind + 1]

    return keep


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

# For single-person
# output - [batch x num_joints x h x w]
def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    n, c, h, w = output.size()
    idx = list(range(c))  # [0, 1, ..., num_joints-1]
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = max_preds(output)  # [batch, num_joints, 2] (x, y)
        target, _ = max_preds(target)
        norm = np.ones((n, 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

# anno - joints [Ngt x num_joints x 3]  (x, y, is_visible)
# ref_scale [Ngt] - 0.6 * norm(head)
# pred - joints [Npred x num_joints x 3]    (x, y, score)
def compute_pck(pred, anno, ref_scale, threshold):
    batch_size = pred.shape[0]  # Ngt
    num_joints = pred.shape[1]
    pck = np.zeros((batch_size))
    if pred_count == 0:
        return np.zeros((anno_count, pred_count)), np.zeros((anno_count, pred_count, num_joints))
    dist = (2 * threshold) * np.ones((anno_count, pred_count, num_joints))
    GT_count = np.zeros(anno_count)         # [Ngt]
    #Compute dist matrix (size gtN*pN*num_joints).
    # for every human keypoint annotation
    for i in range(anno_count):
        anno_joints = anno[i]     # [num_joints, 3]
        is_count = anno_joints[:, 2] < 2   # [cnt]
        GT_count[i] = is_count.sum()
        scale = ref_scale[i]    # Scalar
        if is_count.sum() > 0:
            # # for every predicted human
            # for j in range(pred_count):
            #     pred_joints = pred[j]
            #     dist[i,j,is_count] = anno_joints[is_count,:2].sub(pred_joints[is_count,:2]).pow_(2).sum(1).sqrt_().div_(ref_scale)
            vec = anno_joints[is_count,:2].reshape(1,-1,2) - pred[:,is_count,:2]    # [Np, cnt, 2]
            dist[i,:,is_count] = (np.sqrt(pow(vec,2).sum(2)) / scale).T     # [Np, cnt]

    # Compute PCKh matrix (size Ngt*Np)
    match = dist <= threshold  # [Ngt, Np, num_joints]
    pck = match.sum(2).astype(np.float32) / GT_count.reshape(-1,1)
    return pck, match   # torch.FloatTensor, torch.ByteTensor

# precision [N]
# recall [N]
def compute_ap(precision, recall):
    prec = np.concatenate((np.zeros(1),precision,np.zeros(1)))
    rec = np.concatenate((np.zeros(1),recall,np.ones(1)))
    prec[:-1] = np.fmax(prec[:-1], prec[1:])
    ap = (rec[1:] - rec[:-1]).dot(prec[1:])

    return ap

# anno [num_img]
# ref_scale [num_img]
# pred - joints ([num_img] batch(num_people) x num_joints x 3)
def eval_mAP(pred, anno, ref_scale, delta, dataset='coco'):
    """Evaluate predicted joints and return mAP."""
    oks_all = np.zeros((0))
    oks_num = 0

    # if the image in the predictions, then compute oks
    for i in range(len(pred)):
        if dataset == 'coco':
            oks = compute_oks(pred=pred[i], anno=anno[i], ref_scale=ref_scale[i],
                              delta=delta)

        oks_all = np.concatenate((oks_all, oks), axis=0)
        oks_num += oks.size

    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold) / np.float32(oks_num))
    # mAP = np.mean(average_precision)
    # average_precision.append(mAP)

    return average_precision    # list

# anno [num_img]
# ref_scale [num_img]
# pred - joints ([num_img] num_people x num_joints x 3)
def eval_AP(pred, anno, ref_scale, threshold):
    """Evaluate predicted_file and return AP."""
    num_joints = pred[0].shape[-2]
    GT_all = np.zeros((num_joints))
    score_all = np.zeros((0, num_joints))
    match_all = np.zeros((0, num_joints))
    # if the image in the predictions, then compute pck
    for i in range(len(pred)):
        # GT_count: number of joints in current image
        GT_count = (anno[i][:,:,2] < 2).sum(0) # [num_joints]
        # GT_all: number of joints in all images
        GT_all += GT_count
        pck, match = compute_pck(pred=pred[i], anno=anno[i], ref_scale=ref_scale[i],
                                 threshold=threshold)
        if pck.size == 0:
            continue
        max_ = pck.max(0, keepdims=True)
        pck_ = (pck >= max_) * pck
        max_val = pck_.max(1)
        idx_pred = pck_.argmax(1)
        idx_pred = idx_pred[max_val != 0]  # torch.LongTensor
        idx_gt = np.nonzero(max_val)[0]     # torch.LongTensor
        if idx_pred.shape[0] != idx_gt.shape[0]:
            print("size does not match!!!")
        s = pred[i][idx_pred][:,:,2]    # [matched, num_joints]
        m = np.zeros((idx_pred.shape[0], num_joints)) # [matched, num_joints]
        for k in range(idx_pred.shape[0]):
            m[k] = match[idx_gt[k],idx_pred[k]]

        score_all = np.concatenate((score_all, s), axis=0)
        match_all = np.concatenate((match_all, m), axis=0)    # {0,1}

    sort_score = np.sort(score_all, axis=0)[::-1]
    sort_idx = np.argsort(score_all, axis=0)[::-1]
    sort_match = np.zeros(match_all.shape)
    for i in range(num_joints):
        sort_match[:,i] = match_all[:,i][sort_idx[:,i]]

    sum_match = np.cumsum(sort_match, axis=0)       # [N, num_joints]
    pred_num = np.arange(1,sort_match.shape[0]+1)   # [N]
    precision = sum_match.astype(np.float32) / pred_num.reshape(-1,1)
    recall = sum_match / GT_all.reshape(1,-1)

    precision = np.concatenate((np.zeros((1,num_joints)),precision,np.zeros((1,num_joints))), axis=0)
    recall = np.concatenate((np.zeros((1,num_joints)),recall,np.ones((1,num_joints))), axis=0)
    precision[:-1] = np.fmax(precision[:-1], precision[1:])
    AP = ((recall[1:] - recall[:-1]) * precision[1:]).sum(0)

    # AP = torch.cat((ap, ap.mean()))

    return AP.tolist()   # list
