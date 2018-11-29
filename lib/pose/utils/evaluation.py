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

def final_preds(heatmap, center, scale, adjust_coords=False):
    coords, scores = max_preds(heatmap)
    n, c, h, w = heatmap.size()
    if adjust_coords:
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
            is_count = anno_joints[:, 2] > 0   # [cnt]
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
            if target[n, c, 0] >= 1 and target[n, c, 1] >= 1:
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

# anno - joints [(N x) num_joints x 3]  (x, y, is_visible)
# pred - joints [N x num_joints x 3]    (x, y, score)
# ref_scale - [N] - 0.6 * norm(head)
def compute_pck(pred, anno, ref_scale, threshold):
    pred_joints = pred[:,:,:2] + 1
    anno_joints = anno[:,:,:2]
    is_count = anno[:,:,2] > 0

    dists = np.linalg.norm(pred_joints - anno_joints, axis=2)
    dists = dists / ref_scale

    # Compute PCKh (size N)
    match = (dists <= threshold) * is_count  # [N, num_joints]
    pck = match.sum(0).astype(np.float32) / is_count.sum(0)
    return pck   # [num_joints]

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
# pred - joints ([num_img] batch(num_people) x num_joints x 3)
def eval_AP(pred, anno, ref_scale, threshold):
    """Evaluate predicted_file and return AP."""
    # pred = np.concatenate(pred, axis=0)
    anno = np.concatenate(anno, axis=0)
    ref_scale = np.concatenate(ref_scale, axis=0)
    pck = compute_pck(pred, anno, ref_scale, threshold)

    return pck.tolist()   # list
