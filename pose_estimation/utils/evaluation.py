from __future__ import absolute_import, division

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from utils import *

def nms_single(heatmap, threshold = 0, window_size=3):
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


def nms(heatmap, threshold = 0, window_size = 3):
    dim =  heatmap.dim() # <= 3
    pad = (window_size - 1) // 2
    if torch.is_tensor(heatmap):
        heatmap = Variable(heatmap)
    max_map = F.max_pool2d(heatmap, kernel_size=window_size, stride=1, padding=pad)
    heatmap = heatmap.data
    max_map = max_map.data
    mask = torch.eq(heatmap, max_map)
    if threshold > 0:
        mask = mask.min(heatmap.ge(threshold))
    # nms_map = mask.float().mul(heatmap)
    coords = torch.nonzero(mask)                # [num_peaks, 3]
    if len(coords.size()) > 0:
        coords[:, -2:] = coords[:, [dim-1,dim-2]]   # [c, y, x] => [c, x, y]
    scores = heatmap.masked_select(mask)        # [num_peaks]

    return coords, scores    #torch(.cuda).LongTensor, torch(.cuda).FloatTensor

def nms_sep(heatmap, threshold, window_size):
    if torch.is_tensor(heatmap):
        heatmap = Variable(heatmap)
    coords = torch.zeros((0, 3))
    scores = torch.zeros((0))
    for i in range(heatmap.size(0)):
        pad = (window_size[i] - 1) // 2
        hm = heatmap[i:i+1]
        max_map = F.max_pool2d(hm, kernel_size=window_size[i], stride=1, padding=pad)
        hm = hm.squeeze(0).data
        max_map = max_map.squeeze(0).data
        mask = torch.eq(hm, max_map)
        if threshold[i] > 0:
            mask = mask.min(hm.ge(threshold[i]))
        # nms_map = mask.float().mul(heatmap)
        coord = torch.nonzero(mask).float()           # [num_peaks, 3]
        if len(coord.size()) > 0:
            coord = coord[:, [1,0]]   # [c, y, x] => [c, x, y]
            a = torch.ones((coord.size(0),1)).cuda().mul_(i)
            coord = torch.cat((a, coord), dim=1)
            score = hm.masked_select(mask)        # [num_peaks]
            if len(coords.size()) == 0:
                coords = coord
                scores = score
            else:
                coords = torch.cat((coords, coord))
                scores = torch.cat((scores, score))

    return coords, scores

def gt_detection(joints):
    # joints: num_people, num_joints, 3 (x, y, is_visible)
    coords = []
    scores = []
    for j in range(joints.shape[1]):
        for i in range(joints.shape[0]):
            if joints[i,j,2] < 2:
                coords.append([j,joints[i,j,0],joints[i,j,1]])
                scores.append(1)
    return np.array(coords), np.array(scores)

def gt_connection(joints, limbs, output_res):
    # joints: num_people, num_joints, 3 (x, y, is_visible)
    sigma = 30
    limb_hm = torch.zeros(len(limbs), output_res[0], output_res[1])
    for i in range(joints.shape[0]):
        for j in range(len(limbs)):
            pt1 = joints[i,limbs[j][0]]
            pt2 = joints[i,limbs[j][1]]
            if pt1[2] < 2 and pt2[2] < 2:
                limb_hm[j] = draw_line_gaussian(limb_hm[j], [[pt1[0],pt1[1]],[pt2[0],pt2[1]]], sigma)
    return limb_hm.numpy()

# opt :
#   nms_threshold,
#   nms_window_size,
#   inter_min - minimum interpolates
#   interval  - sampling interval
#   inter_threshold - threshold of interval
#   joints_min - minimum joints number
#   person_threshold - threshold of scores / joints number
# run on cpu (for now)
def piece_people(joint_hm, limb_hm, opt, joints=None, scale=0):
    # hm.dim() == 3
    num_joints = joint_hm.size(0)
    if opt.with_gt == 1 and joints is not None:
        coords, scores = gt_detection(joints)
    else:
        coords, scores = nms(joint_hm, opt.nms_threshold, opt.nms_window_size)
        if len(coords.size()) == 0:
            return np.zeros((0, num_joints, 3))
        coords, scores = coords.cpu().numpy(), scores.cpu().numpy() # copy to cpu
    limb_hm = limb_hm.cpu().numpy()
    height = limb_hm.shape[1]
    width = limb_hm.shape[2]
    idxs = np.arange(0, coords.shape[0])

    # group coords, scores, idxs
    peaks = np.concatenate((coords[:,1:], scores.reshape(-1,1), idxs.reshape(-1,1)), axis=1)
    peak_counter = 0
    peaks_group = []
    for j in range(num_joints):
        num_peaks = (coords[:,0] == j).sum()
        peaks_this = peaks[peak_counter:peak_counter+num_peaks]
        if opt.norm_threshold > 0:
            indices = np.argsort(peaks_this[:,2])[-1::-1]
            peaks_this = peaks_this[indices]
            reserve = np.ones(peaks_this.shape[0]).astype(bool)
            for i in range(peaks_this.shape[0]-1):
                if reserve[i] > 0:
                    for k in range(i+1, peaks_this.shape[0]):
                        if np.linalg.norm(peaks_this[i,:2] - peaks_this[k,:2]) < opt.norm_threshold:
                            reserve[k] = 0
            peaks_this = peaks_this[reserve]

        peaks_group.append(peaks_this)
        peak_counter += num_peaks

    # MPII L arms: 13(shoulder), 14(elbow), 15(wrist)
    #      R arms: 12(shoulder), 11(elbow), 10(wrist)
    #      L leg: 3(hip), 4(knee), 5(ankle)
    #      R leg: 2(hip), 1(knee), 0(ankle)
    #      6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top
    # MPII R arms: 2(shoulder), 3(elbow), 4(wrist)
    #      L arms: 5(shoulder), 6(elbow), 7(wrist)
    #      R leg: 8(hip), 9(knee), 10(ankle)
    #      L leg: 11(hip), 12(knee), 13(ankle)
    #      14 - pelvis-thorax, 1 - upper neck, 0 - head top
    num_limbs = limb_hm.shape[0]
    if opt.dataset == 'mpii':
        if num_limbs == 14:
            limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                    [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]
        elif num_limbs == 17:
            limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                    [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13],
                    [8,11],[2,8],[5,11]]
        elif num_limbs == 18:
            limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                    [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13],
                    [2,5],[8,11],[2,8],[5,11]]

    # AIC  R arms: 0(shoulder), 1(elbow), 2(wrist)
    #      L arms: 3(shoulder), 4(elbow), 5(wrist)
    #      R leg: 6(hip), 7(knee), 8(ankle)
    #      L leg: 9(hip), 10(knee), 11(ankle)
    #      12 - head top,  13 - neck
    elif opt.dataset == 'aic':
        if num_limbs == 14:
            limbs = [[12,13],
                    [13,0],[0,1],[1,2],[13,3],[3,4],[4,5],
                    [0,6],[6,7],[7,8],[3,9],[9,10],[10,11],
                    [6,9]]

    # COCO R arm: 6(shoulder), 8(elbow), 10(wrist)
    #      L arm: 5(shoulder), 7(elbow), 9(wrist)
    #      R leg: 12(hip), 14(knee), 16(ankle)
    #      L leg: 11(hip), 13(knee), 15(ankle)
    #       face: 0(nose), 1(l-eye), 2(r-eye), 3(l-ear), 4(r-ear)
    elif opt.dataset == 'coco':
        if num_limbs == 17:
            limbs = [[0,2],[2,4],[0,1],[1,3],
                    [0,6],[6,8],[8,10],[0,5],[5,7],[7,9],
                    [6,12],[12,14],[14,16],[5,11],[11,13],[13,15],
                    [12,11]]
        elif num_limbs == 19:
            limbs = [[15, 13],[13, 11],[16, 14],[14, 12],[11, 12],
                    [5, 11],[6, 12],[5, 6],
                    [5, 7],[6, 8],[7, 9],[8, 10],
                    [1, 2],[0, 1],[0, 2],[1, 3],[2, 4],[3, 5],[4, 6]]

    if opt.with_gt == 2 and joints is not None:
        limb_hm = gt_connection(joints, limbs, limb_hm.shape[1:])
    # compute line intergral and choose connection
    connection_group = []
    for k in range(num_limbs):
        l_map = limb_hm[k]
        peakA = peaks_group[limbs[k][0]]
        peakB = peaks_group[limbs[k][1]]
        nA = peakA.shape[0]
        nB = peakB.shape[0]

        candidates = []     # list
        for i in range(nA):
            for j in range(nB):
                norm = np.sqrt(pow((peakB[j,:2] - peakA[i,:2]),2).sum())
                steps = max(opt.inter_min, int(norm/opt.interval))
                # steps = min(opt.inter_max, steps)
                inter_pts = zip(np.linspace(peakA[i,0], peakB[j,0], num=steps).round(),
                                np.linspace(peakA[i,1], peakB[j,1], num=steps).round())
                # l_map[y, x]
                score_pts = np.array([l_map[int(inter_pts[c][1]),int(inter_pts[c][0])] for c in range(steps)
                                        if int(inter_pts[c][1]) < height and int(inter_pts[c][0]) < width])
                line_score = score_pts.sum() / steps
                if scale > 0:
                    line_score += min(0.5*scale/max(norm,1e-3)-1, 0)
                if (score_pts > opt.inter_threshold).sum() > 0.8*steps:
                    if (scale > 0 and line_score > 0) or scale <= 0:
                        candidates.append([i,j,line_score])

        connection = np.zeros((0,3))
        if len(candidates) == 0:
            connection_group.append(connection)
            continue
        candidates.sort(key=lambda x:x[2], reverse=True)
        occurA = np.zeros(nA)
        occurB = np.zeros(nB)
        for c in range(len(candidates)):
            i,j,s = candidates[c]
            if occurA[i] == 0 and occurB[j] == 0:
                connection = np.vstack([connection, [peakA[i,3], peakB[j,3], s]])
                occurA[i] = 1
                occurB[j] = 1
                if connection.shape[0] >= min(nA, nB):
                    break
        connection_group.append(connection)

    # identify limbs belong to same person
    person = -1 * np.ones((0, num_joints+2))
    for k in range(num_limbs):
        connection = connection_group[k]
        if connection.shape[0] == 0:
            continue
        jointA = connection[:,0].astype(int)   # idxs
        jointB = connection[:,1].astype(int)
        indexA, indexB = limbs[k]
        for i in range(connection.shape[0]):
            occur = 0
            pA = -1
            pB = -1
            for j in range(person.shape[0]):
                if person[j][indexA] == jointA[i]:
                    pA = j
                    occur += 1
                elif person[j][indexB] == jointB[i]:
                    pB = j
                    occur += 1
                if occur == 2:
                    break

            if occur == 0:
                p_one = -1 * np.ones(num_joints+2)
                p_one[indexA] = jointA[i]
                p_one[indexB] = jointB[i]
                p_one[-1] = 2
                p_one[-2] = scores[connection[i,:2].astype(int)].sum() + connection[i,2]
                person = np.vstack([person, p_one])
            elif occur == 1:
                if pA >= 0:
                    if person[pA][indexB] != jointB[i]:
                        person[pA][indexB] = jointB[i]
                        person[pA][-1] += 1
                        person[pA][-2] += (scores[jointB[i]] + connection[i,2])
                else:
                    if person[pB][indexA] != jointA[i]:
                        person[pB][indexA] = jointA[i]
                        person[pB][-1] += 1
                        person[pB][-2] += (scores[jointA[i]] + connection[i,2])
            elif occur == 2:
                joint_fill = (person[pA,:-2]>=0).astype(int) + (person[pB,:-2]>=0).astype(int)
                if np.nonzero(joint_fill==2)[0].shape[0] == 0:
                    person[pA][:-2] += (person[pB][:-2] + 1)
                    person[pA][-2:] += person[pB][-2:]
                    person[pA][-2] += connection[i,2]
                    person = np.delete(person, pB, axis=0)
                else:
                    if pB >= 0 and person[pB][indexA] == -1:
                        person[pB][indexA] = jointA[i]
                        person[pB][-1] += 1
                        person[pB][-2] += (scores[jointA[i]] + connection[i,2])
                    if pA >= 0 and person[pA][indexB] == -1:
                        person[pA][indexB] = jointB[i]
                        person[pA][-1] += 1
                        person[pA][-2] += (scores[jointB[i]] + connection[i,2])
                    # else:
                        # print "warning: have both joint !!!!!"
                        #print person
                        #print indexA, indexB
                        #print pA, pB
                        #print jointA, jointB
                        #print i

            # occur = 0
            # person_idx = [-1, -1]
            # for j in range(person.shape[0]):
            #     if person[j][indexA] == jointA[i] or person[j][indexB] == jointB[i]:
            #         person_idx[occur] = j
            #         occur += 1
            #         if occur == 2:
            #             break
            # if occur == 0:
            #     p_one = -1 * np.ones(num_joints+2)
            #     p_one[indexA] = jointA[i]
            #     p_one[indexB] = jointB[i]
            #     p_one[-1] = 2
            #     p_one[-2] = scores[connection[i,:2]].sum() + connection[i,2]
            #     person = np.vstack([person, p_one])
            # elif occur == 1:
            #     j = person_idx[0]
            #     if person[j][indexB] != jointB[i]:
            #         person[j][indexB] = jointB[i]
            #         person[j][-1] += 1
            #         person[j][-2] += (scores[jointB[i]] + connection[i,2])
            # elif occur == 2:
            #     j1, j2 = person_idx
            #     joint_fill = (person[j1,:-2]>=0).astype(int) + (person[j2,:-2]>=0).astype(int)
            #     if np.nonzero(joint_fill==2)[0].shape[0] == 0:
            #         person[j1][:-2] += (person[j2][:-2] + 1)
            #         person[j1][-2:] += person[j2][-2:]
            #         person[j1][-2] += connection[i,2]
            #         person = np.delete(person, j2, axis=0)
            #     else: # as like found == 1 (???)
            #         person[j][indexB] = jointB[i]
            #         person[j][-1] += 1
            #         person[j][-2] += (scores[jointB[i]] + connection[i,2])

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(person.shape[0]):
        if person[i,-1] < opt.joints_min or person[i,-2]/person[i,-1] < opt.person_threshold:
            deleteIdx.append(i)
    person = np.delete(person, deleteIdx, axis=0)   # [num_person, num_joints+2]
    # ==> [num_person, num_joints, 3] (x, y, score)
    pred = np.zeros((person.shape[0], num_joints, 3))
    for i in range(person.shape[0]):
        found = person[i,:-2] >= 0
        pred[i,found] = peaks[person[i,:-2][found].astype(int),:3]

    return pred


def piece_pred(joint_hm, limb_hm, opt):
    # hm.dim() == 3
    joint_hm = joint_hm[:-1]  # if include background channel
    num_joints = joint_hm.size(0)
    coords, scores = nms(joint_hm, opt.nms_threshold, opt.nms_window_size)
    if len(coords.size()) == 0:
        return np.zeros((0, num_joints, 3))
    coords, scores = coords.cpu().numpy(), scores.cpu().numpy() # copy to cpu
    limb_hm = limb_hm.cpu().numpy()
    idxs = np.arange(0, coords.shape[0])

    # group coords, scores, idxs
    peaks = np.concatenate((coords[:,1:], scores.reshape(-1,1), idxs.reshape(-1,1)), axis=1)
    peak_counter = 0
    peaks_group = []
    for j in range(num_joints):
        num_peaks = (coords[:,0] == j).sum()
        peaks_this = peaks[peak_counter:peak_counter+num_peaks]
        peaks_group.append(peaks_this)
        peak_counter += num_peaks

    # MPII L arms: 13(shoulder), 14(elbow), 15(wrist)
    #      R arms: 12(shoulder), 11(elbow), 10(wrist)
    #      L leg: 3(hip), 4(knee), 5(ankle)
    #      R leg: 2(hip), 1(knee), 0(ankle)
    #      6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top
    num_limbs = limb_hm.shape[0]
    if opt.dataset == 'mpii':
        if num_limbs == 14:
            limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                    [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]
        elif num_limbs == 17:
            limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                    [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13],
                    [8,11],[2,8],[5,11]]
        elif num_limbs == 18:
            limbs = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],
                    [1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13],
                    [2,5],[8,11],[2,8],[5,11]]

    # AIC  R arms: 0(shoulder), 1(elbow), 2(wrist)
    #      L arms: 3(shoulder), 4(elbow), 5(wrist)
    #      R leg: 6(hip), 7(knee), 8(ankle)
    #      L leg: 9(hip), 10(knee), 11(ankle)
    #      12 - head top,  13 - neck
    elif opt.dataset == 'aic':
        if num_limbs == 14:
            limbs = [[12,13],
                    [13,0],[0,1],[1,2],[13,3],[3,4],[4,5],
                    [0,6],[6,7],[7,8],[3,9],[9,10],[10,11],
                    [6,9]]

    # COCO R arm: 6(shoulder), 8(elbow), 10(wrist)
    #      L arm: 5(shoulder), 7(elbow), 9(wrist)
    #      R leg: 12(hip), 14(knee), 16(ankle)
    #      L leg: 11(hip), 13(knee), 15(ankle)
    #       face: 0(nose), 1(l-eye), 2(r-eye), 3(l-ear), 4(r-ear)
    elif opt.dataset == 'coco':
        if num_limbs == 17:
            limbs = [[0,2],[2,4],[0,1],[1,3],
                    [0,6],[6,8],[8,10],[0,5],[5,7],[7,9],
                    [6,12],[12,14],[14,16],[5,11],[11,13],[13,15],
                    [12,11]]
        elif num_limbs == 19:
            limbs = [[15, 13],[13, 11],[16, 14],[14, 12],[11, 12],
                    [5, 11],[6, 12],[5, 6],
                    [5, 7],[6, 8],[7, 9],[8, 10],
                    [1, 2],[0, 1],[0, 2],[1, 3],[2, 4],[3, 5],[4, 6]]

    # compute line intergral and choose connection
    connection_group = []
    for k in range(num_limbs):
        l_map = limb_hm[k]
        peakA = peaks_group[limbs[k][0]]
        peakB = peaks_group[limbs[k][1]]
        nA = peakA.shape[0]
        nB = peakB.shape[0]

        candidates = []     # list
        for i in range(nA):
            for j in range(nB):
                norm = np.sqrt(pow((peakB[j,:2] - peakA[i,:2]),2).sum())
                steps = max(opt.inter_min, int(norm/opt.interval))
                inter_pts = zip(np.linspace(peakA[i,0], peakB[j,0], num=steps).round(),
                                np.linspace(peakA[i,1], peakB[j,1], num=steps).round())
                # l_map[y, x]
                score_pts = np.array([l_map[int(inter_pts[c][1]),int(inter_pts[c][0])] for c in range(steps)])
                line_score = score_pts.sum() / steps
                if (score_pts > opt.inter_threshold).sum() >= 0.8*steps:
                    candidates.append([i,j,line_score])

        connection = np.zeros((0,3))
        if len(candidates) == 0:
            connection_group.append(connection)
            continue
        candidates.sort(key=lambda x:x[2], reverse=True)
        occurA = np.zeros(nA)
        occurB = np.zeros(nB)
        for c in range(len(candidates)):
            i,j,s = candidates[c]
            if occurA[i] == 0 and occurB[j] == 0:
                connection = np.vstack([connection, [peakA[i,3], peakB[j,3], s]])
                occurA[i] = 1
                occurB[j] = 1
                if connection.shape[0] >= min(nA, nB):
                    break
        connection_group.append(connection)

    # identify limbs belong to same person
    person = -1 * np.ones((0, num_joints+2))
    for k in range(num_limbs):
        connection = connection_group[k]
        if connection.shape[0] == 0:
            continue
        jointA = connection[:,0].astype(int)   # idxs
        jointB = connection[:,1].astype(int)
        indexA, indexB = limbs[k]
        for i in range(connection.shape[0]):
            occur = 0
            pA = -1
            pB = -1
            for j in range(person.shape[0]):
                if person[j][indexA] == jointA[i]:
                    pA = j
                    occur += 1
                elif person[j][indexB] == jointB[i]:
                    pB = j
                    occur += 1
                if occur == 2:
                    break

            if occur == 0:
                p_one = -1 * np.ones(num_joints+2)
                p_one[indexA] = jointA[i]
                p_one[indexB] = jointB[i]
                p_one[-1] = 2
                p_one[-2] = scores[connection[i,:2].astype(int)].sum() + connection[i,2]
                person = np.vstack([person, p_one])
            elif occur == 1:
                if pA >= 0:
                    if person[pA][indexB] != jointB[i]:
                        person[pA][indexB] = jointB[i]
                        person[pA][-1] += 1
                        person[pA][-2] += (scores[jointB[i]] + connection[i,2])
                else:
                    if person[pB][indexA] != jointA[i]:
                        person[pB][indexA] = jointA[i]
                        person[pB][-1] += 1
                        person[pB][-2] += (scores[jointA[i]] + connection[i,2])
            elif occur == 2:
                joint_fill = (person[pA,:-2]>=0).astype(int) + (person[pB,:-2]>=0).astype(int)
                if np.nonzero(joint_fill==2)[0].shape[0] == 0:
                    person[pA][:-2] += (person[pB][:-2] + 1)
                    person[pA][-2:] += person[pB][-2:]
                    person[pA][-2] += connection[i,2]
                    person = np.delete(person, pB, axis=0)
                else:
                    if pB >= 0 and person[pB][indexA] == -1:
                        person[pB][indexA] = jointA[i]
                        person[pB][-1] += 1
                        person[pB][-2] += (scores[jointA[i]] + connection[i,2])
                    if pA >= 0 and person[pA][indexB] == -1:
                        person[pA][indexB] = jointB[i]
                        person[pA][-1] += 1
                        person[pA][-2] += (scores[jointB[i]] + connection[i,2])
                    else:
                        print "warning: have both joint !!!!!"
                        #print person
                        #print indexA, indexB
                        #print pA, pB
                        #print jointA, jointB
                        #print i

    # delete some rows of subset which has few parts occur
    deleteIdx = []
    for i in range(person.shape[0]):
        if person[i,-1] < opt.joints_min or person[i,-2]/person[i,-1] < opt.person_threshold:
            deleteIdx.append(i)
    person = np.delete(person, deleteIdx, axis=0)   # [num_person, num_joints+2]
    # ==> [num_person, num_joints, 3] (x, y, is_visible)
    pred = np.zeros((person.shape[0], num_joints, 3))
    for i in range(person.shape[0]):
        found = person[i,:-2] >= 0
        pred[i,found,:2] = peaks[person[i,:-2][found].astype(int),:2]
        pred[i,found,2] = 1

    return pred.astype(int)

# x
# anno - joints [Ngt x num_joints x 3]  (x, y, is_visible)
#      - ref_scale [Ngt] - bbox area
# pred - joints [Npred x num_joints x 3]    (x, y, score)
def compute_oks_aic(pred, anno, delta):
    """Compute oks array."""
    batch_size = anno['joints'].shape[0]
    oks = np.zeros((batch_size))
    # for every human keypoint annotation
    for i in range(batch_size):
        anno_joints = anno['joints'][i]     # [num_joints, 3]
        pred_joints = pred[i]               # [num_joints, 3]
        is_count = anno_joints[:, 2] == 1   # [cnt]
        ref_scale = anno['ref_scale'][i]    # Scalar
        if is_count.sum() != 0:
            dist = pow((anno_joints[is_count,:2]-pred_joints[is_count,:2]),2).sum(1)   # [cnt]
            d = pow(delta[is_count],2)      # [cnt]
            oks[i] = np.exp(-dist/2/d/(ref_scale+np.spacing(1))).mean()
    return oks

# x
# anno {'joints', 'ref_scale'}
# pred - joints (batch(num_people) x num_joints x 3)
def compute_oks(pred, anno, delta):
    """Compute oks array."""
    batch_size = anno['joints'].shape[0]
    oks = np.zeros((batch_size))
    # for every human keypoint annotation
    for i in range(batch_size):
        anno_joints = anno['joints'][i]     # [num_joints, 3]
        pred_joints = pred[i]               # [num_joints, 3]
        is_count = anno_joints[:, 2] != 2   # [cnt]
        ref_scale = anno['ref_scale'][i]    # Scalar
        if is_count.sum() != 0:
            dist = pow((anno_joints[is_count,:2]-pred_joints[is_count,:2]),2).sum(1)   # [cnt]
            d = pow(delta[is_count],2)      # [cnt]
            oks[i] = np.exp(-dist/2/d/(ref_scale+np.spacing(1))).mean()
        # else:
        #     pass
    return oks

# anno - joints [Ngt x num_joints x 3]  (x, y, is_visible)
#      - ref_scale [Ngt] - 0.6 * norm(head)
# pred - joints [Npred x num_joints x 3]    (x, y, score)
def compute_pck(pred, anno, threshold):
    batch_size = anno['joints'].shape[0]  # Ngt
    num_joints = pred.shape[1]
    pck = np.zeros((batch_size))
    if pred_count == 0:
        return np.zeros((anno_count, pred_count)), np.zeros((anno_count, pred_count, num_joints))
    dist = (2 * threshold) * np.ones((anno_count, pred_count, num_joints))
    GT_count = np.zeros(anno_count)         # [Ngt]
    #Compute dist matrix (size gtN*pN*num_joints).
    # for every human keypoint annotation
    for i in range(anno_count):
        anno_joints = anno['joints'][i]     # [num_joints, 3]
        is_count = anno_joints[:, 2] < 2   # [cnt]
        GT_count[i] = is_count.sum()
        ref_scale = anno['ref_scale'][i]    # Scalar
        if is_count.sum() > 0:
            # # for every predicted human
            # for j in range(pred_count):
            #     pred_joints = pred[j]
            #     dist[i,j,is_count] = anno_joints[is_count,:2].sub(pred_joints[is_count,:2]).pow_(2).sum(1).sqrt_().div_(ref_scale)
            vec = anno_joints[is_count,:2].reshape(1,-1,2) - pred[:,is_count,:2]    # [Np, cnt, 2]
            dist[i,:,is_count] = (np.sqrt(pow(vec,2).sum(2)) / ref_scale).T     # [Np, cnt]

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

# x
# anno [num_img] {'joints', 'ref_scale'}
# pred - joints ([num_img] batch(num_people) x num_joints x 3)
def eval_mAP(pred, anno, delta, dataset='coco'):
    """Evaluate predicted joints and return mAP."""
    oks_all = np.zeros((0))
    oks_num = 0

    # if the image in the predictions, then compute oks
    for i in range(len(pred)):
        if dataset == 'coco':
            oks = compute_oks(pred=pred[i], anno=anno[i], delta=delta)
        elif dataset == 'aic':
            oks = compute_oks_aic(pred=pred[i], anno=anno[i], delta=delta)

        oks_all = np.concatenate((oks_all, oks), axis=0)
        oks_num += oks.size

    # compute mAP by APs under different oks thresholds
    average_precision = []
    for threshold in np.linspace(0.5, 0.95, 10):
        average_precision.append(np.sum(oks_all > threshold) / np.float32(oks_num))
    # mAP = np.mean(average_precision)
    # average_precision.append(mAP)

    return average_precision    # list

# anno [num_img] {'joints', 'ref_scale'}
# pred - joints ([num_img] num_people x num_joints x 3)
def eval_AP(pred, anno, threshold):
    """Evaluate predicted_file and return AP."""
    num_joints = pred[0].shape[-2]
    GT_all = np.zeros((num_joints))
    score_all = np.zeros((0, num_joints))
    match_all = np.zeros((0, num_joints))
    # if the image in the predictions, then compute pck
    for i in range(len(pred)):
        # GT_count: number of joints in current image
        GT_count = (anno[i]['joints'][:,:,2] < 2).sum(0) # [num_joints]
        # GT_all: number of joints in all images
        GT_all += GT_count
        pck, match = compute_pck(pred=pred[i], anno=anno[i], threshold=threshold)
        if pck.size == 0:
            continue
        max_ = pck.max(0, keepdims=True)
        pck_ = (pck >= max_) * pck
        max_val = pck_.max(1)
        idx_pred = pck_.argmax(1)
        idx_pred = idx_pred[max_val != 0]  # torch.LongTensor
        idx_gt = np.nonzero(max_val)[0]     # torch.LongTensor
        if idx_pred.shape[0] != idx_gt.shape[0]:
            print "size does not match!!!"
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
