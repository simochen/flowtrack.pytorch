from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

from detection.model.test import im_detect
from detection.model.nms_wrapper import nms as detect_nms
from pose.transforms import transfrom_image
from pose.evaluation import final_preds

def detect(net, im, person_id, thresh=0.3, prop_dets=None):
    """Detect human in an image.
    Arguments:
        net (nn.Module)
        im (ndarray): a color image in BGR order
        person_id: class id of 'person'
        thresh: NMS threshold
    Returns:
        boxes (ndarray): [N, 5]
                        detected human bounding boxes
    """
    scores, boxes = im_detect(net, im)
    person_boxes = boxes[:, 4*person_id:4*(person_id+1)]
    person_scores = scores[:, person_id]
    dets = np.hstack((person_boxes,person_scores[:,np.newaxis])).astype(np.float32)
    if prop_dets != None:
        dets = np.concatenate((dets, prop_dets), axis=0)
    keep = detect_nms(torch.from_numpy(dets), thresh)
    dets = dets[keep.numpy(), :]

    return dets

def pose_est(net, im, boxes, inp_res=(256,192), max_batch=32, flip_test=False):
    """Parallel single person pose estimation.
    Arguments:
        net (nn.Module)
        im (ndarray): a color image in BGR order
        boxes (ndarray): [N, 4]
        inp_res (tuple): input resolution of network
        max_batch: max batch size to feed in network
    Returns:
        keypoints (ndarray): [N, num_kpts, 3]
                             detected keypoints
    """
    num_boxes = boxes.shape[0]
    centers = np.stack((boxes[:,[0,2]].mean(1), boxes[:,[1,3]].mean(1)), axis=1)  # [N,2]
    scales = np.maximum(boxes[:,3]-boxes[:,1], (boxes[:,2]-boxes[:,0])/inp_res[1]*inp_res[0])
    for i in range(num_boxes):
        im_crop = transfrom_image(im, centers[i], scales[i], inp_res)
        im_crop = im_crop.view(1,3,inp_res[0],inp_res[1])
        if i == 0:
            images = im_crop
        else:
            images = torch.cat((images, im_crop), dim=0)

    num_batch = int(np.ceil(num_boxes / max_batch))
    for i in range(num_batch):
        end = max(num_boxes, max_batch*(i+1))
        im_batch = Variable(images[max_batch*i:end], volatile=True).cuda()
        hm_batch = net(im_batch)
        if i == 0:
            heatmaps = hm_batch
        else:
            heatmaps = torch.cat((heatmaps, hm_batch), dim=0)
    coords, scores = final_preds(heatmaps, centers, scales, adjust_coords=True)
    keypoints = np.concatenate((coords, scores), axis=2)

    return keypoints

def flow_est(net, prev_im, cur_im):
    """Parallel single person pose estimation.
    Arguments:
        net (nn.Module)
        prev_im, cur_im (ndarray): color images in BGR order
    Returns:
        flow (ndarray): [2, H, W]
                        estimated optical flow
    """
    # convert to RGB order
    prev_im = np.flip(prev_im, axis=2)
    cur_im = np.flip(cur_im, axis=2)
    # [batch, 3(RGB), 2(pair), H, W]
    imgs = np.array([[prev_im, cur_im]]).transpose((0,4,1,2,3)).astype(np.float32)
    imgs = Variable(torch.from_numpy(imgs), volatile=True).cuda()
    # [batch, 2, H, W]
    flow = net(imgs).data.cpu()
    flow = flow.squeeze(0).numpy()

    return flow
