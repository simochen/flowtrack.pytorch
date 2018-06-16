from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import argparse
import cv2
# detection
# pose
# flownet
# tracking
from tracking.net_utils import detect, pose_est, flow_est
from tracking.flow_utils import box_propagation

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Estimation in Video')
    parser.add_augument('--detect_net', help='vgg16, res50, res101, res152, mobile',
                        default='res101', type=str)
    parser.add_augument('--detect_model', help='human detection model', type=str)
    

def process_frame(detect_net, pose_net, flownet, prev_im, cur_im, prev_dets, prev_keypoints):
    flow = flow_est(flownet, prev_im, cur_im)
    prop_boxes = box_propagation(prev_keypoints, flow)
    prop_dets = np.concatenate((prop_boxes, prev_dets[:,4][:,np.newaxis]), axis=1)
    cur_dets = detect(detect_net, cur_im, PERSON_ID, prop_dets=prop_dets)
    cur_keypoints = pose_est(pose_net, cur_im, cur_dets[:,:4])

    return cur_dets, cur_keypoints    

