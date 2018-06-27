from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import argparse
import cv2
import numpy as np
# detection
from detection.nets.vgg16 import vgg16
from detection.nets.resnet_v1 import resnetv1
from detection.nets.mobilenet_v1 import mobilenetv1
# pose
from pose.models import deconv_resnet
# flownet
from flownet.model import models
# tracking
from tracking.net_utils import detect, pose_est, flow_est
from tracking.flow_utils import box_propagation

def parse_args():
    parser = argparse.ArgumentParser(description='Pose Estimation in Video')
    parser.add_argument('--detect_net', default='res101', type=str, , help='vgg16, res50, res101, res152, mobile')
    parser.add_argument('--detect_dataset', default='coco', type=str, help='coco, pascal_voc')
    parser.add_argument('--detect_model', type=str, help='human detection model')
    parser.add_argument('--pose_backbone', default=152, type=int, help='50, 101, 152')
    parser.add_argument('--pose_model', type=str, help='pose estimation model')
    parser.add_argument('--flow_net', type=str, default='FlowNet2S', help='FlowNet2, FlowNet2C, FlowNet2S, FlowNet2SD, FlowNet2CS, FlowNet2CSS')
    paeser.add_argument('--flow_model', type=str, help='optical flow estimation model')
    args = parser.parse_args()
    return args


def process_frame(detect_net, pose_net, flownet, prev_im, cur_im, prev_dets, prev_keypoints):
    flow = flow_est(flownet, prev_im, cur_im)
    prop_boxes = box_propagation(prev_keypoints, flow)
    prop_dets = np.concatenate((prop_boxes, prev_dets[:,4][:,np.newaxis]), axis=1)
    cur_dets = detect(detect_net, cur_im, PERSON_ID, prop_dets=prop_dets)
    cur_keypoints = pose_est(pose_net, cur_im, cur_dets[:,:4])

    return cur_dets, cur_keypoints

if __name__ == '__main__':
    global PERSON_ID
    args = parse_args()

    if not os.path.isfile(args.detect_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(args.detect_model))

    # detection network
    if args.detect_net == 'vgg16':
        detect_net = vgg16()
    elif args.detect_net == 'res50':
        detect_net = resnetv1(num_layers=50)
    elif args.detect_net == 'res101':
        detect_net = resnetv1(num_layers=101)
    elif args.detect_net == 'res152':
        detect_net = resnetv1(num_layers=152)
    elif args.detect_net == 'mobile':
        detect_net = mobilenetv1()
    else:
        raise NotImplementedError

    if args.detect_dataset == 'pascal_voc':
        num_classes = 21
        PERSON_ID = 15
    elif args.detect_dataset == 'coco':
        num_classes = 81
        PERSON_ID = 1

    detect_net.create_architecture(num_classes, tag='default', anchor_scales=[8, 16, 32])
    detect_net.load_state_dict(torch.load(args.detect_model))
    detect_net.eval()
    detect_net.cuda()

    # pose estimation network
    pose_net = deconv_resnet(num_layers=args.pose_backbone, num_classes=17, pretrained=False)
    pose_model = torch.load(args.pose_model)
    pose_net.load_state_dict(pose_model['state_dict'])
    pose_net.eval()
    pose_net.cuda()

    # optical flow network
    flow_net = getattr(models, args.flow_net)()
    flow_model = torch.load(args.flow_model)
    flow_net.load_state_dict(flow_model['state_dict'])
    flow_net.eval()
    flow_net.cuda()

    
