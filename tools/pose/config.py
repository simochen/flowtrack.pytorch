# -*- coding:utf-8 -*-
import warnings
import numpy as np

class DefaultConfig(object):

    # ====================== General Options ======================
    # dataset = 'mpii'
    # data_path = './data/mpii'
    dataset = 'coco'
    data_path = './data/coco'
    num_workers = 4 # how many workers for loading data
    use_gpu = True  # user GPU or not

    # ======================== Data Options ========================
    # Data Augmentation
    # Flip
    flip_prob = 0.5
    # Scale
    scale_prob = 1.0
    scale_min = 0.7
    scale_max = 1.3
    # Rotate
    rotate_prob = 1.0
    rotate_degree_max = 40

    # Resolution
    input_res = (256, 192)
    # heatmap
    sigma = 1

    # ======================= Model Options =======================
    model = 'deconv_resnet'
    backbone = 'resnet152'
    stride = 4
    with_logits = False
    with_bg = False
    with_mask = True

    # ====================== Training Options ======================
    batch_size = 128 # training batch size
    test_batch_size = 128
    start_epoch = 0
    max_epoch = 140
    optim = 'adam'  # optimizer
    lr = 2e-3 # initial learning rate
    momentum = 0.9
    weight_decay = 0 # loss function

    # ====================== checkpoint Options ======================
    checkpoint_path = './checkpoints'
    exp_id = 'exp_01'
    resume = None   # resume from a checkpoint
    save_every = 10  # save every N epoch
    
    # ====================== Evaluation Options ======================
    nms_threshold = 0.0
    nms_window_size = 3


def parse(self, kwargs):
    '''
    update config parameters according to kwargs
    '''
    for k,v in kwargs.iteritems():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k,v)

    print('user config:')
    for k,v in self.__class__.__dict__.iteritems():
        if not k.startswith('__'):
            print(k,getattr(self,k))


DefaultConfig.parse = parse
opt = DefaultConfig()
