# -*- coding:utf-8 -*-
import warnings
import numpy as np

class DefaultConfig(object):

    # # ====================== General Options ======================
    # dataset = 'coco'
    # data_path = './data/coco'
    # num_workers = 4 # how many workers for loading data
    # use_gpu = True  # user GPU or not
    #
    # # ======================== Data Options ========================
    # # Data Augmentation
    # # Flip
    # flip_prob = 0.5
    # # Scale
    # scale_prob = 1.0
    # scale_min = 0.7
    # scale_max = 1.3
    # # Rotate
    # rotate_prob = 0.6
    # rotate_degree_max = 40
    #
    # # Resolution
    # input_res = (256, 192)
    # # heatmap
    # sigma = 2
    #
    # # ======================= Model Options =======================
    # model = 'deconv_resnet'
    # backbone = 'resnet50'
    # stride = 4
    # with_logits = False
    # with_mask = False
    # with_bg = False
    # target_type = 'gaussian'
    #
    # # ====================== Training Options ======================
    # debug = False
    # batch_size = 32 # training batch size
    # test_batch_size = 32
    # start_epoch = 0
    # max_epoch = 140
    # optim = 'adam'  # optimizer
    # lr = 1e-3 # initial learning rate
    # momentum = 0.9
    # weight_decay = 1e-4 # loss function
    # draw_freq = 100
    #
    # # ====================== checkpoint Options ======================
    # checkpoint_path = './checkpoints'
    # exp_id = 'exp_01'
    # resume = None#'deconv_resnet_resnet50_best.pth'   # resume from a checkpoint
    # save_every = 5  # save every N epoch
    # run_type = 'trainval'
    #
    # # ====================== Evaluation Options ======================
    # adjust_coords = True
    # flip_test = False
    # oks_threshold = 0.9
    # kpt_threshold = 0.2

    # ====================== General Options ======================
    dataset = 'mpii'
    data_path = './data/mpii'
    num_workers = 4 # how many workers for loading data
    use_gpu = True  # user GPU or not

    # ======================== Data Options ========================
    # Data Augmentation
    # Flip
    flip_prob = 0.5
    # Scale
    scale_prob = 1.0
    scale_min = 0.75
    scale_max = 1.25
    # Rotate
    rotate_prob = 0.6
    rotate_degree_max = 30

    # Resolution
    input_res = (256, 256)
    # heatmap
    sigma = 2

    # ======================= Model Options =======================
    model = 'fpn'
    backbone = 'densenet121'
    stride = 4
    with_logits = False
    with_mask = False
    with_bg = False
    target_type = 'gaussian'
    pretrained = True
    tensorboard = False

    # ====================== Training Options ======================
    debug = False
    batch_size = 32 # training batch size
    test_batch_size = 32
    start_epoch = 0
    max_epoch = 140
    optim = 'adam'  # optimizer
    lr = 1e-3 # initial learning rate
    momentum = 0.9
    weight_decay = 1e-4 # loss function
    draw_freq = 100

    # ====================== checkpoint Options ======================
    checkpoint_path = './checkpoints'
    exp_id = 'pretrained_01'
    resume = None#'deconv_resnet50_best.pth'   # resume from a checkpoint
    save_every = 20  # save every N epoch
    run_type = 'trainval'

    # ====================== Evaluation Options ======================
    adjust_coords = True
    flip_test = False
    oks_threshold = 0.9
    kpt_threshold = 0.2


def parse(self, kwargs):
    '''
    update config parameters according to kwargs
    '''
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("Warning: opt has not attribut %s" %k)
        setattr(self,k,v)

    print('user config:')
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k,getattr(self,k))


DefaultConfig.parse = parse
opt = DefaultConfig()
