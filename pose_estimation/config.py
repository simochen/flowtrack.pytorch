# -*- coding:utf-8 -*-
import warnings
import numpy as np

class DefaultConfig(object):

    # ====================== General Options ======================
    # dataset = 'mpii'
    # data_path = './data/mpii'
    dataset = 'coco'
    data_path = './data/coco'
    # dataset = 'aic'
    # data_path = './data/aic'
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
    model = 'deconv_resnet' # 使用的模型，名字必须与models/__init__.py中的名字一致
    backbone = 'resnet152'
    stride = 4
    with_logits = False
    with_bg = True

    # ====================== Training Options ======================
    batch_size = 128 # training batch size
    start_epoch = 0
    max_epoch = 140
    lr = 1e-3 # initial learning rate
    min_lr = 1e-5 #5e-6
    momentum = 0.9
    weight_decay = 0 # 1e-4 # 损失函数

    # ====================== checkpoint Options ======================
    checkpoint_path = './checkpoints'
    exp_id = 'mse'
    #resume = 'hg_1branch_bottleneck_best.pth'
    resume = None   # resume from a checkpoint
    run_mode = -1   # <0:train&val; 0:search lr; 1:valid models; 2:test pred
    test_mode = 0   # 0:resize to input_res; 1:resize according to scale
    with_gt = 0
    save_every = 5  # save every N epoch
    print_every = 100 # print every N batch
    valid_every = 20  # validate
    save_every_batch = 10000

    # ====================== Evaluation Options ======================
    test_id = 214
    test_json = 'mpii_val_group.json'
    test_factor = 1
    nms_threshold = 0.0
    nms_window_size = 3
    norm_threshold = -1
    inter_min = 10
    inter_max = 20
    interval = 3
    inter_threshold = 0.1
    joints_min = 3
    person_threshold = 0.1
    flip = False
    scale_search = [0.5,1,1.5,2]


def parse(self, kwargs):
        '''
        根据字典 kwargs 更新 config 参数
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
# opt.parse = parse
