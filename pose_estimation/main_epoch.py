# -*- coding:utf-8 -*-
from __future__ import print_function, absolute_import

from config import opt
import os
import torch
import cv2
import scipy.io
import json
import models
import datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim.lr_scheduler

import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils import *

best_loss = 1
batch_cnt = 0
result = {'train':[], 'valid':[], 'APs':[]}
num_joints = {'mpii':15, 'aic':14, 'coco':17}
# torch.manual_seed(1)    # 设置随机数种子为固定值

def main(**kwargs):
    global best_loss, num_joints
    # Step 0 : parse Options
    opt.parse(kwargs)
    # check checkpoint path
    work_dir = os.path.join(opt.checkpoint_path, opt.dataset, opt.exp_id)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    # dataset parse
    num_classes = num_joints[opt.dataset]
    if opt.with_bg:
        num_classes += 1

    if opt.backbone == 'resnet50':
        num_layers = 50
    elif opt.backbone == 'resnet101':
        num_layers = 101
    elif opt.backbone == 'resnet152':
        num_layers = 152

    # Step 1 : create model
    print("==> creating model '{}', backbone = {}".format(
        opt.model, opt.backbone) )
    model = getattr(models, opt.model)(num_layers = num_layers, 
                                       num_classes = num_classes, 
                                       pretrained = True)
    if opt.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        #model = model.cuda()

    # Step 2 : loss function and optimizermAP
    lr = opt.lr
    if opt.with_logits:
        criterion = FocalLoss(num_joints[opt.dataset], opt.with_bg)
    else:
        criterion = AMSELoss(num_joints[opt.dataset], opt.with_bg).cuda()
        # criterion = torch.nn.MSELoss(size_average=True).cuda()    
    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                                 lr = lr,
    #                                 momentum = opt.momentum,
    #                                 weight_decay = opt.weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(),
    #                             lr = lr,
    #                             momentum = opt.momentum,
    #                             weight_decay = opt.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = lr,
                                 weight_decay = opt.weight_decay)
    loss = {'epoch':[], 'train':[], 'valid':[], 'APs':[]}

    # (Optional) resume from checkpoint
    prefix = '_'.join([opt.dataset, opt.model, opt.backbone])
    if opt.resume:
        model_path = os.path.join(work_dir, opt.resume)
        loss_path = os.path.join(work_dir, 'loss.t7')
        if os.path.exists(model_path):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(model_path)
            opt.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr = optimizer.param_groups[0]['lr']
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt.resume, checkpoint['epoch']))
            if os.path.exists(loss_path):
                loss = torch.load(loss_path)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True
    print('    Total params: %.4fM' % (sum(p.numel() if p.requires_grad else 0 for p in model.parameters())/1000000.0))

    # learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                    # factor=opt.lr_decay, patience=opt.patience, threshold=opt.lr_threshold,
                    # verbose=True, cooldown=opt.cooldown, min_lr=opt.min_lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,120], gamma=0.1, last_epoch=opt.start_epoch-1)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=np.exp(np.log(10)/2))

    # Step 3 : data loading
    Dataset = getattr(datasets, opt.dataset.upper())
    train_data = Dataset(opt, split='train')
    valid_data = Dataset(opt, split='valid')
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_data, opt.batch_size, shuffle=False,
                              num_workers=opt.num_workers, pin_memory=True)

    # Step 4 : train and validate
    if opt.run_mode < 0:  # train and validate
        for epoch in range(opt.start_epoch, opt.max_epoch):
            scheduler.step()
            print('\nEpoch: %d/%d | LR: %.8f' %(epoch+1, opt.max_epoch, optimizer.param_groups[0]['lr']))
            train_loss = train(train_loader, model, criterion, optimizer, opt)
            # valid_loss = validate(valid_loader, model, criterion, opt)
            valid_loss, APs = validate(valid_loader, model, criterion, opt)

            valid_mAP = np.mean(APs)
            if opt.dataset == 'mpii':
                valid_mAP = np.mean(APs[:-1])

            # scheduler.step(train_loss)
            print('Train loss: %.6f | Test loss: %.6f | mAP: %.6f'%(train_loss, valid_loss, valid_mAP))
            # print('Train loss: %.6f | Test loss: %.6f'%(train_loss, valid_loss))
            # save train and valid loss every epoch
            loss['epoch'].append(epoch+1)
            loss['train'].append(train_loss)
            loss['valid'].append(valid_loss)
            loss['APs'].append(APs)
            torch.save(loss, os.path.join(work_dir, 'loss.t7'))

            # save checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': opt.model,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict() }
                filename = '_'.join([opt.model, opt.backbone, 'best'])+'.pth'
                filename = os.path.join(work_dir, filename)
                torch.save(checkpoint, filename)
            if (epoch+1) % opt.save_every == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': opt.model,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict() }
                filename = '_'.join([opt.model, opt.backbone, str(epoch+1)])+'.pth'
                filename = os.path.join(work_dir, filename)
                torch.save(checkpoint, filename)

    elif opt.run_mode == 0: # search for learning rate
        for epoch in range(opt.start_epoch, opt.max_epoch):
            scheduler.step()
            print('\nEpoch: %d/%d | LR: %.8f' %(epoch+1, opt.max_epoch, optimizer.param_groups[0]['lr']))
            train_loss = train(train_loader, model, criterion, optimizer, opt)
            # scheduler.step(train_loss)
            print('Train loss: %.6f'%(train_loss))
            # save train and valid loss every epoch
            loss['epoch'].append(epoch+1)
            loss['train'].append(train_loss)
            torch.save(loss, os.path.join(work_dir, 'loss.t7'))

            # save checkpoint
            if train_loss < best_loss:
                best_loss = train_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': opt.model,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict() }
                filename = '_'.join([opt.model, opt.backbone, 'best'])+'.pth'
                filename = os.path.join(work_dir, filename)
                torch.save(checkpoint, filename)
            if (epoch+1) % opt.save_every == 0:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': opt.model,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict() }
                filename = '_'.join([opt.model, opt.backbone, str(epoch+1)])+'.pth'
                filename = os.path.join(work_dir, filename)
                torch.save(checkpoint, filename)

    elif opt.run_mode == 1:   # test models
        for epoch in range(opt.start_epoch, opt.max_epoch):
            model_path = '_'.join([opt.model, opt.backbone, str(epoch+1)])+'.pth'
            model_path = os.path.join(work_dir, model_path)
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                model.load_state_dict(checkpoint['state_dict'])
                print('\nTest Only | Epoch: %d' %(epoch+1))
                valid_loss, APs = validate(valid_loader, model, criterion, opt)
                valid_mAP = np.mean(APs)
                if opt.dataset == 'mpii':
                    valid_mAP = np.mean(APs[:-1])
                loss['epoch'].append(epoch+1)
                loss['valid'].append(valid_loss)
                loss['APs'].append(APs)
                torch.save(loss, os.path.join(work_dir, 'loss_.t7'))
                print('Test loss: %.6f | mAP: %.6f'%(valid_loss, valid_mAP))

    elif opt.run_mode == 2:
        preds = test(opt.test_json, model, opt)
        pred_path = os.path.join(work_dir, '_'.join(['preds',str(opt.test_id)])+'.mat')
        # with open(pred_path, 'w') as f:
        #     f.write(json.dumps(preds))
        scipy.io.savemat(pred_path, mdict={'det':preds})

def train(train_loader, model, criterion, optimizer, opt):
    losses = []
    pbar = tqdm(total=len(train_loader))
    nan_cnt = 0
    for i, (data, heatmaps) in enumerate(train_loader):
        if opt.dataset == 'coco':
            mask = data[:,-1].numpy().transpose((1,2,0))
            data = data[:,:-1]
            mask = cv2.resize(mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            mask = torch.from_numpy(mask.transpose((2,0,1))).contiguous()
            mask = Variable(mask).cuda(async=True)
        else:
            mask = None

        inputs = Variable(data).cuda()
        target_hm = Variable(heatmaps).cuda(async=True)
        # zero gradient
        optimizer.zero_grad()
        # compute output
        output = model(inputs)
        # compute loss
        loss = criterion(output, target_hm, mask)

        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])

        pbar.set_description("Training")
        pbar.set_postfix(loss=loss.data[0])
        pbar.update(1)
    pbar.close()
    return np.mean(losses)

def validate(valid_loader, model, criterion, opt):
    global num_joints
    losses = []
    APs = []
    # set evaluation model
    model.eval()
    pbar = tqdm(total=len(valid_loader))
    preds = []
    annos = []
    nan_cnt = 0
    for i, (data, heatmaps, meta) in enumerate(valid_loader):
        if opt.dataset == 'coco':
            mask = data[:,-1].numpy().transpose((1,2,0))
            data = data[:,:-1]
            mask = cv2.resize(mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            mask = torch.from_numpy(mask.transpose((2,0,1))).contiguous()
            mask = Variable(mask, volatile=True).cuda(async=True)
        else:
            mask = None
        inputs = Variable(data, volatile=True).cuda()
        target_hm = Variable(heatmaps, volatile=True).cuda(async=True)

        # compute output
        output = model(inputs)
        # compute loss
        loss = criterion(output, target_hm, mask)

        if not np.isnan(loss.data[0]):
            losses.append(loss.data[0])
        else:
            nan_cnt += 1
            print(nan_cnt)

        out_joint = output
        if opt.with_bg:
            out_joint = out_joint[:,:-1]
        if opt.with_logits:
            # pred = piece_people(out_joint.sigmoid(), out_limb.sigmoid(), opt)
            pred = nms_single(out_joint.sigmoid(), opt.nms_threshold, opt.nms_window_size)
        else:
            # pred = piece_people(out_joint, out_limb, opt)
            pred = nms_single(out_joint, opt.nms_threshold, opt.nms_window_size)
        # [batch(num_person), num_joints, 3] (x, y, score)
        preds.append(pred)
        anno = {'joints':meta['joints'].numpy(), 'ref_scale':meta['ref_scale'].numpy()}
        annos.append(anno)

        pbar.set_description("Testing")
        pbar.set_postfix(loss=loss.data[0])
        pbar.update(1)
    pbar.close()
    # set training mode
    model.train()

    if opt.dataset == 'mpii':
        mAP = eval_AP(preds, annos, 0.5)
    elif opt.dataset == 'aic':
        delta = 2 * np.array([0.01388152, 0.01515228, 0.01057665, 0.01417709,
                        0.01497891, 0.01402144, 0.03909642, 0.03686941, 0.01981803,
                        0.03843971, 0.03412318, 0.02415081, 0.01291456, 0.01236173])
        mAP = eval_mAP(preds, annos, delta, 'aic')
    elif opt.dataset == 'coco':
        delta = 2 * np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
        mAP = eval_mAP(preds, annos, delta)

    return np.mean(losses), mAP


def test(jsonfile, model, opt):
    global num_joints
    # load jsonfile
    jsonfile = os.path.join(opt.data_path, jsonfile)
    with open(jsonfile) as anno_file:
        annolist = json.load(anno_file)

    img_folder = os.path.join(opt.data_path, 'images')

    if opt.dataset == 'mpii':
        mean = [110.3341, 113.2268, 112.3129]

    # set evaluation model
    model.eval()
    pbar = tqdm(total=len(annolist))
    preds = []
    for anno in annolist:
        img_path = os.path.join(img_folder, anno['img_name'])
        img = cv2.imread(img_path)
        img = mean_sub(img, np.array(mean)).astype(np.float32)
        h = img.shape[0]
        w = img.shape[1]
        if opt.test_mode == 0:
            sf = max(float(opt.input_res[0])/h, float(opt.input_res[1])/w) * opt.test_factor
        elif opt.test_mode == 1:
            sf = opt.test_factor / anno['scale']
        multiplier = [x * sf for x in opt.scale_search]
        num_ch = num_joints[opt.dataset] + opt.num_limbs
        if opt.with_bg:
            num_ch += 1
        heatmap_avg = np.zeros((h, w, num_ch))
        for m in range(len(multiplier)):
            scale = multiplier[m]
            if scale <= 1:
               interp = cv2.INTER_AREA
            else:
               interp = cv2.INTER_CUBIC
            # interp = cv2.INTER_CUBIC
            img_test = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=interp)
            test_h = img_test.shape[0]
            test_w = img_test.shape[1]
            if opt.flip:
                img_flip = np.flip(img_test, 1).copy()  # H x W x C
            if img_test.shape[0] % opt.stride != 0:
                pad_h = opt.stride - img_test.shape[0] % opt.stride
                img_test = np.concatenate((img_test, np.zeros((pad_h, img_test.shape[1], img_test.shape[2]))), axis=0)
                if opt.flip:
                    img_flip = np.concatenate((img_flip, np.zeros((pad_h, img_test.shape[1], img_test.shape[2]))), axis=0)
            if img_test.shape[1] % opt.stride != 0:
                pad_w = opt.stride - img_test.shape[1] % opt.stride
                img_test = np.concatenate((img_test, np.zeros((img_test.shape[0], pad_w, img_test.shape[2]))), axis=1)
                if opt.flip:
                    img_flip = np.concatenate((img_flip, np.zeros((img_test.shape[0], pad_w, img_test.shape[2]))), axis=1)

            img_test = torch.from_numpy(img_test.astype(np.float32).transpose((2,0,1)))
            if opt.flip:
                img_flip = torch.from_numpy(img_flip.astype(np.float32).transpose((2,0,1)))
                img_test = torch.stack((img_test, img_flip))
            else:
                img_test.unsqueeze_(0)
            inputs = Variable(img_test, volatile=True)
            if opt.use_gpu:
                inputs = inputs.cuda()
            # compute output
            output = model(inputs)
            out_maps = output[-1].data
            if opt.with_logits:
                out_maps = out_maps.sigmoid()
            if test_h >= h:
               interp = cv2.INTER_AREA
            else:
               interp = cv2.INTER_CUBIC
            # interp = cv2.INTER_CUBIC
            for i in range(out_maps.size(0)):
                out_map = out_maps[i].cpu().numpy().transpose((1,2,0))
                out_map = cv2.resize(out_map, (0,0), fx=opt.stride, fy=opt.stride, interpolation=cv2.INTER_CUBIC)
                out_map = out_map[:test_h,:test_w,:]
                out_map = cv2.resize(out_map, (w,h), interpolation=cv2.INTER_CUBIC)
                if i == 1:
                    out_map = np.flip(out_map, 1).copy()
                heatmap_avg += out_map / out_maps.size(0) / len(multiplier)

        joint_avg = heatmap_avg[:,:,:num_joints[opt.dataset]]
        limb_avg = heatmap_avg[:,:,num_joints[opt.dataset]:]
        if opt.with_bg:
            limb_avg = limb_avg[:,:,:-1]

        out_joint = cv2.GaussianBlur(joint_avg, (0,0), sigmaX=5, borderType=cv2.BORDER_REFLECT)
        out_joint = torch.from_numpy(out_joint.transpose((2,0,1)))
        out_limb = torch.from_numpy(limb_avg.transpose((2,0,1)))
        if opt.use_gpu:
            out_joint = out_joint.cuda()
            out_limb = out_limb.cuda()

        ss = 0.8*anno['scale']*200.0 + 0.2*h
        if opt.with_gt:
            joint_dict = anno['joints']
            joints = np.array(joint_dict['_ArrayData_']).reshape(joint_dict['_ArraySize_'][::-1]).transpose(2,1,0)
            joints[:,:,0:2] -= 1
        else:
            joints = None
        pred = piece_people(out_joint, out_limb, opt, joints, scale=ss)

        pred[:,:,:2] += 1

        # [num_person, num_joints, 3] (x, y, score)
        preds.append(pred)

        pbar.set_description("Testing")
        pbar.set_postfix(pred=pred.shape[0], j=joint_avg.mean(), jm=joint_avg.max(), l=limb_avg.mean(), lm=limb_avg.max())
        pbar.update(1)
    pbar.close()
    # set training mode
    model.train()

    return preds

if __name__ == '__main__':
    main()
