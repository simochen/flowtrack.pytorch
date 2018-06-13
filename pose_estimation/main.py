# -*- coding:utf-8 -*-
from __future__ import print_function, absolute_import

from config import opt
import os
import torch
import cv2
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
    if 'hg_1branch' in opt.model:
        num_classes = num_joints[opt.dataset] + opt.num_limbs
        if opt.with_bg:
            num_classes += 1
    else:
        num_classes = [num_joints[opt.dataset]+1, opt.num_limbs]

    # Step 1 : create model
    print("==> creating model '{}', num_stacks = {}, num_blocks = {}, num_feats = {}".format(
        opt.model, opt.num_stacks, opt.num_blocks, opt.num_feats) )
    model = getattr(models, opt.model)(block = getattr(models, opt.block),
                                       num_classes = num_classes,
                                       num_feats = opt.num_feats,
                                       num_stacks = opt.num_stacks,
                                       num_blocks = opt.num_blocks,
                                       depth = opt.depth)
    if opt.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
        #model = model.cuda()

    # Step 2 : loss function and optimizermAP
    lr = opt.lr
    # criterion = torch.nn.MSELoss(size_average=True).cuda()
    if opt.with_logits:
        criterion = FocalLoss(num_joints[opt.dataset], opt.with_bg)
    else:
        criterion = AMSELoss(num_joints[opt.dataset], opt.with_bg)
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
    prefix = '_'.join([opt.dataset, opt.model, opt.block])
    if opt.resume:
        model_path = os.path.join(work_dir, opt.resume)
        loss_path = os.path.join(work_dir, 'loss.t7')
        if os.path.exists(model_path):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(model_path)
            batch_cnt = checkpoint['batch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr = optimizer.param_groups[0]['lr']
            print("=> loaded checkpoint '{}' (batch {})"
                  .format(opt.resume, checkpoint['batch']))
            if os.path.exists(loss_path):
                result = torch.load(loss_path)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    cudnn.benchmark = True
    print('    Total params: %.4fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    # learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                    # factor=opt.lr_decay, patience=opt.patience, threshold=opt.lr_threshold,
                    # verbose=True, cooldown=opt.cooldown, min_lr=opt.min_lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30], gamma=0.1, last_epoch=opt.start_epoch-1)

    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=np.exp(np.log(10)/2))

    # Step 3 : data loading
    Dataset = getattr(datasets, opt.dataset.upper())
    train_data = Dataset(opt, split='train')
    valid_data = Dataset(opt, split='valid')
    train_loader = DataLoader(train_data, opt.batch_size, shuffle=True,
                              num_workers=opt.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_data, 1, shuffle=False,
                              num_workers=opt.num_workers, pin_memory=True)

    opt.start_epoch = batch_cnt // len(train_loader)

    # Step 4 : train and validate
    if opt.run_mode < 0:    # train & validate - lr decay according to train loss
        for epoch in range(opt.start_epoch, opt.max_epoch):
            print('\nEpoch: %d/%d | LR: %.8f' %(epoch+1, opt.max_epoch, lr))
            train(train_loader, valid_loader, model, criterion, optimizer, scheduler, opt)
    elif opt.run_mode == 0: # validate
        print('\nTest Mode: %d | Batch: %d | LR: %.8f' %(opt.test_mode, batch_cnt, lr))
        valid_loss, APs = validate(valid_loader, model, criterion, opt, calc_mAP=True)
        torch.save(APs, os.path.join(opt.checkpoint_path, opt.dataset, 'APs.t7'))
        print('Test loss: %.6f | mAP: %.6f'%(valid_loss, np.array(APs).mean()))
    elif opt.run_mode == 1: # validate prediction
        print('\nTest Mode: %d | Batch: %d | LR: %.8f' %(opt.test_mode, batch_cnt, lr))
        test_data = Dataset(opt, split='test_val')
        test_loader = DataLoader(test_data, 1, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)
        preds = test(test_loader, model, opt)
        pred_path = os.path.join(opt.checkpoint_path, opt.dataset, '_'.join(['preds','val',str(batch_cnt)])+'.json')
        with open(pred_path, 'w') as f:
            f.write(json.dumps(preds))
    elif opt.run_mode == 2: # validate output
        print('\nTest Mode: %d | Batch: %d | LR: %.8f' %(opt.test_mode, batch_cnt, lr))
        test_data = Dataset(opt, split='test_val')
        test_loader = DataLoader(test_data, 1, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)
        img_ids, joint_map, limb_map, sf_list = test_(test_loader, model, opt)
        output = {'image_id':img_ids, 'joint_hm': joint_map, 'limb_hm': limb_map, 'sf': sf_list}
        out_path = os.path.join(opt.checkpoint_path, opt.dataset, '_'.join(['output','val',str(batch_cnt)])+'.t7')
        torch.save(output, out_path)
    else:   # test prediction
        print('\nTest Mode: %d | Batch: %d | LR: %.8f' %(opt.test_mode, batch_cnt, lr))
        test_data = Dataset(opt, split='test_a')
        test_loader = DataLoader(test_data, 1, shuffle=False,
                                 num_workers=opt.num_workers, pin_memory=True)
        preds = test(test_loader, model, opt)
        pred_path = os.path.join(opt.checkpoint_path, opt.dataset, '_'.join(['preds',str(batch_cnt)])+'.json')
        with open(pred_path, 'w') as f:
            f.write(json.dumps(preds))


def train(train_loader, valid_loader, model, criterion, optimizer, scheduler, opt):
    global batch_cnt, best_loss, result, num_joints
    losses = []
    pbar = tqdm(total=len(train_loader))
    nan_cnt = 0
    for i, (data, heatmaps) in enumerate(train_loader):
        batch_cnt += 1

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
        loss = Variable(torch.zeros(1).cuda())
        if 'hg_1branch' in opt.model:
            for j in range(len(output)):
                loss.add_(criterion(output[j], target_hm, mask))
        else:
            for j in range(len(output)/2):
                loss.add_(criterion(output[2*j], target_joint, mask))
                loss.add_(criterion(output[2*j+1], target_limb, mask))

        loss.backward()
        optimizer.step()
        losses.append(loss.data[0])


        pbar.set_description("Training")
        pbar.set_postfix(loss=loss.data[0])
        pbar.update(1)

        # validate
        if batch_cnt % opt.print_every == 0:
            train_loss = np.mean(losses)
            losses = []
            if batch_cnt % (opt.valid_every * opt.print_every) == 0:
                valid_loss, APs = validate(valid_loader, model, criterion, opt, calc_mAP=True)
                valid_mAP = np.mean(APs)
                if opt.dataset == 'mpii':
                    valid_mAP = np.mean(APs[:-1])
                # scheduler.step(train_loss)
                print('Train loss: %.6f | Test loss: %.6f | mAP: %.6f'%(train_loss, valid_loss, valid_mAP))
                result['valid'].append(valid_loss)
                result['APs'].append(APs)
                # save checkpoint
                if train_loss < best_loss:
                    best_loss = train_loss
                    checkpoint = {
                        'batch': batch_cnt,
                        'model': opt.model,
                        'block': opt.block,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss,
                        'optimizer' : optimizer.state_dict() }
                    filename = '_'.join([opt.model, opt.block, 'best'])+'.pth'
                    filename = os.path.join(work_dir, filename)
                    torch.save(checkpoint, filename)
            else:
                print('Train loss: %.6f'%(train_loss))

            # scheduler.step(train_loss)
            scheduler.step()

            # save train and valid loss every epoch
            result['batch'].append(batch_cnt)
            result['train'].append(train_loss)
            torch.save(result, os.path.join(work_dir, 'loss.t7'))

        if batch_cnt % opt.save_every_batch == 0:
            checkpoint = {
                'batch': batch_cnt,
                'model': opt.model,
                'block': opt.block,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict() }
            filename = '_'.join([opt.model, opt.block, str(batch_cnt)])+'.pth'
            filename = os.path.join(work_dir, filename)
            torch.save(checkpoint, filename)

    pbar.close()

def validate(valid_loader, model, criterion, opt, calc_mAP=False):
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
            mask = torch.from_numpy(mask[np.newaxis,:])
            mask = Variable(mask, volatile=True).cuda(async=True)
        else:
            mask = None
        inputs = Variable(data, volatile=True).cuda()
        target_hm = Variable(heatmaps, volatile=True).cuda(async=True)

        # compute output
        output = model(inputs)
        # compute loss
        loss = Variable(torch.zeros(1).cuda())

        if 'hg_1branch' in opt.model:
            for j in range(len(output)):
                loss.add_(criterion(output[j], target_hm, mask))
        else:
            for j in range(len(output)/2):
                loss.add_(criterion(output[2*j], target_joint, mask))
                loss.add_(criterion(output[2*j+1], target_limb, mask))
        if not np.isnan(loss.data[0]):
            losses.append(loss.data[0])
        else:
            nan_cnt += 1
            print(nan_cnt)

        if 'hg_1branch' in opt.model:
            out_maps = output[-1].data.squeeze(0)
            out_joint = out_maps[:num_joints[opt.dataset]]
            out_limb = out_maps[num_joints[opt.dataset]:]
            if opt.with_bg:
                out_limb = out_limb[:-1]
        else:
            out_joint = output[-2].data.squeeze(0)   # [1 x C x H x W]
            out_limb = output[-1].data.squeeze(0)
        if opt.with_logits:
            pred = piece_people(out_joint.sigmoid(), out_limb.sigmoid(), opt)
        else:
            pred = piece_people(out_joint, out_limb, opt)
        # [num_person, num_joints, 3] (x, y, score)
        preds.append(pred)

        if calc_mAP:
            anno = {'joints':meta['joints'].squeeze(0).numpy(), 'ref_scale':meta['ref_scale'].squeeze(0).numpy()}
            annos.append(anno)


        pbar.set_description("Testing")
        pbar.set_postfix(loss=loss.data[0])
        pbar.update(1)
    pbar.close()
    # set training mode
    model.train()

    if calc_mAP:
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

    else:
        return np.mean(losses)

def test(test_loader, model, opt):
    global num_joints
    # set evaluation model
    model.eval()
    pbar = tqdm(total=len(test_loader))
    preds = []
    for i, data in enumerate(test_loader):
        inputs = Variable(data, volatile=True).cuda()
        # compute output
        output = model(inputs)
        tags = output[-1][:,:num_joints[opt.dataset]*opt.tag_dim]
        #tags = tags.view(tags.size(0),opt.tag_dim,-1).permute(0,2,1).contiguous()
        hms = output[-1][:,num_joints[opt.dataset]*opt.tag_dim:]

        # generate predictions
        if 'hg_1branch' in opt.model:
            out_maps = output[-1].data.squeeze(0)
            out_joint = out_maps[:num_joints[opt.dataset]+1]
            out_limb = out_maps[num_joints[opt.dataset]+1:]
        else:
            out_joint = output[-2].data.squeeze(0)   # [1 x C x H x W]
            out_limb = output[-1].data.squeeze(0)

        sf = opt.stride / sf[0]
        if opt.test_mode == 0:
            joint_map = cv2.resize(out_joint.cpu().numpy().transpose(1,2,0), (0,0),fx=sf,fy=sf)
            limb_map = cv2.resize(out_limb.cpu().numpy().transpose(1,2,0), (0,0), fx=sf,fy=sf)
            out_joint = torch.from_numpy(joint_map.transpose(2,0,1)).cuda()
            out_limb = torch.from_numpy(limb_map.transpose(2,0,1)).cuda()
        keypoint = piece_pred(out_joint, out_limb, opt, dataset=opt.dataset)
        if opt.test_mode == 1:
            keypoint[:,:,:2] = (keypoint[:,:,:2] * sf + 1).round().astype(int)
        # [num_person, num_joints, 3] (x, y, is_visible)
        pred = {'image_id':image_id[0], 'keypoint_annotations':{}}
        for i in range(keypoint.shape[0]):
            key = "human" + str(i+1)
            pred['keypoint_annotations'][key] = keypoint[i].reshape(-1).tolist()

        preds.append(pred)

        pbar.set_description("Testing")
        pbar.update(1)
    pbar.close()
    # set training mode
    model.train()

    return preds

def test_(test_loader, model, opt):
    global num_joints
    # set evaluation model
    model.eval()
    pbar = tqdm(total=len(test_loader))
    joint_map = []
    limb_map = []
    sf_list = []
    img_ids = []
    for i, data in enumerate(test_loader):
        inputs = Variable(data, volatile=True).cuda()
        # compute output
        output = model(inputs)

        # generate predictions
        if 'hg_1branch' in opt.model:
            out_maps = output[-1].data.squeeze(0)
            out_joint = out_maps[:num_joints[opt.dataset]+1]
            out_limb = out_maps[num_joints[opt.dataset]+1:]
        else:
            out_joint = output[-2].data.squeeze(0)   # [1 x C x H x W]
            out_limb = output[-1].data.squeeze(0)

        sf = opt.stride / sf[0]

        img_ids.append(image_id[0])
        joint_map.append(out_joint.cpu())
        limb_map.append(out_limb.cpu())
        sf_list.append(sf)

        pbar.set_description("Testing")
        pbar.update(1)
    pbar.close()
    # set training mode
    model.train()

    return img_ids, joint_map, limb_map, sf_list


if __name__ == '__main__':
    main()
