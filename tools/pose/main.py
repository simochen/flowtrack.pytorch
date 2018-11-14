# -*- coding:utf-8 -*-
from __future__ import print_function, absolute_import

import _init_paths
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

import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils import *

best_loss = 1
batch_cnt = 0
result = {'train':[], 'valid':[], 'APs':[]}
num_joints = {'mpii':15, 'aic':14, 'coco':17}
torch.manual_seed(1)    # 设置随机数种子为固定值

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count if self.count != 0 else 0

def main(**kwargs):
	global best_loss, num_joints
	# Step 0 : parse Options
	opt.parse(kwargs)
	# check checkpoint path
	opt.work_dir = os.path.join(opt.checkpoint_path, opt.dataset, opt.exp_id)
	if not os.path.exists(opt.work_dir):
		os.makedirs(opt.work_dir)
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
		# model = torch.nn.DataParallel(model).cuda()
		model = model.cuda()

	# Step 2 : loss function and optimizermAP
	lr = opt.lr
	if opt.with_logits:
		criterion = FocalLoss(num_joints[opt.dataset], opt.with_bg)
	else:
		criterion = AMSELoss(num_joints[opt.dataset], opt.with_bg).cuda()

	if opt.optim == 'sgd':
		optimizer = torch.optim.SGD(model.parameters(),
									lr = lr,
									momentum = opt.momentum,
									weight_decay = opt.weight_decay)
	elif opt.optim == 'rmsprop':
		optimizer = torch.optim.RMSprop(model.parameters(),
										lr = lr,
										momentum = opt.momentum,
										weight_decay = opt.weight_decay)
	elif opt.optim == 'adam':
		optimizer = torch.optim.Adam(model.parameters(),
									 lr = lr)

	loss = {'epoch':[], 'train':[], 'valid':[], 'APs':[]}

	# (Optional) resume from checkpoint
	prefix = '_'.join([opt.dataset, opt.model, opt.backbone])
	if opt.resume:
		model_path = os.path.join(opt.work_dir, opt.resume)
		loss_path = os.path.join(opt.work_dir, 'loss.t7')
		if os.path.exists(model_path):
			print("=> loading checkpoint '{}'".format(opt.resume))
			checkpoint = torch.load(model_path)
			opt.start_epoch = checkpoint['epoch']
			best_loss = checkpoint['best_loss']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])
			lr = optimizer.param_groups[0]['lr']
			# for param_group in optimizer.param_groups:
			#     param_group['lr'] = lr
			print("=> loaded checkpoint '{}' (epoch {})"
				  .format(opt.resume, checkpoint['epoch']))
			if os.path.exists(loss_path):
				loss = torch.load(loss_path)
		else:
			print("=> no checkpoint found at '{}'".format(opt.resume))

	cudnn.benchmark = True
	cudnn.deterministic = False
	cudnn.enabled = True
	print('    Total params: %.4fM' % (sum(p.numel() if p.requires_grad else 0 for p in model.parameters())/1000000.0))

	# learning rate scheduler
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90,120], gamma=0.1, last_epoch=opt.start_epoch-1)
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,70], gamma=0.1, last_epoch=opt.start_epoch-1)

	# Step 3 : data loading
	if opt.debug:
		opt.num_workers = 1
	Dataset = getattr(datasets, opt.dataset.upper())
	train_data = Dataset(opt, split='train')
	valid_data = Dataset(opt, split='val')
	train_loader = DataLoader(train_data, opt.batch_size, shuffle=True,
							  num_workers=opt.num_workers, pin_memory=True)
	valid_loader = DataLoader(valid_data, opt.test_batch_size, shuffle=False,
							  num_workers=opt.num_workers, pin_memory=True)

	# Step 4 : train and validate
	for epoch in range(opt.start_epoch, opt.max_epoch):
		scheduler.step()
		print('\nEpoch: %d/%d | LR: %.8f' %(epoch+1, opt.max_epoch, optimizer.param_groups[0]['lr']))
		train_loss = train(train_loader, model, criterion, optimizer, opt)
		valid_loss, APs = validate(valid_loader, valid_data, model, criterion, opt)

		valid_mAP = np.mean(APs)
		if opt.dataset == 'mpii':
			valid_mAP = np.mean(APs[:-1])

		# scheduler.step(train_loss)
		print('Train loss: %.6f | Test loss: %.6f | mAP: %.6f'%(train_loss, valid_loss, valid_mAP))
		# save train and valid loss every epoch
		loss['epoch'].append(epoch+1)
		loss['train'].append(train_loss)
		loss['valid'].append(valid_loss)
		loss['APs'].append(APs)
		torch.save(loss, os.path.join(opt.work_dir, 'loss.t7'))

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
			filename = os.path.join(opt.work_dir, filename)
			torch.save(checkpoint, filename)
		if (epoch+1) % opt.save_every == 0:
			checkpoint = {
				'epoch': epoch + 1,
				'model': opt.model,
				'state_dict': model.state_dict(),
				'best_loss': best_loss,
				'optimizer' : optimizer.state_dict() }
			filename = '_'.join([opt.model, opt.backbone, str(epoch+1)])+'.pth'
			filename = os.path.join(opt.work_dir, filename)
			torch.save(checkpoint, filename)

def train(train_loader, model, criterion, optimizer, opt):
	pbar = tqdm(total=len(train_loader))
	losses = AverageMeter()
	acc = AverageMeter()

	# switch to train mode
	model.train()

	for i, (data, heatmaps) in enumerate(train_loader):
		if opt.dataset == 'coco' and opt.with_mask:
			mask = data[:,-1].numpy().transpose((1,2,0))
			data = data[:,:-1]
			mask = cv2.resize(mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
			mask = torch.from_numpy(mask.transpose((2,0,1))).contiguous()
			mask = mask.cuda(non_blocking=True)
		else:
			mask = None

		inputs = data.cuda()
		target_hm = heatmaps.cuda(non_blocking=True)
		# zero gradient
		optimizer.zero_grad()
		# compute output
		output = model(inputs)
		# compute loss
		loss = criterion(output, target_hm, mask)

		loss.backward()
		optimizer.step()
		losses.update(loss.item(), inputs.size(0))
		_, avg_acc, cnt, pred = accuracy(output.detach(), target_hm.detach())
		acc.update(avg_acc, cnt)

		pbar.set_description("Training")
		pbar.set_postfix(Loss=losses.val, Loss_AVG=losses.avg, Acc=acc.val, Acc_AVG=acc.avg)
		pbar.update(1)
	pbar.close()
	return losses.avg

def validate(valid_loader, valid_data, model, criterion, opt):
	global num_joints
	losses = AverageMeter()
	acc = AverageMeter()
	APs = []
	# set evaluation model
	model.eval()
	num_samples = len(valid_data)
	all_preds = np.zeros((num_samples, num_joints[opt.dataset], 3), dtype=np.float32)
	all_boxes = np.zeros((num_samples, 6))
	metas = {'image':[], 'area':[], 'score':[]}
	idx = 0
	pbar = tqdm(total=len(valid_loader))
	predictions = []
	annos = []
	ref_scale = []
	#nan_cnt = 0
	with torch.no_grad():
		for i, (data, heatmaps, meta) in enumerate(valid_loader):
			if opt.dataset == 'coco' and opt.with_mask:
				mask = data[:,-1].numpy().transpose((1,2,0))
				data = data[:,:-1]
				mask = cv2.resize(mask, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
				mask = torch.from_numpy(mask.transpose((2,0,1))).contiguous()
				mask = mask.cuda(non_blocking=True)
			else:
				mask = None
			inputs = data.cuda()
			target_hm = heatmaps.cuda(non_blocking=True)

			# compute output
			output = model(inputs)
			# compute loss
			loss = criterion(output, target_hm, mask)

			batch_size = inputs.size(0)
			losses.update(loss.item(), batch_size)
			_, avg_acc, cnt, pred = accuracy(output, target_hm)
			acc.update(avg_acc, cnt)

			# compute actual ACC
			c = meta['center'].numpy()
			s = meta['scale'].numpy()
			preds, scores = final_preds(output, c, s, opt)

			all_preds[idx:idx+batch_size] = np.concatenate((preds, scores), axis=2)
			metas['area'].extend(meta['area'].numpy().tolist())
			metas['score'].extend(meta['score'].numpy().tolist())
			metas['image'].extend(meta['image'].numpy().tolist())

			idx += batch_size

			out_joint = output
			if opt.with_bg:
				out_joint = out_joint[:,:-1]
			if opt.with_logits:
				prediction = nms_heatmap(out_joint.sigmoid(), opt.nms_threshold, opt.nms_window_size)
			else:
				prediction = nms_heatmap(out_joint, opt.nms_threshold, opt.nms_window_size)
			# [batch(num_person), num_joints, 3] (x, y, score)
			predictions.append(prediction)
			annos.append(meta['joints'].numpy())
			ref_scale.append(meta['ref_scale'].numpy())

			pbar.set_description("Testing")
			pbar.set_postfix(Loss=losses.val, Loss_AVG=losses.avg, Acc=acc.val, Acc_AVG=acc.avg)
			pbar.update(1)
		pbar.close()
		# set training mode
		model.train()

		if opt.dataset == 'mpii':
			mAP = eval_AP(predictions, annos, ref_scale, 0.5)
		elif opt.dataset == 'coco':
			delta = 2 * np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
			mAP = eval_mAP(predictions, annos, ref_scale, delta)

			results, ap = valid_data.evaluate(all_preds, metas, opt.work_dir)

	return losses.avg, mAP


if __name__ == '__main__':
	main()
