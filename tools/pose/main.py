# -*- coding:utf-8 -*-
from __future__ import print_function, absolute_import

import _init_paths
from config import opt
import os
import torch
import cv2
import json
import shutil
import models
import datasets
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils import *

best_loss = 1
num_joints = {'mpii':16, 'aic':14, 'coco':17}
# torch.manual_seed(1)    # 设置随机数种子为固定值

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

	# cudnn related setting
	cudnn.benchmark = True
	cudnn.deterministic = False
	cudnn.enabled = True

	# dataset parse
	num_classes = num_joints[opt.dataset]
	if opt.with_bg:
		num_classes += 1

	# Step 1 : create model
	print("==> creating model '{}', backbone = {}".format(
		opt.model, opt.backbone) )
	if opt.model in ['deconv', 'fpn', 'pose']:
		model = getattr(models, opt.model)(opt.backbone,
										   num_classes = num_classes,
										   pretrained = opt.pretrained)
		opt.model_name = '{}_{}'.format(opt.model, opt.backbone)
	elif opt.model == 'hg':
		model = getattr(models, opt.model)(num_classes = num_classes,
										   num_stacks = 8)
		opt.model_name = opt.model

	# tensorboard writer
	writer_dict = None
	if opt.tensorboard and 'train' in opt.run_type:
		log_dir = os.path.join(opt.work_dir, 'log', opt.model_name)
		if os.path.exists(log_dir):
			shutil.rmtree(log_dir)
		writer_dict = {
			'writer': SummaryWriter(log_dir=log_dir),
			'train_global_steps': 0,
			'valid_global_steps': 0,
		}
		dump_input = torch.rand((opt.batch_size,
								 3,
								 opt.input_res[0],
								 opt.input_res[1]))
		writer_dict['writer'].add_graph(model, (dump_input, ), verbose=False)

	if opt.use_gpu:
		# model = torch.nn.DataParallel(model).cuda()
		model = model.cuda()

	# Step 2 : loss function and optimizer mAP
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
	if opt.resume:
		model_path = os.path.join(opt.work_dir, opt.resume)
		loss_path = os.path.join(opt.work_dir, '{}_loss.t7'.format(opt.model_name))
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
	if opt.run_type == 'trainval':
		for epoch in range(opt.start_epoch, opt.max_epoch):
			scheduler.step()
			print('\nEpoch: %d/%d | LR: %.8f' %(epoch+1, opt.max_epoch, optimizer.param_groups[0]['lr']))
			train_loss = train(train_loader, model, criterion, optimizer, opt, writer_dict)
			valid_loss, APs_dict, valid_mAP = validate(valid_loader, valid_data, model, criterion, opt, writer_dict)

			# scheduler.step(train_loss)
			print('Train loss: %.6f | Test loss: %.6f | mAP: %.6f'%(train_loss, valid_loss, valid_mAP))
			# save train and valid loss every epoch
			loss['epoch'].append(epoch+1)
			loss['train'].append(train_loss)
			loss['valid'].append(valid_loss)
			loss['APs'].append(APs_dict)
			torch.save(loss, os.path.join(opt.work_dir, '{}_loss.t7'.format(opt.model_name)))

			# save checkpoint (best valid loss)
			if valid_loss < best_loss:
				best_loss = valid_loss
				checkpoint = {
					'epoch': epoch + 1,
					'model': opt.model,
					'state_dict': model.state_dict(),
					'best_loss': best_loss,
					'optimizer' : optimizer.state_dict() }
				filename = '_'.join([opt.model_name, 'best'])+'.pth'
				filename = os.path.join(opt.work_dir, filename)
				torch.save(checkpoint, filename)
			# save every i epoch
			if (epoch+1) % opt.save_every == 0:
				checkpoint = {
					'epoch': epoch + 1,
					'model': opt.model,
					'state_dict': model.state_dict(),
					'best_loss': best_loss,
					'optimizer' : optimizer.state_dict() }
				filename = '_'.join([opt.model_name, str(epoch+1)])+'.pth'
				filename = os.path.join(opt.work_dir, filename)
				torch.save(checkpoint, filename)
	elif opt.run_type == 'valid':
		valid_loss, APs_dict, valid_mAP = validate(valid_loader, valid_data, model, criterion, opt, writer_dict)
		print('Test loss: %.6f | mAP: %.6f'%(valid_loss, valid_mAP))


def train(train_loader, model, criterion, optimizer, opt, writer_dict=None):
	pbar = tqdm(total=len(train_loader))
	losses = AverageMeter()
	acc = AverageMeter()

	# switch to train mode
	model.train()

	for i, (data, heatmaps, target_weight) in enumerate(train_loader):
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
		target_weight = target_weight.cuda(non_blocking=True)
		# zero gradient
		optimizer.zero_grad()
		# compute output
		output = model(inputs)
		# compute loss
		if opt.model == 'hg':
			loss = torch.zeros(1).cuda()
			for out in output:
				loss.add_(criterion(out, target_hm, target_weight, mask))
			output = output[-1]
		else:
			loss = criterion(output, target_hm, target_weight, mask)

		loss.backward()
		optimizer.step()
		losses.update(loss.item(), inputs.size(0))
		_, avg_acc, cnt, pred = accuracy(output.detach(), target_hm.detach())
		acc.update(avg_acc, cnt)

		if writer_dict and i % opt.draw_freq == 0:
			writer = writer_dict['writer']
			global_steps = writer_dict['train_global_steps']
			writer.add_scalar('train_loss', losses.val, global_steps)
			writer.add_scalar('train_acc', acc.val, global_steps)
			writer_dict['train_global_steps'] = global_steps + 1

		pbar.set_description("Training")
		pbar.set_postfix(Loss=losses.val, Loss_AVG=losses.avg, Acc=acc.val, Acc_AVG=acc.avg)
		pbar.update(1)
	pbar.close()
	return losses.avg

def validate(valid_loader, valid_data, model, criterion, opt, writer_dict=None):
	global num_joints
	losses = AverageMeter()
	acc = AverageMeter()
	# set evaluation model
	model.eval()
	num_samples = len(valid_data)
	all_preds = np.zeros((num_samples, num_joints[opt.dataset], 3), dtype=np.float32)
	all_boxes = np.zeros((num_samples, 6))
	metas = {'image':[], 'area':[], 'score':[]}
	idx = 0
	pbar = tqdm(total=len(valid_loader))
	# predictions = []
	annos = []
	ref_scale = []
	#nan_cnt = 0
	with torch.no_grad():
		for i, (data, heatmaps, target_weight, meta) in enumerate(valid_loader):
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
			target_weight = target_weight.cuda(non_blocking=True)

			# compute output
			output = model(inputs)
			if opt.model == 'hg':
				output = output[-1]

			if opt.flip_test:
				# inputs: [B, C, H, W]
				input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
				input_flipped = torch.from_numpy(input_flipped).cuda()
				output_flipped = model(input_flipped)
				if opt.model == 'hg':
					output_flipped = output_flipped[-1]
				output_flipped = swaplr_image(output_flipped.cpu().numpy(), opt.dataset)
				output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

				output = (output + output_flipped) * 0.5

			# compute loss
			loss = criterion(output, target_hm, target_weight, mask)

			batch_size = inputs.size(0)
			losses.update(loss.item(), batch_size)
			_, avg_acc, cnt, pred = accuracy(output, target_hm)
			acc.update(avg_acc, cnt)

			# compute actual ACC
			c = meta['center'].numpy()
			s = meta['scale'].numpy()

			if opt.with_bg:
				output = output[:,:-1]

			preds, scores = final_preds(output, c, s, opt.adjust_coords)

			all_preds[idx:idx+batch_size] = np.concatenate((preds, scores), axis=2)
			if opt.dataset == 'coco':
				metas['area'].extend(meta['area'].numpy().tolist())
				metas['score'].extend(meta['score'].numpy().tolist())
				metas['image'].extend(meta['image'].numpy().tolist())
			elif opt.dataset == 'mpii':
				annos.append(meta['joints'].numpy())
				ref_scale.append(meta['ref_scale'].numpy())

			idx += batch_size

			# out_joint = output
			# if opt.with_logits:
			# 	prediction = nms_heatmap(out_joint.sigmoid(), opt.nms_threshold, opt.nms_window_size)
			# else:
			# 	prediction = nms_heatmap(out_joint, opt.nms_threshold, opt.nms_window_size)
			# [batch(num_person), num_joints, 3] (x, y, score)
			# predictions.append(prediction)

			pbar.set_description("Testing")
			pbar.set_postfix(Loss=losses.val, Loss_AVG=losses.avg, Acc=acc.val, Acc_AVG=acc.avg)
			pbar.update(1)
		pbar.close()
		# set training mode
		model.train()

		if opt.dataset == 'mpii':
			# mAP = eval_AP(all_preds, annos, ref_scale, 0.5)
			metas['joints'] = np.concatenate(annos, axis=0)
			metas['ref_scale'] = np.concatenate(ref_scale, axis=0)
			results, ap = valid_data.evaluate(all_preds, metas, opt.work_dir)
		elif opt.dataset == 'coco':
			# delta = 2 * np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0
			# mAP = eval_mAP(predictions, annos, ref_scale, delta)
			results, ap = valid_data.evaluate(all_preds, metas, opt.work_dir)

		if isinstance(results, list):
			for item in results:
				_print_name_value(item, opt.model_name)
		else:
			_print_name_value(results, opt.model_name)

		if writer_dict:
			writer = writer_dict['writer']
			global_steps = writer_dict['valid_global_steps']
			writer.add_scalar('valid_loss', losses.avg, global_steps)
			writer.add_scalar('valid_acc', acc.avg, global_steps)
			if isinstance(results, list):
				for item in results:
					writer.add_scalars('valid', dict(item), global_steps)
			else:
				writer.add_scalars('valid', dict(results), global_steps)
			writer_dict['valid_global_steps'] = global_steps + 1

	return losses.avg, dict(results), ap

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    print(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    print('|-----' * (num_values+1) + '|')
    print(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )

if __name__ == '__main__':
	main()
