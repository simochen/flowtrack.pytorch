from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.autograd import Function


class FocalLoss(nn.Module):

    def __init__(self, num_joints, with_bg):
        super(FocalLoss, self).__init__()
        self.num_joints = num_joints
        self.with_bg = with_bg

    def focal_loss(self, x, t, alpha, gamma, mask=None):
        '''Focal loss.

        Args:
          x: (tensor) sized [N{,C},H,W].
          t: (tensor) sized [N{,C},H,W].
        Return:
          (tensor) focal loss.
        '''

        pt = x*t + (1-x)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        if gamma > 0:
            w = w * (1-pt).pow(gamma)
        if mask is not None:
            mask = mask.gt(0).float()
            if w.dim() > mask.dim():
                newsize = list(mask.size())
                newsize.insert(1,1)
                mask = mask.view(newsize).expand_as(w)
            w = w * mask
        # pt = t.neg().add_(1).mul(x.neg().add_(1)).add_(t.mul(x))
        # w = t.neg().add_(1).mul_(1-alpha).add_(t.mul(alpha))
        # w = pt.neg().add_(1).pow(gamma).mul(w)
        return F.binary_cross_entropy(x, t, w, size_average=False)

    def focal_loss_with_logits(self, x, t, alpha, gamma, mask=None):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,*].
          t: (tensor) sized [N,*].

        Return:
          (tensor) focal loss.
        '''
        # alpha = 0.25
        # gamma = 2

        alpha = Variable(torch.Tensor([alpha]))
        if x.is_cuda:
            alpha = alpha.cuda()
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        if gamma > 0:
            w = w * (1-pt).pow(gamma)
        if mask is not None:
            mask = mask.gt(0).float()
            if w.dim() > mask.dim():
                newsize = list(mask.size())
                newsize.insert(1,1)
                mask = mask.view(newsize).expand_as(w)
            w = w * mask
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, t, alpha, gamma=2, beta=1, mask=None):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,*].
          t: (tensor) sized [N,*].

        Return:
          (tensor) focal loss.
        '''

        # alpha = 0.25
        # gamma = 2
        # beta = 1

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (gamma*xt+beta).sigmoid()
        w = alpha*t + (1-alpha)*(1-t)
        if mask is not None:
            mask = mask.gt(0).float()
            if w.dim() > mask.dim():
                newsize = list(mask.size())
                newsize.insert(1,1)
                mask = mask.view(newsize).expand_as(w)
            w = w * mask
        loss = -w*pt.log() / gamma
        return loss.sum()


    def forward(self, inputs, targets, mask=None):
        alpha = [0.5, 0.5]
        gamma = [1, 1]

        targets = targets.gt(0.5).float()

        joint_out = inputs
        joint_hm = targets
        if self.with_bg:
            bg_out = inputs[:,-1]
            joint_out = joint_out[:,:-1]
            bg_hm = targets[:,-1]
            joint_hm = joint_hm[:,:-1]

        if mask is None:
            avg_size = joint_out[:,0].numel()
        else:
            avg_size = mask.gt(0.1).float().sum()
        joint_ch = joint_out.size(1)    # C_j
        joint_loss = self.focal_loss_with_logits(joint_out, joint_hm, alpha[0], gamma[0], mask)
        if self.with_bg:
            bg_loss = self.focal_loss_with_logits(bg_out, bg_hm, alpha[1], gamma[1], mask)
            loss = (joint_loss + bg_loss) / avg_size / (joint_ch + 1)
        else:
            loss = joint_loss / avg_size / joint_ch

        # print('loss: %.3f'%loss.data[0])
        return loss
