from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMSELoss(nn.Module):

    def __init__(self, num_joints, with_bg):
        super(AMSELoss, self).__init__()
        self.num_joints = num_joints
        self.with_bg = with_bg

    def amse_loss(self, x, t, alpha, mask=None):
        '''MSELoss with alpha
        Args:
            x: (tensor) sized [N, *].
            t: (tensor) sized [N, *].
        Return:
            (tensor) amse loss.
        '''
        pos_mask = t.gt(0).float()
        w = alpha * pos_mask + (1-alpha)*(1-pos_mask)
        if mask is not None:
            mask = mask.gt(0).float()
            if w.dim() > mask.dim():
                newsize = list(mask.size())
                newsize.insert(1,1)
                mask = mask.view(newsize).expand_as(w)
            w = w * mask
        loss = w * (x - t).pow(2)
        return loss.sum()


    def amse_loss_with_logits(self, x, t, alpha, mask=None):
        '''MSE loss with alpha.

        Args:
          x: (tensor) sized [N,*].
          t: (tensor) sized [N,*].

        Return:
          (tensor) amse loss.
        '''

        p = x.sigmoid()
        pos_mask = t.gt(0).float()
        w = alpha * pos_mask + (1-alpha)*(1-pos_mask)
        if mask is not None:
            mask = mask.gt(0).float()
            if w.dim() > mask.dim():
                newsize = list(mask.size())
                newsize.insert(1,1)
                mask = mask.view(newsize).expand_as(w)
            w = w * mask
        loss = w * (p - t).pow(2)
        return loss.sum()


    # def forward(self, inputs, targets, mask=None):
    #     alpha = [0.5, 0.5]
    #
    #     b, c, h, w = targets.size()
    #     weights = targets.sum(dim=-1).sum(dim=-1).gt(0).float()
    #     weights = weights.view(b, c, 1, 1)
    #
    #     joint_out = inputs
    #     joint_hm = targets
    #
    #     if self.with_bg:
    #         bg_out = inputs[:,-1]
    #         joint_out = joint_out[:,:-1]
    #         bg_hm = targets[:,-1]
    #         joint_hm = joint_hm[:,:-1]
    #         weights = weights[:,:-1]
    #
    #     if mask is None:
    #         mask_ = torch.ones((b,1,h,w)).cuda()
    #     else:
    #         mask = mask.gt(0.1).float()
    #         mask_ = mask.view(b, 1, h, w)
    #
    #     weights = weights * mask_
    #     fore_size = weights.sum()
    #
    #     joint_ch = joint_out.size(1)    # C_j
    #     joint_loss = self.amse_loss(joint_out, joint_hm, alpha[0], weights)
    #     if self.with_bg:
    #         bg_loss = self.amse_loss(bg_out, bg_hm, alpha[1], mask)
    #         loss = (joint_loss + bg_loss) / (fore_size + mask_.sum())
    #     else:
    #         loss = joint_loss / fore_size
    #
    #     # print('loss: %.3f'%loss.data[0])
    #     return loss


    def forward(self, inputs, targets, weights, mask=None):
        alpha = [0.5, 0.5]

        b, c, h, w = targets.size()
        # weights = targets.sum(dim=-1).sum(dim=-1).gt(0).float()
        weights = weights.view(b, c, 1, 1)

        joint_out = inputs
        joint_hm = targets

        if self.with_bg:
            bg_out = inputs[:,-1]
            joint_out = joint_out[:,:-1]
            bg_hm = targets[:,-1]
            joint_hm = joint_hm[:,:-1]
            weights = weights[:,:-1]

        if mask is None:
            avg_size = joint_out[:,0].numel()
        else:
            avg_size = mask.gt(0.1).float().sum()
            mask = mask.view(b, 1, h, w)
            weights = weights * mask

        joint_ch = joint_out.size(1)    # C_j
        joint_loss = self.amse_loss(joint_out, joint_hm, alpha[0], weights)
        if self.with_bg:
            bg_loss = self.amse_loss(bg_out, bg_hm, alpha[1], mask)
            loss = (joint_loss + bg_loss) / avg_size / (joint_ch + 1)
        else:
            loss = joint_loss / avg_size / joint_ch

        # print('loss: %.3f'%loss.data[0])
        return loss
