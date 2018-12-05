# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *


class bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, ptype='preact', bias=False):
        super(bottleneck, self).__init__()
        planes = outplanes // 2
        if inplanes != outplanes or stride != 1:
            self.shortcut = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=bias)
        else:
            self.shortcut = None
        self.ptype = ptype
        if ptype != 'no_preact':
            self.preact = nn.Sequential(
                            nn.BatchNorm2d(inplanes),
                            nn.ReLU(inplace=True) )
        branch = []
        self.branch = nn.Sequential(
                        nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(planes),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias),
                        nn.BatchNorm2d(planes),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(planes, outplanes, kernel_size=1, bias=bias) )

    def forward(self, x):
        if self.ptype == 'both_preact':
            x = self.preact(x)
        y = x
        if self.ptype != 'no_preact' and self.ptype != 'both_preact':
            x = self.preact(x)

        out = self.branch(x)
        if self.shortcut is not None:
            y = self.shortcut(y)
        out += y

        return out

class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth, bias):
        super(Hourglass, self).__init__()
        self.depth = depth
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hourglass(block, num_blocks, planes, depth, bias)

    def _make_residual(self, block, num_blocks, inplanes, outplanes=None, bias=False):
        if outplanes == None:
            outplanes = inplanes
        layers = []
        layers.append(block(inplanes, outplanes, bias=bias))
        for i in range(1, num_blocks):
            layers.append(block(outplanes, outplanes, bias=bias))
        return nn.Sequential(*layers)

    def _make_hourglass(self, block, num_blocks, planes, depth, bias):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes, bias=bias))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes, bias=bias))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) original feature map.
          y: (Variable) feature map to be upsampled.
        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = x.size()
        return F.upsample(y, size=(H,W), mode='bilinear', align_corners=True) + x
	#return nn.Upsample(size=(H,W), mode='bilinear')(y) + x

    def _upsample_concat(self, x, y):
        '''Upsample and concatenate two feature maps.
        Args:
          x: (Variable) original feature map.
          y: (Variable) feature map to be upsampled.
        Returns:
          (Variable) concatenated feature map.
        '''
        _,_,H,W = x.size()
        out = F.upsample(y, size=(H,W), mode='bilinear', align_corners=True)
        out = torch.cat((out, x), dim=1)
        return out

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = self.maxpool(x)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hourglass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        out = self._upsample_add(up1, low3)
        return out

    def forward(self, x):
        return self._hourglass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_classes, num_stacks, num_blocks, depth=4):
        super(HourglassNet, self).__init__()

        bias = True
        num_feats = 256
        self.num_stacks = num_stacks

        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=bias),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True) )
        # self.pre = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.res1 = self._make_residual(block, 64, 256, 1, 1, 'first', bias=bias)
        self.res2 = self._make_residual(block, 256, 256, 1, bias=bias)
        self.res3 = self._make_residual(block, 256, num_feats, 1, bias=bias)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        hg, res, proj, proj_, heatmap_ = [], [], [], [], []
        heatmap = []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, num_feats, depth, bias=bias))
            res.append(self._make_residual(block, num_feats, num_feats, num_blocks, bias=True))
            # res.append(self._make_residual(block, 2*num_feats, num_feats, num_blocks, bias=True))
            proj.append(self._make_projection(num_feats, num_feats))
            heatmap.append(nn.Conv2d(num_feats, num_classes, kernel_size=1))
            # heatmap.append(self._make_sigmoid(num_feats, num_classes))
            if i < num_stacks-1:
                proj_.append(nn.Conv2d(num_feats, num_feats, kernel_size=1))
                heatmap_.append(nn.Conv2d(num_classes, num_feats, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.proj = nn.ModuleList(proj)
        self.heatmap = nn.ModuleList(heatmap)
        self.proj_ = nn.ModuleList(proj_)
        self.heatmap_ = nn.ModuleList(heatmap_)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_residual(self, block, inplanes, outplanes, num_blocks, stride=1, ptype=None, bias=False):
        preact = 'no_preact' if ptype == 'first' else 'preact'
        layers = []
        layers.append(block(inplanes, outplanes, stride, preact, bias=bias))
        for i in range(1, num_blocks):
            layers.append(block(outplanes, outplanes, bias=bias))

        return nn.Sequential(*layers)

    def _make_projection(self, inplanes, outplanes):
        return nn.Sequential(
                # nn.BatchNorm2d(inplanes),
                # nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, outplanes, kernel_size=1),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True) )

    def _make_sigmoid(self, inplanes, outplanes):
        return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1),
                nn.Sigmoid() )

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        out = []
        x = self.pre(x)
        x = self.res1(x)
        x = self.maxpool(x)
        x = self.res2(x)
        x = self.res3(x)
        # z = x

        for i in range(self.num_stacks):
            # y = self.hg[i](z)
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.proj[i](y)
            heatmap = self.heatmap[i](y)
            out.append(heatmap)
            if i < self.num_stacks-1:
                proj_ = self.proj_[i](y)
                heatmap_ = self.heatmap_[i](heatmap)
                # z = x + proj_ + heatmap_
                x = x + proj_ + heatmap_

        return out

def hg(num_classes, num_stacks):
    num_blocks = 8 // num_stacks
    model = HourglassNet(bottleneck, num_classes, num_stacks, num_blocks)

    return model
