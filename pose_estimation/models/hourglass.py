# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import *

affine = False

class bottleneck(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, ptype='preact', useConv=False, bias=False):
        super(bottleneck, self).__init__()
        planes = outplanes / 2
        if useConv or inplanes != outplanes or stride != 1:
            self.shortcut = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=bias)
        else:
            self.shortcut = None
        self.ptype = ptype
        if ptype != 'no_preact':
            self.preact = nn.Sequential(
                            nn.BatchNorm2d(inplanes, affine=affine),
                            nn.ReLU(inplace=True) )
        branch = []
        self.branch = nn.Sequential(
                        nn.Conv2d(inplanes, planes, kernel_size=1, bias=bias),
                        nn.BatchNorm2d(planes, affine=affine),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias),
                        nn.BatchNorm2d(planes, affine=affine),
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
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hourglass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, inplanes, outplanes=None):
        if outplanes == None:
            outplanes = inplanes
        layers = []
        layers.append(block(inplanes, outplanes))
        for i in range(1, num_blocks):
            layers.append(block(outplanes, outplanes))
        return nn.Sequential(*layers)

    def _make_hourglass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            # for j in range(2):
            #     res.append(self._make_residual(block, num_blocks, planes))
            # if i == 0:
            #     res.append(self._make_residual(block, num_blocks, planes))
            #     res.append(self._make_residual(block, num_blocks, planes))
            # else:
            #     res.append(self._make_residual(block, num_blocks, 2*planes, planes))
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _add(self, x, y):
        '''Add two feature maps.

        Args:
          x: (Variable) original feature map.
          y: (Variable) upsampled feature map.

        Returns:
          (Variable) added feature map.

        Upsampled feature map size is always >= original feature map size.
        The reason why the two feature map sizes may not equal is because when the
        input size is odd, the upsampled feature map size is always 1 pixel
        bigger than the original input size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        '''
        _,_,H,W = x.size()
        return y[:,:,:H,:W] + x

    def _concatenate(self, x, y):
        '''Concatenate two feature maps.

        Args:
          x: (Variable) original feature map.
          y: (Variable) upsampled feature map.

        Returns:
          (Variable) concatenated feature map.

        Upsampled feature map size is always >= original feature map size.
        The reason why the two feature map sizes may not equal is because when the
        input size is odd, the upsampled feature map size is always 1 pixel
        bigger than the original input size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        '''
        _,_,H,W = x.size()
        return torch.cat((y[:,:,:H,:W], x), dim=1)

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
        return F.upsample(y, size=(H,W), mode='bilinear') + x

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = self.maxpool(x)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hourglass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        # up2 = self.upsample(low3)
        # out = self._add(up1, up2)
        # out = self._concatenate(up1, up2)
        out = self._upsample_add(up1, low3)
        return out

    def forward(self, x):
        return self._hourglass_forward(self.depth, x)


class hg_1branch(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_classes, num_feats, num_stacks, num_blocks, depth=4):
        super(hg_1branch, self).__init__()

        self.num_stacks = num_stacks

        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64, affine=affine),
                nn.ReLU(inplace=True) )
        # self.pre = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.res1 = self._make_residual(block, 64, 256, 1, 1, 'first')
        self.res2 = self._make_residual(block, 256, 512, 1)
        self.res3 = self._make_residual(block, 512, num_feats, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        hg, res, proj, proj_, heatmap_ = [], [], [], [], []
        heatmap = []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, num_feats, depth))
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
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # bound = math.sqrt(6. / n)
                # m.weight.data.uniform_(-bound, bound)
            elif affine == True and isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_residual(self, block, inplanes, outplanes, num_blocks, stride=1, ptype=None, bias=False):
        preact = 'no_preact' if ptype == 'first' else 'both_preact'
        layers = []
        layers.append(block(inplanes, outplanes, stride, preact, bias=bias))
        for i in range(1, num_blocks):
            layers.append(block(outplanes, outplanes, bias=bias))

        return nn.Sequential(*layers)

    def _make_projection(self, inplanes, outplanes):
        return nn.Sequential(
                nn.BatchNorm2d(inplanes, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, outplanes, kernel_size=1),
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
        x = self.maxpool(x)
        x = self.res1(x)
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


class hg_middle(nn.Module):
    def __init__(self, block, num_classes, num_feats, num_stacks=2, num_blocks=1, depth=4):
        super(hg_middle, self).__init__()

        self.num_stacks = num_stacks

        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64, affine=affine),
                nn.ReLU(inplace=True) )
        self.res1 = self._make_residual(block, 64, 128, 1, 1, 'first')
        self.res2 = self._make_residual(block, 128, 256, 1)
        self.res3 = self._make_residual(block, 256, num_feats, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        hg, res, proj, hm, proj_, hm_ = [], [], [], [], [], []
        for k in range(len(num_classes)):
            residual, projection, heatmap, projection_, heatmap_ = [], [], [], [], []
            for i in range(num_stacks):
                if k == 0:
                    hg.append(Hourglass(block, num_blocks, num_feats, depth))
                residual.append(self._make_residual(block, num_feats, num_feats, num_blocks))
                projection.append(self._make_projection(num_feats, num_feats))
                heatmap.append(nn.Conv2d(num_feats, num_classes[k], kernel_size=1))
                # heatmap.append(self._make_sigmoid(num_feats, num_classes[k]))
                if i < num_stacks-1:
                   projection_.append(nn.Conv2d(num_feats, num_feats, kernel_size=1))
                   heatmap_.append(nn.Conv2d(num_classes[k], num_feats, kernel_size=1))
            res.append(nn.ModuleList(residual))
            proj.append(nn.ModuleList(projection))
            hm.append(nn.ModuleList(heatmap))
            proj_.append(nn.ModuleList(projection_))
            hm_.append(nn.ModuleList(heatmap_))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.proj = nn.ModuleList(proj)
        self.heatmap = nn.ModuleList(hm)
        self.proj_ = nn.ModuleList(proj_)
        self.heatmap_ = nn.ModuleList(hm_)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # bound = math.sqrt(6. / n)
                # m.weight.data.uniform_(-bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_residual(self, block, inplanes, outplanes, num_blocks, stride=1, ptype=None, bias=False):
        preact = 'no_preact' if ptype == 'first' else 'both_preact'
        layers = []
        layers.append(block(inplanes, outplanes, stride, preact, bias=bias))
        for i in range(1, num_blocks):
            layers.append(block(outplanes, outplanes, bias=bias))

        return nn.Sequential(*layers)

    def _make_projection(self, inplanes, outplanes):
        return nn.Sequential(
                nn.BatchNorm2d(inplanes, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, outplanes, kernel_size=1),
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
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        # z = x

        for i in range(self.num_stacks):
            # y = self.hg[i](z)
            y = self.hg[i](x)
            y0 = self.res[0][i](y)
            y0 = self.proj[0][i](y0)
            heatmap0 = self.heatmap[0][i](y0)
            out.append(heatmap0)
            y1 = self.res[1][i](y)
            y1 = self.proj[1][i](y1)
            heatmap1 = self.heatmap[1][i](y1)
            out.append(heatmap1)
            if i < self.num_stacks-1:
                proj0_ = self.proj_[0][i](y0)
                heatmap0_ = self.heatmap_[0][i](heatmap0)
                proj1_ = self.proj_[1][i](y1)
                heatmap1_ = self.heatmap_[1][i](heatmap1)
                # z = x + proj0_ + heatmap0_ + proj1_ + heatmap1_
                x = x + proj0_ + heatmap0_ + proj1_ + heatmap1_

        return out

class hg_2branch(nn.Module):
    def __init__(self, block, num_classes, num_feats, num_stacks=2, num_blocks=1, depth=4):
        super(hg_2branch, self).__init__()

        self.num_stacks = num_stacks

        self.pre = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64, affine=affine),
                nn.ReLU(inplace=True) )
        self.res1 = self._make_residual(block, 64, 128, 1, 1, 'first')
        self.res2 = self._make_residual(block, 128, 256, 1)
        self.res3 = self._make_residual(block, 256, num_feats, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        hg, res, proj, hm, proj_, hm_ = [], [], [], [], [], []
        for k in range(len(num_classes)):
            hourglass, residual, projection, heatmap, projection_, heatmap_ = [], [], [], [], [], []
            for i in range(num_stacks):
                hourglass.append(Hourglass(block, num_blocks, num_feats, depth))
                residual.append(self._make_residual(block, num_feats, num_feats, num_blocks))
                projection.append(self._make_projection(num_feats, num_feats))
                heatmap.append(nn.Conv2d(num_feats, num_classes[k], kernel_size=1))
                # heatmap.append(self._make_sigmoid(num_feats, num_classes[k]))
                if i < num_stacks-1:
                    projection_.append(nn.Conv2d(num_feats, num_feats, kernel_size=1))
                    heatmap_.append(nn.Conv2d(num_classes[k], num_feats, kernel_size=1))
            hg.append(nn.ModuleList(hourglass))
            res.append(nn.ModuleList(residual))
            proj.append(nn.ModuleList(projection))
            hm.append(nn.ModuleList(heatmap))
            proj_.append(nn.ModuleList(projection_))
            hm_.append(nn.ModuleList(heatmap_))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.proj = nn.ModuleList(proj)
        self.heatmap = nn.ModuleList(hm)
        self.proj_ = nn.ModuleList(proj_)
        self.heatmap_ = nn.ModuleList(hm_)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # bound = math.sqrt(6. / n)
                # m.weight.data.uniform_(-bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_residual(self, block, inplanes, outplanes, num_blocks, stride=1, ptype=None, bias=False):
        preact = 'no_preact' if ptype == 'first' else 'both_preact'
        layers = []
        layers.append(block(inplanes, outplanes, stride, preact, bias=bias))
        for i in range(1, num_blocks):
            layers.append(block(outplanes, outplanes, bias=bias))

    def _make_projection(self, inplanes, outplanes):
        return nn.Sequential(
                nn.BatchNorm2d(inplanes, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, outplanes, kernel_size=1),
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
        x = self.maxpool(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        # z = x

        for i in range(self.num_stacks):
            # y0 = self.hg[0][i](z)
            y0 = self.hg[0][i](x)
            y0 = self.res[0][i](y0)
            y0 = self.proj[0][i](y0)
            heatmap0 = self.heatmap[0][i](y0)
            out.append(heatmap0)
            # y1 = self.hg[1][i](z)
            y1 = self.hg[1][i](x)
            y1 = self.res[1][i](y1)
            y1 = self.proj[1][i](y1)
            heatmap1 = self.heatmap[1][i](y1)
            out.append(heatmap1)
            if i < self.num_stacks-1:
                proj0_ = self.proj_[0][i](y0)
                heatmap0_ = self.heatmap_[0][i](heatmap0)
                proj1_ = self.proj_[1][i](y1)
                heatmap1_ = self.heatmap_[1][i](heatmap1)
                # z = x + proj0_ + heatmap0_ + proj1_ + heatmap1_
                x = x + proj0_ + heatmap0_ + proj1_ + heatmap1_

        return out
