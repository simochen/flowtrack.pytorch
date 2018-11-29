from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

from .blocks import *

BN_MOMENTUM = 0.1

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#         'resnet152']


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

class PoseResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                    bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.deconv_bias = False
        num_feats = 256

        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(2048, num_feats, kernel_size=4, stride=2, padding=1, bias = self.deconv_bias),
                    nn.BatchNorm2d(num_feats, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.deconv_bias),
                    nn.BatchNorm2d(num_feats, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.deconv_bias),
                    nn.BatchNorm2d(num_feats, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True) )

        self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv(x)
        x = self.heatmap(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            print('=> loading pretrained model {}'.format(pretrained))
            pretrained_state_dict = torch.load(pretrained)
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        for m in self.deconv.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.heatmap.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


resnet_dict = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def deconv_resnet(num_layers=50, num_classes=17, pretrained=True):
    block, layers = resnet_dict[num_layers]

    model = PoseResNet(block, layers, num_classes)

    if pretrained:
        model_path = os.path.join('data', 'pretrained', 'resnet{}.pth'.format(num_layers))
        model.init_weights(model_path)

    return model
