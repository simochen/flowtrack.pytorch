from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo

from .blocks import *


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#         'resnet152']


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                    bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class deconv_resnet(nn.Module):
    def __init__(self, num_layers=50, num_classes=17, pretrained=False):
        super(deconv_resnet, self).__init__()
        num_feats = 256
        bias = False

        if num_layers == 50:
            self.resnet = resnet50()
            # self.resnet = resnet50(pretrained)
            model_path = 'data/pretrained/resnet50-caffe.pth'
        elif num_layers == 101:
            self.resnet = resnet101()
            # self.resnet = resnet101(pretrained)
            model_path = 'data/pretrained/resnet101-caffe.pth'
        elif num_layers == 152:
            self.resnet = resnet152()
            # self.resnet = resnet152(pretrained)
            model_path = 'data/pretrained/resnet152-caffe.pth'

        if pretrained:
            print("Loading pretrained weights from %s" %(model_path))
            state_dict = torch.load(model_path)
            self.resnet.load_state_dict({k:v for k,v in state_dict.items() if k in self.resnet.state_dict()})

        self.backbone = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool,
                                      self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4)

        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(2048, num_feats, kernel_size=4, stride=2, padding=1, bias = bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True) )

        self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)

        for m in self.deconv:
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # bound = math.sqrt(6. / n)
                # m.weight.data.uniform_(-bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        def _init_weights(m):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            # bound = math.sqrt(6. / n)
            # m.weight.data.uniform_(-bound, bound)

        _init_weights(self.heatmap)


        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad=False

        
        # self.resnet.apply(set_bn_fix)

    def forward(self, x):
        x = self.backbone(x)
        x = self.deconv(x)
        out = self.heatmap(x)

        return out

    # def train(self, mode=True):
    #     # Override train so that the training mode is set as we want
    #     nn.Module.train(self, mode)
    #     if mode:
    #         # Set fixed blocks to be in eval mode (not really doing anything)
    #         self.resnet.eval()
    #         if self.FIXED_BLOCKS <= 3:
    #             self.resnet.layer4.train()
    #         if self.FIXED_BLOCKS <= 2:
    #             self.resnet.layer3.train()
    #         if self.FIXED_BLOCKS <= 1:
    #             self.resnet.layer2.train()
    #         if self.FIXED_BLOCKS == 0:
    #             self.resnet.layer1.train()

    #     # Set batchnorm always in eval mode during training
    #     def set_bn_eval(m):
    #         classname = m.__class__.__name__
    #         if classname.find('BatchNorm') != -1:
    #         m.eval()

    #     self.resnet.apply(set_bn_eval)

    def load_pretrained_resnet(self, state_dict):
        self.resnet.load_state_dict({k: state_dict[k] for k in list(self.resnet.state_dict())})