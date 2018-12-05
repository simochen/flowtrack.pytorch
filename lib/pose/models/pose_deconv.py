from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ResNet, resnet_dict
from .densenet import DenseNet, densenet_dict

class DeconvResnet(ResNet):
    def __init__(self, block, layers, num_classes):
        super(DeconvResnet, self).__init__(block, layers)

        self.deconv_bias = False
        num_feats = 256

        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(2048, num_feats, kernel_size=4, stride=2, padding=1, bias = self.deconv_bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.deconv_bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.deconv_bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True) )

        self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)

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

    def init_net(self, pretrained=''):
        self.init_weights(pretrained)

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


class DeconvDensenet(DenseNet):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, num_init_features, growth_rate, block_config, num_classes):

        super(DeconvDensenet, self).__init__(num_init_features, growth_rate, block_config)

        # deconv layers
        self.deconv_bias = False
        num_feats = 256

        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(self.inplanes, num_feats, kernel_size=4, stride=2, padding=1, bias = self.deconv_bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.deconv_bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.deconv_bias),
                    nn.BatchNorm2d(num_feats),
                    nn.ReLU(inplace=True) )

        self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)

        x = self.denseblock1(x)
        x = self.tranition1(x)
        x = self.denseblock2(x)
        x = self.tranition2(x)
        x = self.denseblock3(x)
        x = self.tranition3(x)
        x = self.denseblock4(x)
        x = self.norm5(x)
        x = F.relu(x, inplace=True)

        x = self.deconv(x)
        x = self.heatmap(x)
        return x

    def init_net(self, pretrained=''):
        self.init_weights(pretrained)

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

# def deconv_resnet(layer_msg, num_classes, pretrained):
#     num_layers = int(layer_msg)
#     block, layers = resnet_dict[num_layers]
#
#     model = DeconvResnet(block, layers, num_classes)
#
#     if pretrained:
#         model_path = os.path.join('data', 'pretrained', 'resnet{}.pth'.format(num_layers))
#     else:
#         model_path = ''
#     model.init_net(model_path)
#
#     return model
#
# def deconv_densenet(layer_msg, num_classes, pretrained):
#     num_layers = int(layer_msg)
#     num_init_features, growth_rate, block_config = densenet_dict[num_layers]
#     model = DeconvDensenet(num_init_features, growth_rate, block_config, num_classes=num_classes)
#
#     if pretrained:
#         model_path = os.path.join('data', 'pretrained', 'densenet{}.pth'.format(num_layers))
#     else:
#         model_path = ''
#     model.init_net(model_path)
#
#     return model

net_list = ['resnet', 'densenet']
def deconv(backbone, num_classes, pretrained):
    for net in net_list:
        if net in backbone:
            break
    num_layers = int(backbone[len(net):])
    args = eval('{}_dict'.format(net))[num_layers]
    model = eval('Deconv{}'.format(net.capitalize()))(*args, num_classes=num_classes)

    if pretrained:
        model_path = os.path.join('data', 'pretrained', '{}.pth'.format(backbone))
    else:
        model_path = ''
    model.init_net(model_path)
    return model
