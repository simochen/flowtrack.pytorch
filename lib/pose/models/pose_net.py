import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet, resnet_dict
from .densenet import DenseNet, densenet_dict

def _make_deconv(inplanes, outplanes, kernel_size, stride, padding, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias) )

# def _make_deconv(inplanes, outplanes, kernel_size, stride, padding, bias=True):
#     return nn.Sequential(
#             nn.ConvTranspose2d(inplanes, outplanes, kernel_size=kernel_size,
#                                stride=stride, padding=padding, bias=bias),
#             nn.BatchNorm2d(outplanes),
#             nn.ReLU(inplace=True) )

def _make_conv(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias) )

class PoseResnet(ResNet):
    def __init__(self, block, layers, num_classes):
        super(PoseResnet, self).__init__(block, layers)
        num_feats = 256
        self.bias = False

        # Lateral layers
        # self.latlayer1 = nn.Conv2d(2048, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.latlayer2 = nn.Conv2d(1024, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.latlayer3 = nn.Conv2d( 512, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)
        self.latlayer4 = nn.Conv2d( 256, num_feats, kernel_size=1, stride=1, padding=0, bias=self.bias)

        # Top-down layers
        self.deconv1 = _make_deconv(2048, num_feats, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv2 = _make_deconv(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv3 = _make_deconv(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.bias)

        # self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)
        self.heatmap = _make_conv(num_feats, num_classes, kernel_size=1)

    def forward(self, x):
        # Bottom-up
        c1 = self.conv1(x)
        c1 = self.bn1(c1)
        c1 = self.relu(c1)      # stride: 2; RF: 7
        c1 = self.maxpool(c1)   # stride: 4; RF:11

        c2 = self.layer1(c1)    # stride: 4; RF:35
        c3 = self.layer2(c2)    # stride: 8; RF:91
        c4 = self.layer3(c3)    # stride:16: RF:267
        c5 = self.layer4(c4)    # stride:32; RF:427
        # Top-down
        #p5 = self.latlayer1(c5) # stride:32; RF:427
        p4 = self.deconv1(c5) + self.latlayer2(c4)
        p3 = self.deconv2(p4) + self.latlayer3(c3)
        p2 = self.deconv3(p3) + self.latlayer4(c2)
        out = self.heatmap(p2)
        return out

    def init_net(self, pretrained=''):
        self.init_weights(pretrained)

        for name, m in self.named_modules():
            if 'lat' in name:
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    if self.bias:
                        nn.init.constant_(m.bias, 0)
            elif 'deconv' in name:
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for m in self.heatmap.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class PoseDensenet(DenseNet):
    def __init__(self, num_init_features, growth_rate, block_config, num_classes):
        super(PoseDensenet, self).__init__(num_init_features, growth_rate, block_config)
        num_feats = 256
        self.bias = False

        # Lateral layers
        # self.latlayer1 = nn.Conv2d(self.planes[3], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer2 = nn.Conv2d(self.planes[2], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer3 = nn.Conv2d(self.planes[1], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer4 = nn.Conv2d(self.planes[0], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)

        # Top-down layers
        self.deconv1 = _make_deconv(self.planes[3], num_feats, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv2 = _make_deconv(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.bias)
        self.deconv3 = _make_deconv(num_feats, num_feats, kernel_size=4, stride=2, padding=1, bias=self.bias)

        self.heatmap = nn.Conv2d(num_feats, num_classes, kernel_size=1)

    def forward(self, x):
        # Bottom-up
        c1 = self.conv0(x)
        c1 = self.norm0(c1)
        c1 = self.relu0(c1)
        c1 = self.pool0(c1)

        c2 = self.denseblock1(c1)
        c3 = self.transition1(c2)
        c3 = self.denseblock2(c3)
        c4 = self.transition2(c3)
        c4 = self.denseblock3(c4)
        c5 = self.transition3(c4)
        c5 = self.denseblock4(c5)
        c5 = self.norm5(c5)
        c5 = F.relu(c5, inplace=True)


        # Top-down
        #p5 = self.latlayer1(c5) # stride:32; RF:427
        p4 = self.deconv1(c5) + self.latlayer2(c4)
        p3 = self.deconv2(p4) + self.latlayer3(c3)
        p2 = self.deconv3(p3) + self.latlayer4(c2)
        out = self.heatmap(p2)
        return out

    def init_net(self, pretrained=''):
        self.init_weights(pretrained)

        for name, m in self.named_modules():
            if 'lat' in name:
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    if self.bias:
                        nn.init.constant_(m.bias, 0)
            elif 'deconv' in name:
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for m in self.heatmap.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)


net_list = ['resnet', 'densenet']
def pose(backbone, num_classes, pretrained):
    for net in net_list:
        if net in backbone:
            break
    num_layers = int(backbone[len(net):])
    args = eval('{}_dict'.format(net))[num_layers]
    model = eval('Pose{}'.format(net.capitalize()))(*args, num_classes=num_classes)

    if pretrained:
        model_path = os.path.join('data', 'pretrained', '{}.pth'.format(backbone))
    else:
        model_path = ''
    model.init_net(model_path)
    return model
