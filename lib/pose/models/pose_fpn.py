import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet, resnet_dict
from .densenet import DenseNet, densenet_dict

def _upsample_add(x, y):
    '''Upsample and add two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
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
    _,_,H,W = y.size()
    return F.upsample(x, size=(H,W), mode='bilinear', align_corners=True) + y

def _upsample_concat(x, y):
    '''Upsample and concatenate two feature maps.
    Args:
      x: (Variable) top feature map to be upsampled.
      y: (Variable) lateral feature map.
    Returns:
      (Variable) concatenated feature map.
    '''
    _,_,H,W = y.size()
    out = F.upsample(x, size=(H,W), mode='bilinear', align_corners=True)
    out = torch.cat((out, y), dim=1)
    return out

def _make_conv(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias) )

class FPN_Resnet(ResNet):
    def __init__(self, block, layers, num_classes):
        super(FPN_Resnet, self).__init__(block, layers)
        num_feats = 256
        bias = False

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer2 = nn.Conv2d(1024, num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer3 = nn.Conv2d( 512, num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer4 = nn.Conv2d( 256, num_feats, kernel_size=1, stride=1, padding=0, bias=bias)

        # Top-down layers
        self.toplayer1 = _make_conv(num_feats, num_feats, kernel_size=3, stride=1, padding=1, bias=bias)
        self.toplayer2 = _make_conv(num_feats, num_feats, kernel_size=3, stride=1, padding=1, bias=bias)
        self.toplayer3 = _make_conv(num_feats, num_feats, kernel_size=3, stride=1, padding=1, bias=bias)

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
        p5 = self.latlayer1(c5) # stride:32; RF:427
        p4 = _upsample_add(p5, self.latlayer2(c4))  # stride:16; RF:427
        p4 = self.toplayer1(p4) # stride:16; RF:427+16*3=475
        p3 = _upsample_add(p4, self.latlayer3(c3))  # stride:8; RF:475
        p3 = self.toplayer2(p3) # stride:8; RF:475+8*3=599
        p2 = _upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        out = self.heatmap(p2)
        return out

    def init_net(self, pretrained=''):
        self.init_weights(pretrained)

        for name, m in self.named_modules():
            if 'lat' in name or 'top' in name:
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for m in self.heatmap.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

class FPN_Densenet(DenseNet):
    def __init__(self, num_init_features, growth_rate, block_config, num_classes):
        super(FPN_Densenet, self).__init__(num_init_features, growth_rate, block_config)
        num_feats = 256
        bias = False

        # Lateral layers
        self.latlayer1 = nn.Conv2d(self.planes[3], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer2 = nn.Conv2d(self.planes[2], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer3 = nn.Conv2d(self.planes[1], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)
        self.latlayer4 = nn.Conv2d(self.planes[0], num_feats, kernel_size=1, stride=1, padding=0, bias=bias)

        # Top-down layers
        self.toplayer1 = _make_conv(num_feats, num_feats, kernel_size=3, stride=1, padding=1, bias=bias)
        self.toplayer2 = _make_conv(num_feats, num_feats, kernel_size=3, stride=1, padding=1, bias=bias)
        self.toplayer3 = _make_conv(num_feats, num_feats, kernel_size=3, stride=1, padding=1, bias=bias)

        self.heatmap = _make_conv(num_feats, num_classes, kernel_size=1)

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
        p5 = self.latlayer1(c5) # stride:32; RF:427
        p4 = _upsample_add(p5, self.latlayer2(c4))  # stride:16; RF:427
        p4 = self.toplayer1(p4) # stride:16; RF:427+16*3=475
        p3 = _upsample_add(p4, self.latlayer3(c3))  # stride:8; RF:475
        p3 = self.toplayer2(p3) # stride:8; RF:475+8*3=599
        p2 = _upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        out = self.heatmap(p2)
        return out

    def init_net(self, pretrained=''):
        self.init_weights(pretrained)

        for name, m in self.named_modules():
            if 'lat' in name or 'top' in name:
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        for m in self.heatmap.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

net_list = ['resnet', 'densenet']
def fpn(backbone, num_classes, pretrained):
    for net in net_list:
        if net in backbone:
            break
    num_layers = int(backbone[len(net):])
    args = eval('{}_dict'.format(net))[num_layers]
    model = eval('FPN_{}'.format(net.capitalize()))(*args, num_classes=num_classes)

    if pretrained:
        model_path = os.path.join('data', 'pretrained', '{}.pth'.format(backbone))
    else:
        model_path = ''
    model.init_net(model_path)
    return model
