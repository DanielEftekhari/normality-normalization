# adapted from: https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import activations
import layers as norm_layers


class BasicBlock(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 stride,
                 dropout=0.0,
                 norm_layer='normal',
                 activation='relu',
                 cfg=None,
                 device=None,
                 *args, **kwargs
                 ):
        super(BasicBlock, self).__init__()

        self.activation = getattr(activations, activation.lower())

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.BatchNormalNorm2d
        elif norm_layer == 'batch': self.norm_layer = norm_layers.BatchNorm2d
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        self.norm1 = self.norm_layer(in_planes, device=device, **kwargs)
        self.relu1 = self.activation(inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = self.norm_layer(out_planes, device=device, **kwargs)
        self.relu2 = self.activation(inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.norm1(x)
            x = self.relu1(x)
        else:
            out = self.norm1(x)
            out = self.relu1(out)

        if self.equalInOut:
            out = self.conv1(out)
            out = self.norm2(out)
            out = self.relu2(out)
        else:
            out = self.conv1(x)
            out = self.norm2(out)
            out = self.relu2(out)

        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self,
                 nb_layers,
                 in_planes,
                 out_planes,
                 block,
                 stride,
                 dropout=0.0,
                 norm_layer='normal',
                 activation='relu',
                 cfg=None,
                 device=None,
                 *args, **kwargs
                 ):
        super(NetworkBlock, self).__init__()

        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout,
                                      norm_layer=norm_layer,
                                      activation=activation,
                                      cfg=cfg,
                                      device=device,
                                      **kwargs)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout,
                    norm_layer,
                    activation,
                    cfg,
                    device,
                    *args, **kwargs):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout,
                                norm_layer=norm_layer,
                                activation=activation,
                                cfg=cfg,
                                device=device,
                                **kwargs
                                ))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self,
                 depth,
                 widen_factor=1,
                 dropout=0.0,
                 c_in=3,
                 num_classes=1000,
                 standardize=None,
                 norm_layer='normal',
                 activation='relu',
                 cfg=None,
                 device=None,
                 *args, **kwargs
                 ):
        super(WideResNet, self).__init__()

        self.c_in = c_in
        self._activation = activation
        self.activation = getattr(activations, activation.lower())

        if standardize is not None:
            self.standardize = norm_layers.Standardize(mean=standardize[0], std=standardize[1], device=device)
        else:
            self.standardize = nn.Identity()

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.BatchNormalNorm2d
        elif norm_layer == 'batch': self.norm_layer = norm_layers.BatchNorm2d
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        nChannels = [
                     min(int(16 * 1), int(16 * widen_factor)),
                    #  int(16 * 1),
                     int(16 * widen_factor),
                     int(32 * widen_factor),
                     int(64 * widen_factor)
                    ]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock

        if cfg.dataset.lower() in ('cifar10', 'cifar100', 'svhn', 'tinyimagenet',):
            self.conv1 = nn.Conv2d(self.c_in, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        elif cfg.dataset.lower() in ('stl10',):
            self.conv1 = nn.Conv2d(self.c_in, nChannels[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(self.c_in, nChannels[0], kernel_size=7, stride=2, padding=3, bias=False)

        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout,
                                norm_layer=norm_layer,
                                activation=activation,
                                cfg=cfg,
                                device=device,
                                **kwargs
                                   )
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout,
                                norm_layer=norm_layer,
                                activation=activation,
                                cfg=cfg,
                                device=device,
                                **kwargs
                                   )
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout,
                                norm_layer=norm_layer,
                                activation=activation,
                                cfg=cfg,
                                device=device,
                                **kwargs
                                   )

        # global average pooling and classifier
        self.norm1 = self.norm_layer(nChannels[3], device=device, **kwargs)
        self.relu = self.activation(inplace=False)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (norm_layers.BatchNormalNorm2d, norm_layers.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.standardize(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.norm1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
