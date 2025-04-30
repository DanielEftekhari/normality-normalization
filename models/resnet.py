# adapted from: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

import collections
import torch.nn as nn

import activations
import layers as norm_layers


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        layer_id,
        block_id,
        _layers,
        _aux,
        _layer_num,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer='normal',
        activation='relu',
        cfg=None,
        device=None,
        *args, **kwargs
        ):
        super().__init__()

        self.activation = getattr(activations, activation.lower())

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.BatchNormalNorm2d
        elif norm_layer == 'batch': self.norm_layer = norm_layers.BatchNorm2d
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        # Both self.affine1 and self.downsample layers downsample the input when stride != 1
        _layer_num[0] += 1
        self.add_module('affine1', conv3x3(inplanes, planes, stride))
        self.add_module('norm1', self.norm_layer(planes, device=device, **kwargs))
        self.add_module('activation1', self.activation(inplace=False))
        _aux['layer{}.{}.affine{}'.format(layer_id, block_id, 1)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'affine', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.norm{}'.format(layer_id, block_id, 1)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'norm', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.activation{}'.format(layer_id, block_id, 1)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'activation', 'layer_number': _layer_num[0]}

        _layer_num[0] += 1
        self.add_module('affine2', conv3x3(planes, planes))
        self.add_module('norm2', self.norm_layer(planes, device=device, **kwargs))
        self.add_module('activation2', self.activation(inplace=False))
        _aux['layer{}.{}.affine{}'.format(layer_id, block_id, 2)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 2, 'type': 'affine', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.norm{}'.format(layer_id, block_id, 2)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 2, 'type': 'norm', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.activation{}'.format(layer_id, block_id, 2)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 2, 'type': 'activation', 'layer_number': _layer_num[0]}

        if downsample is not None:
            self.add_module('downsample', downsample)
            _aux['layer{}.{}.downsample.1'.format(layer_id, block_id)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'downsample', 'layer_number': _layer_num[0]}
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.affine1(x)
        out = self.norm1(out)
        out = self.activation1(out)

        out = self.affine2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.affine2)
    # while original implementation places the stride at the first 1x1 convolution(self.affine1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        layer_id,
        block_id,
        _layers,
        _aux,
        _layer_num,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer='normal',
        activation='relu',
        cfg=None,
        device=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.activation = getattr(activations, activation.lower())

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.BatchNormalNorm2d
        elif norm_layer == 'batch': self.norm_layer = norm_layers.BatchNorm2d
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        width = int(planes * (base_width / 64.0)) * groups

        # Both self.affine2 and self.downsample layers downsample the input when stride != 1
        _layer_num[0] += 1
        self.add_module('affine1', conv1x1(inplanes, width))
        self.add_module('norm1', self.norm_layer(width, device=device, **kwargs))
        self.add_module('activation1', self.activation(inplace=False))
        _aux['layer{}.{}.affine{}'.format(layer_id, block_id, 1)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'affine', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.norm{}'.format(layer_id, block_id, 1)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'norm', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.activation{}'.format(layer_id, block_id, 1)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'activation', 'layer_number': _layer_num[0]}

        _layer_num[0] += 1
        self.add_module('affine2', conv3x3(width, width, stride, groups, dilation))
        self.add_module('norm2', self.norm_layer(width, device=device, **kwargs))
        self.add_module('activation2', self.activation(inplace=False))
        _aux['layer{}.{}.affine{}'.format(layer_id, block_id, 2)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 2, 'type': 'affine', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.norm{}'.format(layer_id, block_id, 2)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 2, 'type': 'norm', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.activation{}'.format(layer_id, block_id, 2)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 2, 'type': 'activation', 'layer_number': _layer_num[0]}

        _layer_num[0] += 1
        self.add_module('affine3', conv1x1(width, planes * self.expansion))
        self.add_module('norm3', self.norm_layer(planes * self.expansion, device=device, **kwargs))
        self.add_module('activation3', self.activation(inplace=False))
        _aux['layer{}.{}.affine{}'.format(layer_id, block_id, 3)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 3, 'type': 'affine', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.norm{}'.format(layer_id, block_id, 3)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 3, 'type': 'norm', 'layer_number': _layer_num[0]}
        _aux['layer{}.{}.activation{}'.format(layer_id, block_id, 3)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 3, 'type': 'activation', 'layer_number': _layer_num[0]}

        if downsample is not None:
            self.add_module('downsample', downsample)
            _aux['layer{}.{}.downsample.1'.format(layer_id, block_id)] = {'layer_outer': layer_id, 'block': block_id, 'layer_inner': 1, 'type': 'downsample', 'layer_number': _layer_num[0]}
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.affine1(x)
        out = self.norm1(out)
        out = self.activation1(out)

        out = self.affine2(out)
        out = self.norm2(out)
        out = self.activation2(out)

        out = self.affine3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        c_in=3,
        num_classes=1000,
        standardize=None,
        norm_layer='normal',
        activation='relu',
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        cfg=None,
        device=None,
        *args, **kwargs
    ) -> None:
        super().__init__()

        self.c_in = c_in
        self._activation = activation
        self.activation = getattr(activations, activation.lower())

        self._layers = {}
        self._aux = collections.defaultdict(dict)
        self._layer_num = [0]

        self.add_module('standardize', norm_layers.Standardize(mean=standardize[0], std=standardize[1], device=device))

        self._norm_layer = norm_layer
        if norm_layer == 'normal': self.norm_layer = norm_layers.BatchNormalNorm2d
        elif norm_layer == 'batch': self.norm_layer = norm_layers.BatchNorm2d
        elif norm_layer == 'none': self.norm_layer = nn.Identity

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self._layer_num[0] += 1
        if cfg.dataset.lower() in ('cifar10', 'cifar100', 'svhn', 'tinyimagenet',):
            self.add_module('affine1', nn.Conv2d(self.c_in, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False))
        elif cfg.dataset.lower() in ('stl10',):
            self.add_module('affine1', nn.Conv2d(self.c_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False))
        else:
            # this defaults to the commonly used case
            # (which is captured by the stl10 case, but added here separately to help distinguish the different dataset cases)
            self.add_module('affine1', nn.Conv2d(self.c_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False))
        self.add_module('norm1', self.norm_layer(self.inplanes, device=device, **kwargs))
        self.add_module('activation1', self.activation(inplace=False))
        if cfg.dataset.lower() in ('cifar10', 'cifar100', 'svhn', 'tinyimagenet',):
            self.maxpool = None
        elif cfg.dataset.lower() in ('stl10',):
            self.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            # this defaults to the commonly used case
            # (which is captured by the stl10 case, but added here separately to help distinguish the different dataset cases)
            self.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self._aux['affine1'] = {'layer_outer': 0, 'block': 0, 'layer_inner': 1, 'type': 'affine', 'layer_number': self._layer_num[0]}
        self._aux['norm1'] = {'layer_outer': 0, 'block': 0, 'layer_inner': 1, 'type': 'norm', 'layer_number': self._layer_num[0]}
        self._aux['activation1'] = {'layer_outer': 0, 'block': 0, 'layer_inner': 1, 'type': 'activation', 'layer_number': self._layer_num[0]}

        self.add_module('layer1', self._make_layer(1, block, 64, layers[0], stride=1, cfg=cfg, device=device, **kwargs))
        self.add_module('layer2', self._make_layer(2, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], cfg=cfg, device=device, **kwargs))
        self.add_module('layer3', self._make_layer(3, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], cfg=cfg, device=device, **kwargs))
        self.add_module('layer4', self._make_layer(4, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], cfg=cfg, device=device, **kwargs))

        self.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.add_module('fc', nn.Linear(512 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (norm_layers.BatchNormalNorm2d, norm_layers.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.norm3.weight is not None:
                    nn.init.constant_(m.norm3.weight, 0)
                elif isinstance(m, BasicBlock) and m.norm2.weight is not None:
                    nn.init.constant_(m.norm2.weight, 0)

        self._layers = {}
        for elem in self.named_modules():
            if len(elem[0]) > 0:
                self._layers[elem[0]] = elem[1]
        self._names = list(self._layers.keys())

    def _make_layer(
        self,
        layer_id,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        cfg=None,
        device=None,
        *args,
        **kwargs,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                self.norm_layer(planes * block.expansion, device=device, **kwargs),
            )

        block_id = 0
        layers = []
        layers.append(
            block(
                layer_id=layer_id,
                block_id=block_id,
                _layers=self._layers,
                _aux=self._aux,
                _layer_num=self._layer_num,
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                groups=self.groups,
                base_width=self.base_width,
                dilation=previous_dilation,
                norm_layer=norm_layer,
                activation=self._activation,
                cfg=cfg,
                device=device,
                **kwargs
            )
        )
        block_id += 1

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    layer_id=layer_id,
                    block_id=block_id,
                    _layers=self._layers,
                    _aux=self._aux,
                    _layer_num=self._layer_num,
                    inplanes=self.inplanes,
                    planes=planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=self._activation,
                    cfg=cfg,
                    device=device,
                    **kwargs
                )
            )
            block_id += 1

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.standardize(x)

        x = self.affine1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
