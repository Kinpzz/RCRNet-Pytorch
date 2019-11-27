#!/usr/bin/env python
# coding: utf-8
#
# Author:   Pengxiang Yan
# Email:    yanpx (at) mail2.sysu.edu.cn

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_dilation import resnet50, Bottleneck, conv1x1

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)

class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, output_stride):
        super(_ASPPModule, self).__init__()
        if output_stride == 8:
            pyramids = [12, 24, 36]
        elif output_stride == 16:
            pyramids = [6, 12, 18]
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d((1,1))),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )
        self.fire = nn.Sequential(
            OrderedDict(
                [
                    ("conv", _ConvBatchNormReLU(out_channels * 5, out_channels, 3, 1, 1, 1)),
                    ("dropout", nn.Dropout2d(0.1))
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        h = self.fire(h)
        return h

class _RefinementModule(nn.Module):
    """ Reduce channels and refinment module"""

    def __init__(self,
        bottom_up_channels,
        reduce_channels,
        top_down_channels,
        refinement_channels,
        expansion=2
    ):
        super(_RefinementModule, self).__init__()
        downsample = None
        if bottom_up_channels != reduce_channels:
            downsample = nn.Sequential(
                conv1x1(bottom_up_channels, reduce_channels),
                nn.BatchNorm2d(reduce_channels),
            )
        self.skip = Bottleneck(bottom_up_channels, reduce_channels // expansion, 1, 1, downsample, expansion)
        self.refine = _ConvBatchNormReLU(reduce_channels + top_down_channels, refinement_channels, 3, 1, 1, 1)
    def forward(self, td, bu):
        td = self.skip(td)
        x = torch.cat((bu, td), dim=1)
        x = self.refine(x)
        return x

class RCRNet(nn.Module):

    def __init__(self, n_classes, output_stride, input_channels=3, pretrained=False):
        super(RCRNet, self).__init__()
        self.resnet = resnet50(pretrained=pretrained, output_stride=output_stride, input_channels=input_channels)
        self.aspp = _ASPPModule(2048, 256, output_stride)
        # Decoder
        self.decoder = nn.Sequential(
                    OrderedDict(
                        [
                            ("conv1", _ConvBatchNormReLU(128, 256, 3, 1, 1, 1)),
                            ("conv2", nn.Conv2d(256, n_classes, kernel_size=1)),
                        ]
                    )
                )
        self.add_module("refinement1", _RefinementModule(1024, 96, 256, 128, 2))
        self.add_module("refinement2", _RefinementModule(512, 96, 128, 128, 2))
        self.add_module("refinement3", _RefinementModule(256, 96, 128, 128, 2))

        if pretrained:
            for key in self.state_dict():
                if 'resnet' not in key:
                    self.init_layer(key)

    def init_layer(self, key):
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                if self.state_dict()[key].ndimension() >= 2:
                    nn.init.kaiming_normal_(self.state_dict()[key], mode='fan_out', nonlinearity='relu')
            elif 'bn' in key:
                self.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            self.state_dict()[key][...] = 0.001

    def feat_conv(self, x):
        '''
            Spatial feature extractor
        '''
        block0 = self.resnet.conv1(x)
        block0 = self.resnet.bn1(block0)
        block0 = self.resnet.relu(block0)
        block0 = self.resnet.maxpool(block0)

        block1 = self.resnet.layer1(block0)
        block2 = self.resnet.layer2(block1)
        block3 = self.resnet.layer3(block2)
        block4 = self.resnet.layer4(block3)
        block4 = self.aspp(block4)
        return block1, block2, block3, block4

    def seg_conv(self, block1, block2, block3, block4, shape):
        '''
            Pixel-wise classifer
        '''
        bu1 = self.refinement1(block3, block4)
        bu1 = F.interpolate(bu1, size=block2.shape[2:], mode="bilinear", align_corners=False)
        bu2 = self.refinement2(block2, bu1)
        bu2 = F.interpolate(bu2, size=block1.shape[2:], mode="bilinear", align_corners=False)
        bu3 = self.refinement3(block1, bu2)
        bu3 = F.interpolate(bu3, size=shape, mode="bilinear", align_corners=False)
        seg = self.decoder(bu3)
        return seg

    def forward(self, x):
        block1, block2, block3, block4 = self.feat_conv(x)
        seg = self.seg_conv(block1, block2, block3, block4, x.shape[2:])
        return seg