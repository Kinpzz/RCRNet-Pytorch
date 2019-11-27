#!/usr/bin/env python
# coding: utf-8
#
# Author:   Pengxiang Yan
# Email:    yanpx (at) mail2.sysu.edu.cn

from libs.networks.rcrnet import RCRNet

from flownet2.models import FlowNet2
from flownet2.networks.resample2d_package.resample2d import Resample2d

import torch
import torch.nn as nn

_mean = [0.485, 0.456, 0.406]
_std  = [0.229, 0.224, 0.225]

def normalize_flow(flow):
    origin_size = flow.shape[2:]
    flow[:, 0, :, :] /= origin_size[1] # dx
    flow[:, 1, :, :] /= origin_size[0] # dy
    norm_flow = (flow[:, 0, :, :] ** 2 + flow[:, 1, :, :] ** 2) ** 0.5
    return norm_flow.unsqueeze(1)

def compute_flow(flownet, data, data_ref):
    # flow from data_ref to data
    images = [data[0].clone(), data_ref[0].clone()]
    for image in images:
        for i, (mean, std) in enumerate(zip(_mean, _std)):
            image[i].mul_(std).add_(mean)
    images = torch.stack(images)
    images = images.permute((1, 0, 2, 3)) # to channel, 2, h, w
    im = images.unsqueeze(0).float() # add batch_size = 1 [batch_size, channel, 2, h, w]
    return flownet(im)

def resize_flow(flow, size):
    origin_size = flow.shape[2:]
    flow = F.interpolate(flow.clone(), size=size, mode="near")
    flow[:, 0, :, :] /= origin_size[1] / size[1] # dx
    flow[:, 1, :, :] /= origin_size[0] / size[0] # dy
    return flow

class FGPLG(nn.Module):
    def __init__(self, args, output_stride=16):
        super(FGPLG, self).__init__()
        self.flownet = FlowNet2(args)
        self.warp = Resample2d()
        channels = 7

        self.backbone = RCRNet(
            n_classes=1,
            output_stride=output_stride,
            pretrained=False,
            input_channels=channels
        )

        self.freeze_bn()
        self.freeze_layer()

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()
    def freeze_layer(self):
        if hasattr(self, 'flownet'):
            for p in self.flownet.parameters():
                p.requires_grad = False

    def generate_pseudo_label(self, frame, frame_l, frame_r, label_l, label_r):
        flow_forward = compute_flow(self.flownet, frame, frame_l)
        flow_backward = compute_flow(self.flownet, frame, frame_r)
        warp_label_l = self.warp(label_l, flow_forward)
        warp_label_r = self.warp(label_r, flow_backward)
        inputs = torch.cat((
            frame,
            warp_label_l,
            warp_label_r,
            normalize_flow(flow_forward),
            normalize_flow(flow_backward)
            ), 1)
        pseudo_label = self.backbone(inputs)
        return pseudo_label

    def forward(self, clip, clip_label):
        pseudo_label = self.generate_pseudo_label(clip[1], clip[0], clip[2], clip_label[0], clip_label[2])
        return pseudo_label, clip_label[1]
