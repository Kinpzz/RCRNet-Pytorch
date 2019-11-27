#!/usr/bin/env python
# coding: utf-8
#
# Author:   Pengxiang Yan
# Email:    yanpx (at) mail2.sysu.edu.cn

import random

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.utils import data
import numpy as np

def get_transforms(image_mode, input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    data_transforms = {
        'train': transforms.Compose([
            ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.3, image_mode=image_mode),
            RandomResizedCrop(input_size, image_mode),
            RandomFlip(image_mode),
            ToTensor(),
            Normalize(mean=mean,
                    std=std)
        ]) if image_mode else transforms.Compose([
                                Resize(input_size),
                                ToTensor(),
                                Normalize(mean=mean,
                                        std=std)
        ]),
        'val': transforms.Compose([
            Resize(input_size),
            ToTensor(),
            Normalize(mean=mean,
                            std=std)
        ]),
        'test': transforms.Compose([
            Resize(input_size),
            ToTensor(),
            Normalize(mean=mean,
                            std=std)
        ]),
    }
    return data_transforms

class ColorJitter(transforms.ColorJitter):
    def __init__(self, image_mode, **kwargs):
        super(ColorJitter, self).__init__(**kwargs)
        self.transform = None
        self.image_mode = image_mode
    def __call__(self, sample):
        if self.transform is None or self.image_mode:
            self.transform = self.get_params(self.brightness, self.contrast,
                                        self.saturation, self.hue)
        sample['image'] = self.transform(sample['image'])
        return sample

class RandomResizedCrop(object):
    """
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, image_mode, scale=(0.7, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.i, self.j, self.h, self.w = None, None, None, None
        self.image_mode = image_mode
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.i is None or self.image_mode:
            self.i, self.j, self.h, self.w = transforms.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = F.resized_crop(image, self.i, self.j, self.h, self.w, self.size, Image.BILINEAR)
        label = F.resized_crop(label, self.i, self.j, self.h, self.w, self.size, Image.BILINEAR)
        sample['image'], sample['label'] = image, label
        return sample

class RandomFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    """
    def __init__(self, image_mode):
        self.rand_flip_index = None
        self.image_mode = image_mode
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.rand_flip_index is None or self.image_mode:
            self.rand_flip_index = random.randint(-1,2)
        # 0: horizontal flip, 1: vertical flip, -1: horizontal and vertical flip
        if self.rand_flip_index == 0:
            image = F.hflip(image)
            label = F.hflip(label)
        elif self.rand_flip_index == 1:
            image = F.vflip(image)
            label = F.vflip(label)
        elif self.rand_flip_index == 2:
            image = F.vflip(F.hflip(image))
            label = F.vflip(F.hflip(label))
        sample['image'], sample['label'] = image, label
        return sample

class Resize(object):
    """ Resize PIL image use both for training and inference"""
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.resize(image, self.size, Image.BILINEAR)
        if label is not None:
            label = F.resize(label, self.size, Image.BILINEAR)
        sample['image'], sample['label'] = image, label
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # Image range from [0~255] to [0.0 ~ 1.0]
        image = F.to_tensor(image)
        if label is not None:
            label = torch.from_numpy(np.array(label)).unsqueeze(0).float()
        return {'image': image, 'label': label}

class Normalize(object):
    """ Normalize a tensor image with mean and standard deviation.
        args:    tensor (Tensor) – Tensor image of size (C, H, W) to be normalized.
        Returns: Normalized Tensor image.
    """
    # default caffe mode
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}

