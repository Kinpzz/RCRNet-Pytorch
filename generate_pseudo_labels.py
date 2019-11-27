#!/usr/bin/env python
# coding: utf-8
#
# Author:   Pengxiang Yan
# Email:    yanpx (at) mail2.sysu.edu.cn

from __future__ import absolute_import, division, print_function
import os
import sys
sys.path.append('flownet2')

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import functional as TF

import argparse
from tqdm import tqdm

from libs.datasets import get_transforms, get_datasets
from libs.networks.pseudo_label_generator import FGPLG
from libs.utils.pyt_utils import load_model

parser = argparse.ArgumentParser()

# Dataloading-related settings
parser.add_argument('--data', type=str, default='data/datasets/',
                    help='path to datasets folder')
parser.add_argument('--checkpoint', default='models/pseudo_label_generator_5.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--pseudo-label-folder', default='data/pseudo-labels',
                    help='location to save generated pseudo-labels')
parser.add_argument("--label_interval", default=5, type=int,
                    help="the interval of ground truth labels")
parser.add_argument("--frame_between_label_num", default=1, type=int,
                    help="the number of generated pseudo-labels in each interval")
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers.')

# Model settings
parser.add_argument('--size', default=448, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride.')

# FlowNet setting
parser.add_argument("--fp16", action="store_true",
                    help="Run model in pseudo-fp16 mode (fp16 storage fp32 math).")
parser.add_argument("--rgb_max", type=float, default=1.)

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

if cuda:
    torch.backends.cudnn.benchmark = True
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")

data_transforms = get_transforms(
    input_size=(args.size, args.size),
    image_mode=False
)
dataset = get_datasets(
    name_list=["DAVIS2016", "FBMS", "VOS"],
    split_list=["train", "train", "train"],
    config_path=args.dataset_config,
    root=args.data,
    training=True, # provide labels
    transforms=data_transforms['test'],
    read_clip=True,
    random_reverse_clip=False,
    label_interval=args.label_interval,
    frame_between_label_num=args.frame_between_label_num,
    clip_len=args.frame_between_label_num+2
)

dataloader = data.DataLoader(
    dataset=dataset,
    batch_size=1, # only support 1 video clip
    num_workers=args.num_workers,
    shuffle=False,
    drop_last=True
)

pseudo_label_generator = FGPLG(args=args, output_stride=args.os)

# load pretrained models
if os.path.exists(args.checkpoint):
    print('Loading state dict from: {0}'.format(args.checkpoint))
    pseudo_label_generator = load_model(model=pseudo_label_generator, model_file=args.checkpoint, is_restore=True)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))

pseudo_label_generator.to(device)

pseudo_label_folder = os.path.join(args.pseudo_label_folder, "{}_{}".format(args.frame_between_label_num, args.label_interval))
if not os.path.exists(pseudo_label_folder):
    os.makedirs(pseudo_label_folder)

def generate_pseudo_label():
    pseudo_label_generator.eval()

    for data in tqdm(dataloader):
        images = []
        labels = []
        for frame in data:
            images.append(frame['image'].to(device))
            labels.append(frame['label'].to(device) if 'label' in frame else None)
        with torch.no_grad():
            for i in range(1, args.frame_between_label_num+1):
                pseudo_label = pseudo_label_generator.generate_pseudo_label(images[i], images[0], images[-1], labels[0], labels[-1])
                labels[i] = torch.sigmoid(pseudo_label).detach()
            # save pseudo-labels
            for i, label_ in enumerate(labels):
                for j, label in enumerate(label_.detach().cpu()):
                    dataset = data[i]['dataset'][j]
                    image_id = data[i]['image_id'][j]
                    pseudo_label_path = os.path.join(pseudo_label_folder, "{}/{}.png".format(dataset, image_id))

                    height = data[i]['height'].item()
                    width = data[i]['width'].item()
                    result = TF.to_pil_image(label)
                    result = result.resize((height, width))
                    dirname = os.path.dirname(pseudo_label_path)
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)
                    result.save(pseudo_label_path)

if __name__ == "__main__":
    print("Generating pseudo-labels at {}".format(args.pseudo_label_folder))
    print("label interval: {}".format(args.label_interval))
    print("frame between label num: {}".format(args.frame_between_label_num))
    generate_pseudo_label()
