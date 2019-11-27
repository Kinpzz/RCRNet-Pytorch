#!/usr/bin/env python
# coding: utf-8
#
# Author:   Pengxiang Yan
# Email:    yanpx (at) mail2.sysu.edu.cn

from __future__ import absolute_import, division, print_function
import os

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.transforms import functional as TF

import argparse
from tqdm import tqdm

from libs.datasets import get_transforms, get_datasets
from libs.networks import VideoModel
from libs.utils.pyt_utils import load_model

parser = argparse.ArgumentParser()

# Dataloading-related settings
parser.add_argument('--data', type=str, default='data/datasets/',
                    help='path to datasets folder')
parser.add_argument('--dataset', default='VOS', type=str,
                    help='dataset name for inference')
parser.add_argument('--split', default='test', type=str,
                    help='dataset split for inference')
parser.add_argument('--checkpoint', default='models/video_best_model.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--results-folder', default='data/results/',
                    help='location to save predicted saliency maps')
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers.')

# Model settings
parser.add_argument('--size', default=448, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride.')
parser.add_argument("--clip_len", type=int, default=4,
                    help="the number of frames in a video clip.")

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
    name_list=args.dataset,
    split_list=args.split,
    config_path=args.dataset_config,
    root=args.data,
    training=False,
    transforms=data_transforms['test'],
    read_clip=True,
    random_reverse_clip=False,
    label_interval=1,
    frame_between_label_num=0,
    clip_len=args.clip_len
)

dataloader = data.DataLoader(
    dataset=dataset,
    batch_size=1, # only support 1 video clip
    num_workers=args.num_workers,
    shuffle=False
)

model = VideoModel(output_stride=args.os)

# load pretrained models
if os.path.exists(args.checkpoint):
    print('Loading state dict from: {0}'.format(args.checkpoint))
    model = load_model(model=model, model_file=args.checkpoint, is_restore=True)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))

model.to(device)

def inference():
    model.eval()
    print("Begin inference on {} {}.".format(args.dataset, args.split))
    for data in tqdm(dataloader):
        images = [frame['image'].to(device) for frame in data]
        with torch.no_grad():
            preds = model(images)
            preds = [torch.sigmoid(pred) for pred in preds]
        # save predicted saliency maps
        for i, pred_ in enumerate(preds):
            for j, pred in enumerate(pred_.detach().cpu()):
                dataset = data[i]['dataset'][j]
                image_id = data[i]['image_id'][j]
                height = data[i]['height'].item()
                width = data[i]['width'].item()
                result_path = os.path.join(args.results_folder, "{}/{}.png".format(dataset, image_id))

                result = TF.to_pil_image(pred)
                result = result.resize((height, width))
                dirname = os.path.dirname(result_path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                result.save(result_path)

if __name__ == "__main__":
    inference()
