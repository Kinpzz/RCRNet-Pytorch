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

import argparse
from tqdm import tqdm
import numpy as np

from libs.datasets import get_transforms, get_datasets
from libs.networks.pseudo_label_generator import FGPLG
from libs.utils.metric import StructureMeasure
from libs.utils.pyt_utils import load_model

parser = argparse.ArgumentParser()

# Dataloading-related settings
parser.add_argument('--data', type=str, default='data/datasets/',
                    help='path to datasets folder')
parser.add_argument('--checkpoint', default='models/image_pretrained_model.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--flownet-checkpoint', default='models/FlowNet2_checkpoint.pth.tar',
                    help='path to the checkpoint of pretrained flownet2')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--save-folder', default='models/checkpoints',
                    help='location to save checkpoint models')
parser.add_argument("--label_interval", default=5, type=int,
                    help="the interval of ground truth labels")
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers.')

# Training procedure settings
parser.add_argument('--batch-size', default=1, type=int,
                    help='batch size for each gpu. Only support 1 for video clips.')
parser.add_argument('--backup-epochs', type=int, default=1,
                    help='iteration epoch to perform state backups')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--start-epoch', type=int, default=0,
                    help='epoch number to resume')
parser.add_argument('--eval-first', default=False, action='store_true',
                    help='evaluate model weights before training')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    help='initial learning rate')

# Model settings
parser.add_argument('--size', default=448, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride.')
parser.add_argument("--clip_len", type=int, default=3,
                    help="the number of frames in a video clip.")

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

train_dataset = get_datasets(
    name_list=["DAVIS2016", "FBMS", "VOS"],
    split_list=["train", "train", "train"],
    config_path=args.dataset_config,
    root=args.data,
    training=True,
    transforms=data_transforms['train'],
    read_clip=True,
    random_reverse_clip=True,
    label_interval=args.label_interval,
    clip_len=args.clip_len
)
val_dataset = get_datasets(
    name_list="VOS",
    split_list="val",
    config_path=args.dataset_config,
    root=args.data,
    training=True,
    transforms=data_transforms['val'],
    read_clip=True,
    random_reverse_clip=False,
    label_interval=args.label_interval,
    clip_len=args.clip_len
)

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=True
)
val_dataloader = data.DataLoader(
    dataset=val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False
)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

pseudo_label_generator = FGPLG(args=args, output_stride=args.os)

if os.path.exists(args.checkpoint):
    print('Loading state dict from: {0}'.format(args.checkpoint))
    if args.start_epoch == 0:
        pseudo_label_generator = load_model(model=pseudo_label_generator, model_file=args.checkpoint, is_restore=False)
        if os.path.exists(args.flownet_checkpoint):
            pseudo_label_generator.flownet = load_model(model=pseudo_label_generator.flownet, model_file=args.flownet_checkpoint, is_restore=True)
        else:
            raise ValueError("Cannot pretrained flownet model file at {}".format(args.flownet_checkpoint))
    else:
        pseudo_label_generator = load_model(model=pseudo_label_generator, model_file=args.checkpoint, is_restore=True)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))

pseudo_label_generator = nn.DataParallel(pseudo_label_generator)
pseudo_label_generator.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pseudo_label_generator.module.parameters()), lr=args.lr)

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

def train():
    best_smeasure = 0.0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs+1):
        # Each epoch has a training and validation phase
        if args.eval_first:
            phases = ['val']
        else:
            phases = ['train', 'val']

        for phase in phases:
            if phase == 'train':
                pseudo_label_generator.train()  # Set model to training mode
                pseudo_label_generator.module.freeze_bn()
            else:
                pseudo_label_generator.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_acc = 0.0
            running_iou = 0.0
            running_mae = 0.0
            running_smean = 0.0
            print("{} epoch {}...".format(phase, epoch))
            # Iterate over data.
            for data in tqdm(dataloaders[phase]):
                images, labels = [], []
                for frame in data:
                    images.append(frame['image'].to(device))
                    labels.append(frame['label'].to(device))
                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # read clips
                    preds, labels = pseudo_label_generator(images, labels)
                    loss = criterion(preds, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        torch.autograd.backward(loss)
                        optimizer.step()
                # statistics
                running_loss += loss.item()
                preds = torch.sigmoid(preds) # activation

                pred_idx = preds.squeeze().detach().cpu().numpy()
                label_idx = labels.squeeze().detach().cpu().numpy()
                if phase == 'val':
                    running_smean += StructureMeasure(pred_idx.astype(np.float32), (label_idx>=0.5).astype(np.bool))
                running_mae += np.abs(pred_idx - label_idx).mean()

            samples_num = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / samples_num
            epoch_mae = running_mae / samples_num
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} MAE: {:.4f}'.format(phase, epoch_mae))

            # save current best epoch
            if phase == 'val':
                epoch_smeasure = running_smean / samples_num
                print('{} S-measure: {:.4f}'.format(phase, epoch_smeasure))
                if epoch_smeasure > best_smeasure:
                    best_smeasure = epoch_smeasure
                    best_epoch = epoch
                    model_path = os.path.join(args.save_folder, "fgplg_current_best_model.pth")
                    print("Saving current best model at: {}".format(model_path) )
                    torch.save(
                        pseudo_label_generator.module.state_dict(),
                        model_path,
                    )
        if epoch > 0 and epoch % args.backup_epochs == 0:
            # save model
            model_path = os.path.join(args.save_folder, "fgplg_epoch-{}.pth".format(epoch))
            print("Backup model at: {}".format(model_path))
            torch.save(
                pseudo_label_generator.module.state_dict(),
                model_path,
            )

    print('Best S-measure: {} at epoch {}'.format(best_smeasure, best_epoch))

if __name__ == "__main__":
    train()
