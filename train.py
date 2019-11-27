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

import argparse
from tqdm import tqdm
import numpy as np

from libs.datasets import get_transforms, get_datasets
from libs.networks import VideoModel
from libs.utils.metric import StructureMeasure
from libs.utils.pyt_utils import load_model

parser = argparse.ArgumentParser()

# Dataloading-related settings
parser.add_argument('--data', type=str, default='data/datasets/',
                    help='path to datasets folder')
parser.add_argument('--checkpoint', default='models/image_pretrained_model.pth',
                    help='path to the pretrained checkpoint')
parser.add_argument('--dataset-config', default='config/datasets.yaml',
                    help='dataset config file')
parser.add_argument('--save-folder', default='models/checkpoints',
                    help='location to save checkpoint models')
parser.add_argument('--pseudo-label-folder', default='',
                    help='location to load pseudo-labels')
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

train_dataset = get_datasets(
    name_list=["DAVIS2016", "FBMS", "VOS"],
    split_list=["train", "train", "train"],
    config_path=args.dataset_config,
    root=args.data,
    training=True,
    transforms=data_transforms['train'],
    read_clip=True,
    random_reverse_clip=True,
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

# loading pseudo-labels for training
if os.path.exists(args.pseudo_label_folder):
    print("Loading pseudo-labels from {}".format(args.pseudo_label_folder))
    datasets = dataloaders['train'].dataset
    for dataset in datasets.datasets:
        dataset._reset_files(clip_len=args.clip_len, label_dir=os.path.join(args.pseudo_label_folder, dataset.name))
    if isinstance(datasets, data.ConcatDataset):
        datasets.cumulative_sizes = datasets.cumsum(datasets.datasets)

model = VideoModel(output_stride=args.os)
# load pretrained models
if os.path.exists(args.checkpoint):
    print('Loading state dict from: {0}'.format(args.checkpoint))
    if args.start_epoch == 0:
        model = load_model(model=model, model_file=args.checkpoint, is_restore=False)
    else:
        model = load_model(model=model, model_file=args.checkpoint, is_restore=True)
else:
    raise ValueError("Cannot find model file at {}".format(args.checkpoint))

model = nn.DataParallel(model)
model.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=args.lr)

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
                model.train()  # Set model to training mode
                model.module.freeze_bn()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
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
                    preds = model(images)
                    loss = []
                    for pred, label in zip(preds, labels):
                        loss.append(criterion(pred, label))
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        torch.autograd.backward(loss)
                        optimizer.step()
                # statistics
                for _loss in loss:
                    running_loss += _loss.item()
                preds = [torch.sigmoid(pred) for pred in preds] # activation

                # iterate list
                for i, (label_, pred_) in enumerate(zip(labels, preds)):
                    # interate batch
                    for j, (label, pred) in enumerate(zip(label_.detach().cpu(), pred_.detach().cpu())):
                        pred_idx = pred[0,:,:].numpy()
                        label_idx = label[0,:,:].numpy()
                        if phase == 'val':
                            running_smean += StructureMeasure(pred_idx.astype(np.float32), (label_idx>=0.5).astype(np.bool))
                        running_mae += np.abs(pred_idx - label_idx).mean()

            samples_num = len(dataloaders[phase].dataset)
            samples_num *= args.clip_len
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
                    model_path = os.path.join(args.save_folder, "video_current_best_model.pth")
                    print("Saving current best model at: {}".format(model_path) )
                    torch.save(
                        model.module.state_dict(),
                        model_path,
                    )
        if epoch > 0 and epoch % args.backup_epochs == 0:
            # save model
            model_path = os.path.join(args.save_folder, "video_epoch-{}.pth".format(epoch))
            print("Backup model at: {}".format(model_path))
            torch.save(
                model.module.state_dict(),
                model_path,
            )

    print('Best S-measure: {} at epoch {}'.format(best_smeasure, best_epoch))

if __name__ == "__main__":
    train()
