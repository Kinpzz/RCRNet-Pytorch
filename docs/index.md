# RCRNet-Pytorch

## Introduction

This repository is the PyTorch implementation of "Semi-Supervised Video Salient Object Detection Using Pseudo-Labels", International Conference on Computer Vision(ICCV), 2019, by Pengxiang Yan, Guanbin Li, Yuan Xie, Zhen Li, Chuan Wang, Tianshui Chen, Liang Lin.

Paper links: [[arxiv](https://arxiv.org/abs/1908.04051)] or [[CVF](http://openaccess.thecvf.com/content_ICCV_2019/html/Yan_Semi-Supervised_Video_Salient_Object_Detection_Using_Pseudo-Labels_ICCV_2019_paper.html)]

## Usage

### Requirements

This code is tested on Ubuntu 16.04, Python=3.6 (via Anaconda3), PyTorch=0.4.1, CUDA=9.0.

```
# Install PyTorch=0.4.1
$ conda install pytorch==0.4.1 torchvision==0.2.1 cuda90 -c pytorch

# Install other packages
$ pip install pyyaml==3.13 addict==2.2.0 tqdm==4.28.1 scipy==1.1.0
```
### Datasets
Our proposed RCRNet is evaluated on three public benchmark VSOD datsets including [VOS](http://cvteam.net/projects/TIP18-VOS/VOS.html), [DAVIS](https://davischallenge.org/) (version: 2016, 480p), and [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/). Please orginaize the datasets according to `config/datasets.yaml` and put them in `data/datasets`. Or you can set argument `--data` to the path of the dataset folder.

### Evaluation
#### Comparison with State-of-the-Art
![comp_video_sota](docs/comp_video_sota.png)
If you want to compare with our method:

**Option 1:** you can download the saliency maps predicted by our model from [Google Drive](https://drive.google.com/open?id=1feY3GdNBS-LUBt0UDWwpA3fl9yHI4Vxr) / [Baidu Pan](https://pan.baidu.com/s/1oXBr9qxyF-8vvilvV5kcPg) (passwd: u079).

**Option 2:** Or you can use our trained model for inference. The weights of trained model are available at [Google Drive](https://drive.google.com/open?id=1TSmi1DyKIvuzuXE1aw7t_ygmcUmjYnN_) / [Baidu Pan](https://pan.baidu.com/s/1PLoajL6X_s29I-4mreSuSQ) (passwd: 6pi3). Then run the following command for inference.
```
# VOS
$ CUDA_VISIBLE_DEVICES=0 python inference.py --data data/datasets --dataset VOS --split test

# DAVIS
$ CUDA_VISIBLE_DEVICES=0 python inference.py --data data/datasets --dataset DAVIS --split val

# FBMS
$ CUDA_VISIBLE_DEVICES=0 python inference.py --data data/datasets --dataset FBMS --split test
```

Then, you can evaluate the saliency maps using your own evaluation code.

### Training
If you want to train the proposed RCRNet from scratch, please refer to our paper and the following instruction carefully.

The proposed RCRNet is built upon an ResNet-50 pretrained on ImageNet.

<img src="static_model.png" style="zoom:80%" />

**First**, we use two image saliency datasets, i.e., [MSRA-B](https://mmcheng.net/msra10k/) and [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html), to pretrain the RCRNet (Figure 2), which contains a spatial feature extractor and a pixel-wise classifer. Here, we provide the weights of RCRNet pretrained on image saliency datasets at at [Google Drive](https://drive.google.com/open?id=1S7nao9WEhIiTmTC-E0nujMxm5Emypti9) or [Baidu Pan](https://pan.baidu.com/s/196cUbTInWJKd8FmiP9Jv_A) (passwd: j839). For simplicity, we do not provide the training code of this step. If you want to train this step you can implement your own training code.

![video_model](video_model.png)

**Second**, we use the RCRNet pretrained on image saliency datasets as the backbone. Then we combine the training set of three video saliency datasets including VOS, DAVIS, and FBMS, to train the full video model, i.e., RCRNet equipped with NER module (Figure 3). You can run the following commands to train the RCRNet+NER.
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
                            --data data/datasets \
                            --checkpoint models/image_pretrained_model.pth
```

### Using psedo-labels for training

![pseudo_label_generator](pseudo_label_generator.png)

As for the second step, if you want train the RCRNet+NER using generated pseudo-labels for joint supervision. You can use our proposed flow-guied pseudo-label generator (FGPLG, Figure 4) to generate the pesdu-labels with a part of ground truth images.

Note that the FGPLG requires flownet2.0 for flow estimation. Thus, please install the pytorch implementation of flownet2.0 using the following commands.
```
# Install FlowNet 2.0 (implemented by NVIDIA)
$ cd flownet2
$ bash install.sh
```

#### Generating pseudo-labels using FGPLG
We provide the weights of FGPLG which is trained under the supervision of 20% ground truth images at [Baidu Pan](https://pan.baidu.com/s/1dw8O2Ua5pKmOKYVgKRyADQ) (passwd: hbsu). You can generate the pseduo-labels by
```
$ CUDA_VISIBLE_DEVICES=0 python generate_pseudo_labels.py \
                            --data                    data/datasets \
                            --checkpoint              models/pseudo_label_generator_5.pth \
                            --pseudo-label-folder     data/pseudo-labels \
                            --label_interval          5 \
                            --frame_between_label_num 1
```

Then you can train the video model under the joint supervision of pseudo-labels.
```
$ CUDA_VISIBLE_DEVICES=0 python train.py \
                            --data                data/datasets \
                            --checkpoint          models/image_pretrained_model.pth \
                            --pseudo-label-folder data/pseudo-labels/1_5
```

#### (Optional) Training FGPLG
You can also train the FGPLG using other propotions of ground truth images by

(Note that need to download the pretrained model of [Flownet2](https://github.com/NVIDIA/flownet2-pytorch#converted-caffe-pre-trained-models))
```
# set l
$ CUDA_VISIBLE_DEVICES=0 python train_fgplg.py \
                            --data               data/datasets \
                            --label_interval     l \
                            --checkpoint         models/image_pretrained_model.pth \
                            --flownet-checkpoint models/FlowNet2_checkpoint.pth.tar
```

Then you can use the trained FGPLG to generate pseudo labels based different numbers of GT images.
```
# set l and m
$ CUDA_VISIBLE_DEVICES=0 python generate_pseudo_labels.py \
                            --data                    data/datasets \
                            --checkpoint              models/pseudo_label_generator_m.pth \
                            --label_interval          l \
                            --frame_between_label_num m \
                            --pseudo-label-folder     data/pseudo-labels
```
Finally, you can train the video model under the joint supervision of pseudo-labels.
```
# set l and m
$ CUDA_VISIBLE_DEVICES=0 python train.py \
                            --data                data/datasets \
                            --checkpoint          models/image_pretrained_model.pth \
                            --pseudo-label-folder data/pseudo-labels/m_l
```


## Citation
If you find this work helpful, please consider citing
```
@inproceedings{yan2019semi,
  title={Semi-Supervised Video Salient Object Detection Using Pseudo-Labels},
  author={Yan, Pengxiang and Li, Guanbin and Xie, Yuan and Li, Zhen and Wang, Chuan and Chen, Tianshui and Lin, Liang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={7284--7293},
  year={2019}
}
```

## Acknowledge
Thanks to the third-party libraries:
* [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) by kazuto1011
* [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) by NVIDIA
* [pytorch-segmentation-toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox) by speedinghzl
* [Non-local_pytorch](https://github.com/AlexHex7/Non-local_pytorch) by AlexHex7