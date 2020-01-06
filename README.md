![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch 1.1](https://img.shields.io/badge/pytorch-1.1-yellow.svg)

# Mutual Mean-Teaching (MMT)

The *official* implementation for the [Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification](https://openreview.net/forum?id=rJlnOhVYPS) which is accepted by [ICLR-2020](https://iclr.cc).

![framework](figs/framework.png)

## Installation

```shell
git clone https://github.com/yxgeee/MMT.git
cd MMT
python setup.py install
```

## Prepare Datasets

```shell
cd example && mkdir data
```
Download the raw datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf), [MSMT17](https://arxiv.org/abs/1711.08565),
and then unzip them under the directory like
```
MMT/example/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
└── msmt17
    └── MSMT17_V1
```

## Prepare Pre-trained Models
When *training with the backbone of [IBN-ResNet-50](https://arxiv.org/abs/1807.09441)*, you need to download the [ImageNet](http://www.image-net.org/) pre-trained model from this [link](https://drive.google.com/drive/folders/1thS2B8UOSBi_cJX6zRy6YYRwz_nVFI_S) and save it under the path of `logs/pretrained/`.
```shell
mkdir logs && cd logs
mkdir pretrained
```
The file tree should be
```
MMT/logs
└── pretrained
    └── resnet50_ibn_a.pth.tar
```

## Example #1:
Transferring from [DukeMTMC-reID](https://arxiv.org/abs/1609.01775) to [Market-1501](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.pdf) on the backbone of [ResNet-50](https://arxiv.org/abs/1512.03385), *i.e. Duke-to-Market (ResNet-50)*.

### Train
We utilize 4 GTX-1080TI GPUs for training.

#### Stage I: Pre-training on the source domain

```shell
sh scripts/pretrain.sh dukemtmc market1501 resnet50 1
sh scripts/pretrain.sh dukemtmc market1501 resnet50 2
```

#### Stage II: End-to-end training with MMT-500

```shell
sh scripts/train.sh dukemtmc market1501 resnet50 500
```

### Test
We utilize 1 GTX-1080TI GPU for testing.
Test the trained model with best performance by
```shell
sh scripts/test.sh market1501 resnet50 logs/dukemtmcTOmarket1501/resnet50-MMT-500/model_best.pth.tar
```

## Other Examples:
**Duke-to-Market (IBN-ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain.sh dukemtmc market1501 resnet_ibn50a 1
sh scripts/pretrain.sh dukemtmc market1501 resnet_ibn50a 2
# end-to-end training with MMT-500
sh scripts/train.sh dukemtmc market1501 resnet_ibn50a 500
# or MMT-700
sh scripts/train.sh dukemtmc market1501 resnet_ibn50a 700
# testing the best model
sh scripts/test.sh market1501 resnet_ibn50a logs/dukemtmcTOmarket1501/resnet_ibn50a-MMT-500/model_best.pth.tar
sh scripts/test.sh market1501 resnet_ibn50a logs/dukemtmcTOmarket1501/resnet_ibn50a-MMT-700/model_best.pth.tar
```
**Duke-to-MSMT (ResNet-50)**
```shell
# pre-training on the source domain
sh scripts/pretrain.sh dukemtmc msmt17 resnet50 1
sh scripts/pretrain.sh dukemtmc msmt17 resnet50 2
# end-to-end training with MMT-500
sh scripts/train.sh dukemtmc msmt17 resnet50 500
# or MMT-1000
sh scripts/train.sh dukemtmc msmt17 resnet50 1000
# testing the best model
sh scripts/test.sh msmt17 resnet50 logs/dukemtmcTOmsmt17/resnet50-MMT-500/model_best.pth.tar
sh scripts/test.sh msmt17 resnet50 logs/dukemtmcTOmsmt17/resnet50-MMT-1000/model_best.pth.tar
```


## Download Trained Models
*Source-domain pre-trained models and our MMT models can be downloaded from the [link](https://drive.google.com/open?id=1WC4JgbkaAr40uEew_JEqjUxgKIiIQx-W).*
![results](figs/results.png)


## Citation
If you find this code useful for your research, please cite our paper
```
@inproceedings{
  ge2020mutual,
  title={Mutual Mean-Teaching: Pseudo Label Refinery for Unsupervised Domain Adaptation on Person Re-identification},
  author={Yixiao Ge and Dapeng Chen and Hongsheng Li},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=rJlnOhVYPS}
}
```