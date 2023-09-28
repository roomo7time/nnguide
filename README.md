# Nearest Neighbor Guidance for Out-of-Distribution Detection

This is the official repository of the paper Nearest Neighbor Guidance for Out-of-Distribution Detection ([arxiv](https://arxiv.org/abs/2309.14888))

## Setup

### Conda
The experiments have been conducted on the following settings:
 - Ubuntu 20.04
 - CUDA 11.3

The conda environment is installed by
```
conda create -n nnguide python=3.8.13
```
and then then on the `nnguide` conda environment, the required packages are installed by
```
chmod +x install_packages.sh
./install_packages.sh
```


### Dataset

To set up dataset folder structures, refer to `README.md` in the `config` folder.

#### 1. Download ImageNet-1k:
Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` from the official ImageNet [website]()

#### 2. Download iNaturalist, SUN, Places, Textures, OpenImage-O OOD datasets:
To download iNaturalist, SUN, and Places
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
Download Textures from the official [website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
Download OpenImage-O from the official website.


### Pretrained models
Download `resnet50-supcon.pt` from the [link](https://www.dropbox.com/scl/fi/f3bfipk2o96f27vibpozb/resnet50-supcon.pt?rlkey=auxw68wcgqcx4ze6yhnmm395y&dl=0) and put it in the directory `pretrained_models`.

## Run Experiments

To run experiments, execute
```
chmod +x run.sh
./run.sh
```


## Acknowledgements
Parts of our codebase have been adopted from the official repositories for [KNN-OOD](https://github.com/deeplearning-wisc/knn-ood) and [VIM](https://github.com/haoqiwang/vim), and we benefited from the pretrained weights made available through these sources.