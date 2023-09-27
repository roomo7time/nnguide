# Nearest Neighbor Guidance for Out-of-Distribution Detection

Paper: Nearest Neighbor Guidance for Out-of-Distribution Detection ([arxiv](https://arxiv.org/abs/2309.14888))

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

To set up datasets, refer to `README.md` in the `config` folder.

### Pretrained models
Download `resnet50-supcon.pt` from the [link](https://www.dropbox.com/scl/fi/f3bfipk2o96f27vibpozb/resnet50-supcon.pt?rlkey=auxw68wcgqcx4ze6yhnmm395y&dl=0) and put it in the directory `pretrained_models`.


## Acknowledgements
Parts of our codebase have been adopted from the official repositories for [KNN-OOD](https://github.com/deeplearning-wisc/knn-ood) and [VIM](https://github.com/haoqiwang/vim), and we benefited from the pretrained weights made available through these sources.