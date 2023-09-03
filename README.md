# Official Implementation of NNGuide

Paper: Nearest Neighbor Guidance for Out-of-Distribution Detection (arxiv)

## Setup

### Conda
The experiments have been conducted on the following settings:
 - Ubuntu 20.04
 - CUDA 11.3

The conda environment is installed by
```
conda create -n nnguide python=3.8.13
```
and then then on the `nnguide` conda environment, the following packages are installed
```
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -y scipy==1.9.1
conda install -y scikit-learn==1.1.1
conda install -y pandas==1.5.1
pip install faiss-gpu==1.7.2
pip install prettytable==3.8.0 tabulate==0.9.0 natsort==8.4.0 pyyaml==6.0.1 tqdm==4.66.1
```

(auto)
```
conda env create -f environment.yml
```

### Dataset

To set up datasets, refer to `README.md` in the `config` folder.

### Pretrained models
Download `resnet50-supcon.pt` from the [link](https://www.dropbox.com/scl/fi/f3bfipk2o96f27vibpozb/resnet50-supcon.pt?rlkey=auxw68wcgqcx4ze6yhnmm395y&dl=0) and put it in the directory `pretrained_models`.


## Acknowledgements
Parts of our codebase have been adopted from the official repositories for [KNN-OOD](https://github.com/deeplearning-wisc/knn-ood) and [VIM](https://github.com/haoqiwang/vim), and we benefited from the pretrained weights made available through these sources.