# Nearest Neighbor Guidance for Out-of-Distribution Detection

This is the official repository of the ICCV2023 paper Nearest Neighbor Guidance for Out-of-Distribution Detection ([arxiv](https://arxiv.org/abs/2309.14888))

## Setup

### Conda
The experiments have been conducted on the following settings:
 - Ubuntu 20.04
 - CUDA 11.3

The conda environment is installed by
```
conda create -n nnguide python=3.8.13
```
and then on the `nnguide` conda environment, the required packages are installed by
```
chmod +x install_packages.sh
./install_packages.sh
```


### Dataset

To set up dataset folder structures, refer to `README.md` in the `./dataloaders` folder.

#### 1. Download ImageNet-1k:
Download `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` from the official ImageNet [website](). And use `./dataloaders/assets/extract_ILSVRC.sh` to unzip the zip files.

#### 2. Download iNaturalist, SUN, Places, Textures, OpenImage-O OOD datasets:
To download iNaturalist, SUN, and Places
```
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/SUN.tar.gz
wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/Places.tar.gz
```
Download Textures from the official [website](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
Download OpenImage-O from the official [website](https://github.com/haoqiwang/vim/tree/master/datalists).

#### Debug: CIFAR, SVHN
The datasets CIFAR10/100 and SVHN are provided only for the debugging purpose.

### Pretrained models
Download `resnet50-supcon.pt` from the [link](https://www.dropbox.com/scl/fi/f3bfipk2o96f27vibpozb/resnet50-supcon.pt?rlkey=auxw68wcgqcx4ze6yhnmm395y&dl=0) and put it in the directory `pretrained_models` as `./pretrained_models/resnet50-supcon.py`.

To fully reproduce the reported results, download `saved_model_outputs` from the [link](https://www.dropbox.com/scl/fi/fk6a51cz4jsx83h6qpiar/saved_model_outputs.zip?rlkey=xfcyo1mntv8vrf74as1ono138&dl=0) and save it with the path `./saved_model_outputs`.

## Run Experiments

To run experiments, execute
```
chmod +x run.sh
./run.sh
```

## Visualization
<table>
  <tr>
    <td align="center">
      <img src="assets/knn.jpg" alt="KNN" width="250"/><br/>
      <sub>KNN</sub>
    </td>
    <td align="center">
      <img src="assets/classifier.jpg" alt="Classifier" width="250"/><br/>
      <sub>Classifier</sub>
    </td>
    <td align="center">
      <img src="assets/nnguide.jpg" alt="NNGuide" width="250"/><br/>
      <sub>NNGuide</sub>
    </td>
  </tr>
</table>

## Acknowledgements
Parts of our codebase have been adopted from the official repositories for [KNN-OOD](https://github.com/deeplearning-wisc/knn-ood) and [VIM](https://github.com/haoqiwang/vim), and we benefited from the pretrained weights made available through these sources. Our code style is largely inspired by [OpenOOD](https://github.com/Jingkang50/OpenOOD).


## Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@inproceedings{park2023nearest,
  title={Nearest Neighbor Guidance for Out-of-Distribution Detection},
  author={Park, Jaewoo and Jung, Yoon Gyo and Teoh, Andrew Beng Jin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1686--1695},
  year={2023}
}
```