#!/bin/bash
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -y scipy==1.9.1
conda install -y scikit-learn==1.1.1
conda install -y pandas==1.5.1
pip install faiss-gpu==1.7.2
pip install prettytable==3.8.0 tabulate==0.9.0 natsort==8.4.0 pyyaml==6.0.1 tqdm==4.66.1