#!/bin/bash

# 1. install conda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

. ~/miniconda3/bin/activate

# 2. create conda env
conda env create -f ~/KVCache-Factory/environment.yml
conda clean --all

# 2.5 activate the environment
source ~/miniconda3/bin/activate
conda activate MInference

# 3. install pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 4. pip install other requirements
pip install -r requirements.txt 
pip cache purge
