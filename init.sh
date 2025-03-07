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

# 3. install pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. pip install other requirements
pip install -r requirements.txt 
pip cache purge

# 5. bulid flash-attn from source code
cd /workspace/flash-attention
MAX_JOBS=4 python setup.py install