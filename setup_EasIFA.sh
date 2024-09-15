#!/bin/bash

echo "Automatically configuring the EasIFA environment..."

# 设置环境变量
export CONDA_HOME=$(conda info --base)
export EASIFA_ROOT=$(pwd)

# 安装 gdown 工具
pip install gdown

# 创建需要的目录
mkdir -vp ~/.cache/torch/hub/checkpoints
mkdir -vp $CONDA_HOME/envs
mkdir -vp $EASIFA_ROOT/dataset/raw_dataset/chebi

# 检查并下载文件：ESM 模型
if [ ! -f ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt ]; then
    echo "Downloading ESM model..."
    wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt -O ~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
else
    echo "ESM model already exists, skipping download."
fi

# 检查并下载 easifa_env.tar.gz
if [ ! -f $CONDA_HOME/envs/easifa_env.tar.gz ]; then
    echo "Downloading easifa_env.tar.gz..."
    gdown https://drive.google.com/uc?id=1kLIx2h6kPk6OvRVPWWqJvoKAYVvBx80x -O $CONDA_HOME/envs/easifa_env.tar.gz
else
    echo "easifa_env.tar.gz already exists, skipping download."
fi

# 检查并下载 checkpoints.zip
if [ ! -f $EASIFA_ROOT/checkpoints.zip ]; then
    echo "Downloading checkpoints.zip..."
    gdown https://drive.google.com/uc?id=1ra11M4PpIalKx9ZZP-mrgj13IuFakjz3 -O $EASIFA_ROOT/checkpoints.zip
else
    echo "checkpoints.zip already exists, skipping download."
fi

# 解压 checkpoints.zip 文件
if [ -f $EASIFA_ROOT/checkpoints.zip ]; then
    echo "Unzipping checkpoints..."
    unzip -o $EASIFA_ROOT/checkpoints.zip
fi

# 解压 easifa_env.tar.gz 文件
if [ ! -d $CONDA_HOME/envs/easifa_env ]; then
    echo "Extracting easifa_env.tar.gz..."
    mkdir $CONDA_HOME/envs/easifa_env
    tar -xvf $CONDA_HOME/envs/easifa_env.tar.gz -C $CONDA_HOME/envs/easifa_env
fi

# 激活 conda 环境
source activate $CONDA_HOME/envs/easifa_env
conda unpack

# 安装 Python 包
pip uninstall fair-esm -y
pip install git+https://github.com/facebookresearch/esm.git
pip install mysql-connector-python==8.2.0 mysqlclient==2.2.1 rxnfp flask_wtf

# 检查并下载 structures.csv.gz
if [ ! -f $EASIFA_ROOT/dataset/raw_dataset/chebi/structures.csv.gz ]; then
    echo "Downloading structures.csv.gz..."
    wget https://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/structures.csv.gz -O $EASIFA_ROOT/dataset/raw_dataset/chebi/structures.csv.gz
else
    echo "structures.csv.gz already exists, skipping download."
fi

echo "EasIFA setup done!"
