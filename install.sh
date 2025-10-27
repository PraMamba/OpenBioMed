#!/bin/bash
set -e  # 出错即退出
echo ">>> [1/8] 激活 Conda 环境：Axolotl"
source ~/anaconda3/etc/profile.d/conda.sh

echo ">>> [2/8] 安装 OpenBioMed (editable 模式)"
cd ~/OpenBioMed
pip install -e . --no-deps

echo ">>> [3/8] 安装依赖包：rdkit、torch_geometric、selfies"
pip install rdkit torch_geometric selfies

echo ">>> [4/8] 安装 torch_scatter（匹配 CUDA 12.6 + Torch 2.7.0）"
cd ~
wget -nc https://data.pyg.org/whl/torch-2.7.0%2Bcu126/torch_scatter-2.1.2%2Bpt27cu126-cp311-cp311-linux_x86_64.whl
pip install ./torch_scatter-2.1.2+pt27cu126-cp311-cp311-linux_x86_64.whl

echo ">>> [5/8] 克隆 Geneformer 仓库"
if [ ! -d ~/Geneformer ]; then
    git clone https://huggingface.co/ctheodoris/Geneformer ~/Geneformer
else
    echo "Geneformer 目录已存在，跳过克隆。"
fi

echo ">>> [6/8] 安装 Geneformer"
cd ~/Geneformer
pip install . --no-deps

echo ">>> [7/8] 测试安装"
python -c "import openbiomed, geneformer; print('✅ OpenBioMed + Geneformer 导入成功！')"

echo ">>> [8/8] 安装完成 🎉"
