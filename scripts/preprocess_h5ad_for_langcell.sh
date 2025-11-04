#!/bin/bash
set -eu

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl

cd ~/OpenBioMed/examples 

# Run preprocessing with standardization
# Note: The input h5ad file should have cell_type metadata that can be standardized
python preprocess_h5ad_for_langcell.py \
  --input /gpfs/Mamba/Project/Single_Cell/Benchmark/Cell_Type/Cell2Sentence/Processed_Data/D096_subset_processed_w_cell2sentence.h5ad \
  --output /gpfs/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/Processed_Data/D096 \
  --apply-standardization