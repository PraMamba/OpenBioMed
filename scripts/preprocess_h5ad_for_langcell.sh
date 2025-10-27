#!/bin/bash


cd /home/scbjtfy/OpenBioMed/examples 

conda run -n Axolotl python preprocess_h5ad_for_langcell.py \
  --input /data/Mamba/Project/Single_Cell/Benchmark/Cell2Setence/Data/D099_processed_w_cell2sentence.h5ad \
  --output /data/Mamba/Project/Single_Cell/Benchmark/LangCell/data_D099 \
  --dataset D099