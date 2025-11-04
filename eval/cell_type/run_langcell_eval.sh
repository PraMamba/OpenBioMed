#!/bin/bash
# Run LangCell Evaluation Script
#
# This script evaluates LangCell model on preprocessed datasets.
# Requires Axolotl conda environment.
#
# Usage:
#   ./run_langcell_eval.sh A013    # Evaluate on A013 dataset
#   ./run_langcell_eval.sh D099    # Evaluate on D099 dataset
#   ./run_langcell_eval.sh A013 8 cuda:1  # Custom batch size and device

set -eu

# === Environment Setup ===
source ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Axolotl

cd ~/OpenBioMed

# ============================================================================
# Configuration
# ============================================================================
DATASET_ID="${1:-D099_sampled_0.1}"  
BASE_DATA_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/Processed_Data"
OUTPUT_DIR="/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/D099_sampled_0.1/eval_results"
BATCH_SIZE="${2:-2}"  # Default batch size is 4
DEVICE="${3:-cuda:5}"  # Default device
DATA_DIR="${BASE_DATA_DIR}/${DATASET_ID}"


echo "=========================================================================="
echo "LangCell Evaluation"
echo "=========================================================================="
echo "Dataset ID:   $DATASET_ID"
echo "Data Dir:     $DATA_DIR"
echo "Output Dir:   $OUTPUT_DIR"
echo "Batch Size:   $BATCH_SIZE"
echo "Device:       $DEVICE"
echo "=========================================================================="


python ./eval/cell_type/langcell_eval.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset_id "$DATASET_ID" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"

echo "=========================================================================="
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
