"""
LangCell Zero-shot Cell Type Annotation Evaluation Script

This script evaluates LangCell model on cell type annotation tasks.

Usage:
    python langcell_eval.py \
        --data_dir /path/to/Processed_Data/A013 \
        --output_dir /path/to/output \
        --dataset_id A013 \
        --batch_size 4
"""

import os
import sys
import json
import argparse
from typing import List, Dict
from datetime import datetime
from tqdm import tqdm
import torch

# Add OpenBioMed root to path
# The file is at: OpenBioMed/eval/cell_type/langcell_eval.py
# We need to add: OpenBioMed/ to sys.path
eval_dir = os.path.dirname(os.path.abspath(__file__))  # .../eval/cell_type
parent_eval = os.path.dirname(eval_dir)  # .../eval
openbiomed_root = os.path.dirname(parent_eval)  # .../OpenBioMed
sys.path.insert(0, openbiomed_root)

from open_biomed.core.pipeline import InferencePipeline
from open_biomed.data import Cell, Text
from datasets import load_from_disk

from celltype_standardizer import CellTypeStandardizer, save_unmapped_report


def load_langcell_data(data_dir: str) -> tuple:
    """
    Load preprocessed LangCell dataset and cell type descriptions.
    
    Args:
        data_dir: Path to processed data directory
        
    Returns:
        Tuple of (dataset, type2text)
    """
    print(f"[INFO] Loading dataset from: {data_dir}")
    
    dataset_path = os.path.join(data_dir, "dataset")
    type2text_path = os.path.join(data_dir, "type2text.json")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not os.path.exists(type2text_path):
        raise FileNotFoundError(f"type2text.json not found: {type2text_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
    except Exception as e:
        print(f"[WARNING] Failed to load with load_from_disk: {e}")
        print("[INFO] Attempting alternative loading method...")
        # Try loading directly from arrow file
        import pyarrow as pa
        from datasets import Dataset
        arrow_file = os.path.join(dataset_path, "data-00000-of-00001.arrow")
        if os.path.exists(arrow_file):
            table = pa.ipc.open_file(arrow_file).read_all()
            dataset = Dataset(table)
            print(f"[INFO] Successfully loaded from arrow file")
        else:
            raise FileNotFoundError(f"Could not find arrow file: {arrow_file}")
    
    with open(type2text_path, 'r', encoding='utf-8') as f:
        type2text = json.load(f)
    
    print(f"[INFO] Loaded {len(dataset)} cells with {len(type2text)} cell types")
    return dataset, type2text


def prepare_langcell_inputs(dataset, type2text: Dict[str, str]) -> Dict:
    """
    Prepare inputs for LangCell model.
    
    Args:
        dataset: Hugging Face Dataset with tokenized cells
        type2text: Dictionary of cell type to description mapping
        
    Returns:
        Dictionary with 'cell', 'class_texts', and 'label' lists
    """
    print("[INFO] Preparing model inputs...")
    
    # Prepare text descriptions
    texts = []
    type2label = {}
    for cell_type in type2text:
        texts.append(Text.from_str(type2text[cell_type]))
        type2label[cell_type] = len(texts) - 1
    
    # Prepare inputs
    inputs = {'cell': [], 'class_texts': [], 'label': []}
    labels_str = []
    
    for data in dataset:
        inputs['cell'].append(Cell.from_sequence(data['input_ids']))
        inputs['class_texts'].append(texts)
        inputs['label'].append(type2label[data['str_labels']])
        labels_str.append(data['str_labels'])
    
    print(f"[INFO] Prepared {len(inputs['cell'])} cells")
    return inputs, labels_str, type2label


def run_langcell_inference(pipeline, inputs: Dict, batch_size: int = 4) -> List[int]:
    """
    Run LangCell inference.
    
    Args:
        pipeline: LangCell inference pipeline
        inputs: Prepared inputs dictionary
        batch_size: Batch size for inference
        
    Returns:
        List of predicted class indices
    """
    print(f"[INFO] Running inference with batch_size={batch_size}...")
    
    try:
        preds, _ = pipeline.run(batch_size=batch_size, **inputs)
        preds = [p.item() for p in preds]
        print(f"[INFO] Inference complete: {len(preds)} predictions")
        return preds
    
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        raise


def calculate_metrics(predictions: List[str], ground_truths: List[str]) -> Dict:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions: List of predicted cell types
        ground_truths: List of ground truth cell types
        
    Returns:
        Dictionary with metrics
    """
    total = len(predictions)
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) 
                  if pred.lower().strip() == gt.lower().strip())
    accuracy = correct / total if total > 0 else 0.0
    
    metrics = {
        "task_type": "cell type",
        "task_variant": "langcell_singlecell",
        "total_cells": total,
        "correct_predictions": correct,
        "accuracy": accuracy
    }
    
    return metrics


def run_evaluation(
    data_dir: str,
    output_dir: str,
    dataset_id: str = "unknown",
    batch_size: int = 4,
    device: str = "cuda:0"
):
    """
    Run LangCell evaluation on a dataset.
    
    Args:
        data_dir: Path to preprocessed data directory
        output_dir: Directory to save results
        dataset_id: Dataset identifier (e.g., 'A013', 'D099')
        batch_size: Batch size for inference
        device: Device to use for inference
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    dataset, type2text = load_langcell_data(data_dir)
    
    # Initialize cell type standardizer
    standardizer = CellTypeStandardizer()
    
    # Prepare inputs
    inputs, labels_str, type2label = prepare_langcell_inputs(dataset, type2text)
    
    # Create reverse mapping from label to cell type
    label2type = {v: k for k, v in type2label.items()}
    
    # Initialize LangCell pipeline
    print(f"[INFO] Loading LangCell model on {device}...")
    pipeline = InferencePipeline(
        model='langcell',
        task='cell_annotation',
        device=device
    )
    
    # Run inference
    pred_indices = run_langcell_inference(pipeline, inputs, batch_size)
    
    # Convert indices to cell type names
    pred_types_raw = [label2type[idx] for idx in pred_indices]
    
    # Standardize cell type names
    print("[INFO] Standardizing cell type names...")
    results = []
    predictions_std = []
    ground_truths_std = []
    unmapped_records = []
    
    for idx, (pred_raw, gt_raw) in enumerate(zip(pred_types_raw, labels_str)):
        # Standardize cell types
        gt_std, gt_is_mapped = standardizer.standardize_single_celltype(gt_raw)
        pred_std, pred_is_mapped = standardizer.standardize_single_celltype(pred_raw)
        
        # Track unmapped types
        if not gt_is_mapped and gt_raw:
            unmapped_records.append({
                "index": idx,
                "source": "ground_truth",
                "original_type": gt_raw,
                "full_answer": gt_raw
            })
        if not pred_is_mapped and pred_raw:
            unmapped_records.append({
                "index": idx,
                "source": "predicted_answer",
                "original_type": pred_raw,
                "full_answer": pred_raw
            })
        
        # Get LangCell's dual inputs - complete actual model inputs
        # Note: Actual model input includes CLS token prepended by DataCollator
        tokenized_input = dataset[idx]['input_ids']
        
        # Get CLS token ID (same as used in DataCollator)
        try:
            from open_biomed.models.cell.langcell._geneformer_compat import token_dictionary
            cls_token_id = token_dictionary.get("<cls>")
        except:
            # Fallback: try to get from pipeline's collator if available
            cls_token_id = None
        
        # Convert token IDs to list format
        if hasattr(tokenized_input, 'tolist'):
            token_ids_list = tokenized_input.tolist()
        elif isinstance(tokenized_input, (list, tuple)):
            token_ids_list = list(tokenized_input)
        else:
            token_ids_list = list(tokenized_input)
        
        # Add CLS token at the beginning (as DataCollator does) to get actual model input
        if cls_token_id is not None:
            actual_model_input = [cls_token_id] + token_ids_list[:2047]  # Max length 2048 (CLS + 2047 tokens)
        else:
            # If CLS token ID not available, use original input (should not happen in practice)
            actual_model_input = token_ids_list
        
        # Get all candidate cell type descriptions (complete text descriptions)
        candidate_descriptions = []
        for cell_type in sorted(type2text.keys()):  # Sort for consistent ordering
            candidate_descriptions.append(type2text[cell_type])
        
        # Format question with complete actual model inputs
        # 1. Complete tokenized gene expression (all token IDs including CLS token)
        # 2. Complete list of all candidate cell type text descriptions
        question_dict = {
            "cell_input": {
                "tokenized_genes": actual_model_input,
                "num_tokens": len(actual_model_input),
                "note": "Includes CLS token prepended by DataCollator (actual model input)"
            },
            "candidate_types": candidate_descriptions,
            "num_candidates": len(candidate_descriptions)
        }
        
        # Convert to JSON string for question field
        input_repr = json.dumps(question_dict, ensure_ascii=False, indent=2)
        
        # Create result item matching target format
        result_item = {
            "model_name": "LangCell",
            "dataset_id": dataset_id,
            "index": idx,
            "task_type": "cell type",
            "task_variant": "singlecell_openended",
            "question": input_repr,  # Complete actual model inputs: full token IDs + all candidate descriptions
            "ground_truth": gt_std,
            "predicted_answer": pred_std,
            "full_response": pred_raw,  # Original predicted type before standardization
            "group": ""  # LangCell doesn't have group concept
        }
        results.append(result_item)
        predictions_std.append(pred_std)
        ground_truths_std.append(gt_std)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions_std, ground_truths_std)
    
    # Print summary
    print("\n" + "="*60)
    print("LANGCELL EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset ID:          {dataset_id}")
    print(f"Total Cells:         {metrics['total_cells']}")
    print(f"Correct Predictions: {metrics['correct_predictions']}")
    print(f"Accuracy:            {metrics['accuracy']:.2%}")
    print("="*60)
    
    # Save predictions
    predictions_file = os.path.join(
        output_dir,
        f"langcell_{dataset_id}_predictions_{timestamp}.json"
    )
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Predictions saved to: {predictions_file}")
    
    # Save metrics
    metrics_file = os.path.join(
        output_dir,
        f"langcell_{dataset_id}_metrics_{timestamp}.json"
    )
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to: {metrics_file}")
    
    # Save unmapped cell types report
    save_unmapped_report(
        unmapped_records,
        output_dir,
        f"langcell_{dataset_id}",
        timestamp
    )
    
    print("\n[INFO] Evaluation complete!")
    return metrics, results


def main():
    parser = argparse.ArgumentParser(
        description="LangCell Zero-shot Cell Type Annotation Evaluation"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to preprocessed data directory (contains dataset/ and type2text.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default="unknown",
        help="Dataset identifier (e.g., 'A013', 'D099')"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (default: 4)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (default: cuda:0)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    run_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_id=args.dataset_id,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == "__main__":
    main()

