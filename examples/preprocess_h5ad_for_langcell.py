#!/usr/bin/env python
"""
Preprocess h5ad files for LangCell Zero-shot Cell Type Annotation

Usage:
    # Basic usage (cell types should already be standardized in h5ad file)
    python preprocess_h5ad_for_langcell.py --input /path/to/input.h5ad --output /path/to/output
    
    # With standardization mapping (requires metadata_standard_mapping.py)
    python preprocess_h5ad_for_langcell.py --input /path/to/input.h5ad --output /path/to/output --apply-standardization
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
import scanpy as sc
from datasets import Dataset
from tqdm import tqdm

# Add project root to path
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from open_biomed.models.cell.langcell.langcell_utils import LangCellTranscriptomeTokenizer
from standardized_cell_type_descriptions import get_all_standardized_descriptions, DETAILED_CELL_TYPE_DESCRIPTIONS

# Try to import standardization mapping (optional)
try:
    sys.path.insert(0, '/home/scbjtfy/RVQ-Alpha/data_process/metadata_standard')
    from metadata_standard_mapping import CELL_TYPE_MAPPING
    STANDARDIZATION_AVAILABLE = True
except ImportError:
    CELL_TYPE_MAPPING = None
    STANDARDIZATION_AVAILABLE = False


def standardize_cell_types(cell_types, apply_standardization=True):
    """
    Standardize cell type names using the metadata_standard_mapping
    
    Args:
        cell_types: List of cell type names
        apply_standardization: Whether to apply standardization mapping
    
    Returns:
        list: Standardized cell type names
        dict: Mapping of original to standardized names (for tracking)
    """
    if not apply_standardization or not STANDARDIZATION_AVAILABLE:
        print("Standardization mapping not applied (use --apply-standardization to enable)")
        return cell_types, {ct: ct for ct in cell_types}
    
    standardized = []
    mapping_used = {}
    unmapped = []
    
    for ct in cell_types:
        if ct in CELL_TYPE_MAPPING:
            std_ct = CELL_TYPE_MAPPING[ct]
            standardized.append(std_ct)
            mapping_used[ct] = std_ct
        else:
            standardized.append(ct)
            mapping_used[ct] = ct
            unmapped.append(ct)
    
    if unmapped:
        print(f"Warning: {len(set(unmapped))} cell types not found in standardization mapping:")
        for ct in sorted(set(unmapped))[:10]:
            print(f"  - {ct}")
        if len(set(unmapped)) > 10:
            print(f"  ... and {len(set(unmapped)) - 10} more")
    
    return standardized, mapping_used


def tokenize_anndata(adata, tokenizer):
    """
    Tokenize AnnData using LangCellTranscriptomeTokenizer
    
    Args:
        adata: AnnData object
        tokenizer: LangCellTranscriptomeTokenizer instance
    
    Returns:
        list: Tokenized cells (list of token lists)
    """
    print("Starting tokenization...")
    
    # Ensure data format is correct
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    
    # Add filter_pass column (mark all cells as passed QC)
    if 'filter_pass' not in adata.obs.columns:
        adata.obs['filter_pass'] = 1
    
    # Add n_counts column if not present
    if 'n_counts' not in adata.obs.columns:
        adata.obs['n_counts'] = adata.X.sum(axis=1)
    
    # Tokenize using tokenizer
    tokenized_cells, _ = tokenizer.tokenize_anndata(adata)
    
    print(f"Tokenization complete: {len(tokenized_cells)} cells")
    return tokenized_cells


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Preprocess h5ad files for LangCell Zero-shot Cell Type Annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (assumes cell types are already standardized)
  python preprocess_h5ad_for_langcell.py \\
    --input /path/to/input.h5ad \\
    --output /path/to/output
  
  # With cell type standardization (requires metadata_standard_mapping.py)
  python preprocess_h5ad_for_langcell.py \\
    --input /path/to/input.h5ad \\
    --output /path/to/output \\
    --apply-standardization
  
  # With custom Geneformer directory
  python preprocess_h5ad_for_langcell.py \\
    --input /path/to/input.h5ad \\
    --output /path/to/output \\
    --geneformer-dir /custom/path/to/Geneformer
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input h5ad file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--apply-standardization',
        action='store_true',
        help='Apply cell type name standardization using metadata_standard_mapping.py'
    )
    
    parser.add_argument(
        '--geneformer-dir',
        type=str,
        default='/home/scbjtfy/Geneformer',
        help='Path to Geneformer directory (default: /home/scbjtfy/Geneformer)'
    )
    
    parser.add_argument(
        '--cell-type-col',
        type=str,
        default='cell_type',
        help='Column name for cell type labels in adata.obs (default: cell_type)'
    )
    
    parser.add_argument(
        '--nproc',
        type=int,
        default=1,
        help='Number of processes for tokenization (default: 1)'
    )
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 80)
    print("LangCell Data Preprocessing Pipeline (Standardized)")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Apply standardization: {args.apply_standardization}")
    if args.apply_standardization and not STANDARDIZATION_AVAILABLE:
        print("WARNING: Standardization requested but metadata_standard_mapping.py not available!")
    print("=" * 80)
    
    # ===== Step 1: Load data =====
    print(f"\nStep 1: Loading h5ad file...")
    adata = sc.read_h5ad(args.input)
    print(f"✓ Loaded: {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Check required columns
    if args.cell_type_col not in adata.obs.columns:
        raise ValueError(
            f"Cell type column '{args.cell_type_col}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    if 'ensembl_id' not in adata.var.columns:
        raise ValueError(
            "Column 'ensembl_id' not found in adata.var. "
            "Geneformer tokenizer requires Ensembl gene IDs."
        )
    
    # ===== Step 2: Initialize tokenizer =====
    print(f"\nStep 2: Initializing Geneformer tokenizer...")
    
    # Load Geneformer dictionaries
    token_dict_path = os.path.join(args.geneformer_dir, "geneformer/token_dictionary_gc104M.pkl")
    gene_median_path = os.path.join(args.geneformer_dir, "geneformer/gene_median_dictionary_gc104M.pkl")
    
    if not os.path.exists(token_dict_path):
        raise FileNotFoundError(f"Token dictionary not found: {token_dict_path}")
    if not os.path.exists(gene_median_path):
        raise FileNotFoundError(f"Gene median dictionary not found: {gene_median_path}")
    
    # Create tokenizer
    tokenizer = LangCellTranscriptomeTokenizer(
        custom_attr_name_dict=None,
        nproc=args.nproc,
        gene_median_file=gene_median_path,
        token_dictionary_file=token_dict_path
    )
    print("✓ Tokenizer initialized")
    
    # ===== Step 3: Tokenization =====
    print(f"\nStep 3: Tokenizing cells...")
    tokenized_cells = tokenize_anndata(adata, tokenizer)
    
    # ===== Step 4: Standardize cell types (optional) =====
    original_cell_types = adata.obs[args.cell_type_col].tolist()
    unique_original_types = sorted(set(original_cell_types))
    print(f"\nStep 4: Cell type standardization...")
    print(f"Found {len(unique_original_types)} unique original cell types in data")
    
    # Apply standardization if requested
    standardized_cell_types, standardization_mapping = standardize_cell_types(
        original_cell_types, 
        apply_standardization=args.apply_standardization
    )
    
    # Update adata with standardized names
    if args.apply_standardization and STANDARDIZATION_AVAILABLE:
        adata.obs[f'{args.cell_type_col}_original'] = original_cell_types
        adata.obs[args.cell_type_col] = standardized_cell_types
        print(f"✓ Applied standardization mapping")
        print(f"  Unique standardized cell types: {len(set(standardized_cell_types))}")
    
    # ===== Step 5: Prepare cell type descriptions =====
    print(f"\nStep 5: Preparing cell type descriptions...")
    
    # Get unique standardized cell types
    unique_cell_types = sorted(set(adata.obs[args.cell_type_col].tolist()))
    
    # Get descriptions for all cell types using the unified system
    cell_type_descriptions = get_all_standardized_descriptions(unique_cell_types)
    
    print(f"✓ Generated descriptions for {len(cell_type_descriptions)} cell types")
    
    # Save type2text mapping
    type2text_path = os.path.join(args.output, "type2text.json")
    with open(type2text_path, 'w', encoding='utf-8') as f:
        json.dump(cell_type_descriptions, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved cell type descriptions: {type2text_path}")
    print(f"✓ Total: {len(cell_type_descriptions)} cell types")
    
    # Save standardization mapping if applied
    if args.apply_standardization and STANDARDIZATION_AVAILABLE:
        standardization_stats = {
            "original_to_standardized": {k: v for k, v in standardization_mapping.items() if k != v},
            "total_original_types": len(unique_original_types),
            "total_standardized_types": len(set(standardized_cell_types)),
            "unchanged_types": sum(1 for k, v in standardization_mapping.items() if k == v)
        }
        standardization_path = os.path.join(args.output, "standardization_mapping.json")
        with open(standardization_path, 'w', encoding='utf-8') as f:
            json.dump(standardization_stats, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved standardization mapping: {standardization_path}")
    
    # ===== Step 6: Create Dataset =====
    print(f"\nStep 6: Creating Hugging Face Dataset...")
    
    dataset_dict = {
        'input_ids': tokenized_cells,
        'str_labels': adata.obs[args.cell_type_col].tolist()
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save dataset
    dataset_path = os.path.join(args.output, "dataset")
    dataset.save_to_disk(dataset_path)
    print(f"✓ Saved dataset: {dataset_path}")
    
    # ===== Step 7: Generate statistics =====
    print(f"\nStep 7: Generating statistics...")
    
    token_lengths = [len(cell) for cell in tokenized_cells]
    
    stats = {
        "input_file": args.input,
        "standardization_applied": args.apply_standardization and STANDARDIZATION_AVAILABLE,
        "total_cells": len(tokenized_cells),
        "total_genes": adata.n_vars,
        "total_cell_types": len(unique_cell_types),
        "cell_type_distribution": adata.obs[args.cell_type_col].value_counts().to_dict(),
        "tokenized_cell_length_stats": {
            "min": int(min(token_lengths)),
            "max": int(max(token_lengths)),
            "mean": float(np.mean(token_lengths)),
            "median": float(np.median(token_lengths)),
            "std": float(np.std(token_lengths))
        },
        "detailed_descriptions": len([ct for ct in unique_cell_types if ct in DETAILED_CELL_TYPE_DESCRIPTIONS]),
        "simple_descriptions": len([ct for ct in unique_cell_types if ct not in DETAILED_CELL_TYPE_DESCRIPTIONS])
    }
    
    # Add standardization stats if applied
    if args.apply_standardization and STANDARDIZATION_AVAILABLE:
        stats["standardization_stats"] = {
            "original_cell_types": len(unique_original_types),
            "standardized_cell_types": len(unique_cell_types),
            "types_changed": len([k for k, v in standardization_mapping.items() if k != v])
        }
    
    stats_path = os.path.join(args.output, "stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved statistics: {stats_path}")
    
    # ===== Complete =====
    print("\n" + "=" * 80)
    print("✓ Preprocessing Complete!")
    print("=" * 80)
    print(f"\nOutput directory: {args.output}")
    print(f"Files created:")
    print(f"  - dataset/         : Hugging Face Dataset (tokenized cells)")
    print(f"  - type2text.json   : Cell type descriptions")
    print(f"  - stats.json       : Dataset statistics")
    
    print(f"\nDataset Statistics:")
    print(f"  - Total cells: {stats['total_cells']}")
    print(f"  - Total genes: {stats['total_genes']}")
    print(f"  - Cell types: {stats['total_cell_types']}")
    print(f"  - Token length: {stats['tokenized_cell_length_stats']['min']} - {stats['tokenized_cell_length_stats']['max']}")
    print(f"  - Token length (mean ± std): {stats['tokenized_cell_length_stats']['mean']:.1f} ± {stats['tokenized_cell_length_stats']['std']:.1f}")
    print(f"  - Detailed descriptions: {stats['detailed_descriptions']}")
    print(f"  - Simple descriptions: {stats['simple_descriptions']}")
    
    if args.apply_standardization and STANDARDIZATION_AVAILABLE:
        print(f"\nStandardization Statistics:")
        print(f"  - Original cell types: {stats['standardization_stats']['original_cell_types']}")
        print(f"  - Standardized cell types: {stats['standardization_stats']['standardized_cell_types']}")
        print(f"  - Types changed: {stats['standardization_stats']['types_changed']}")
    
    print(f"\nNext steps:")
    print(f"  1. Review cell type descriptions in type2text.json")
    if args.apply_standardization and STANDARDIZATION_AVAILABLE:
        print(f"  2. Review standardization mapping in standardization_mapping.json")
        print(f"  3. Run prediction with LangCell model")
    else:
        print(f"  2. Run prediction with LangCell model")
    print(f"  - Data directory: {args.output}")


if __name__ == "__main__":
    main()
