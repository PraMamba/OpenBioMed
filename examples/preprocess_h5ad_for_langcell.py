#!/usr/bin/env python
"""
Preprocess h5ad files for LangCell Zero-shot Cell Type Annotation

Usage:
    python preprocess_h5ad_for_langcell.py --input /path/to/input.h5ad --output /path/to/output --dataset A013
    python preprocess_h5ad_for_langcell.py --input /path/to/input.h5ad --output /path/to/output --dataset D099
    python preprocess_h5ad_for_langcell.py --input /path/to/input.h5ad --output /path/to/output --dataset custom
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

def get_pbmc_cell_type_descriptions():
    """
    Get detailed cell type descriptions for PBMC (Peripheral Blood Mononuclear Cell) datasets
    Based on Cell Ontology (CL) with enhanced biological context
    """
    return {
        # T cells - CD4+
        "central memory CD4-positive, alpha-beta T cell": (
            "cell type: central memory cd4-positive, alpha-beta t cell. "
            "a memory t cell subset that expresses cd4 and has undergone antigen-driven clonal expansion. "
            "characterized by ccr7-positive, cd45ra-negative, cd127-positive, and cd62l-positive phenotype. "
            "these cells reside in secondary lymphoid organs and possess high proliferative capacity upon antigen re-encounter."
        ),
        "naive thymus-derived CD4-positive, alpha-beta T cell": (
            "cell type: naive thymus-derived cd4-positive, alpha-beta t cell. "
            "a cd4-positive t cell that has emigrated from the thymus but has not yet encountered its cognate antigen. "
            "characterized by ccr7-positive, cd45ra-positive, cd45ro-negative, and cd62l-high phenotype. "
            "these cells circulate through blood and lymphoid tissues seeking antigen presentation."
        ),
        "effector memory CD4-positive, alpha-beta T cell": (
            "cell type: effector memory cd4-positive, alpha-beta t cell. "
            "a cd4-positive memory t cell with immediate effector function upon antigen re-stimulation. "
            "characterized by ccr7-negative, cd45ra-negative, cd127-positive phenotype. "
            "these cells rapidly produce cytokines and migrate to peripheral tissues for immune surveillance."
        ),
        "regulatory T cell": (
            "cell type: regulatory t cell. "
            "a specialized cd4-positive t cell subset that maintains immune homeostasis and self-tolerance. "
            "characterized by cd25-high, foxp3-positive, cd127-low phenotype. "
            "these cells suppress excessive immune responses through cell contact and immunosuppressive cytokine secretion."
        ),
        "CD4-positive, alpha-beta cytotoxic T cell": (
            "cell type: cd4-positive, alpha-beta cytotoxic t cell. "
            "an atypical cd4-positive t cell with cytotoxic capabilities typically associated with cd8-positive t cells. "
            "expresses granzymes and perforin, capable of direct target cell lysis. "
            "these cells play important roles in chronic viral infections and tumor immunity."
        ),
        
        # T cells - CD8+
        "naive thymus-derived CD8-positive, alpha-beta T cell": (
            "cell type: naive thymus-derived cd8-positive, alpha-beta t cell. "
            "a cd8-positive t cell that has emigrated from the thymus without prior antigen exposure. "
            "characterized by ccr7-positive, cd45ra-positive, cd45ro-negative phenotype. "
            "these cells patrol lymphoid tissues awaiting activation by antigen-presenting cells."
        ),
        "effector memory CD8-positive, alpha-beta T cell": (
            "cell type: effector memory cd8-positive, alpha-beta t cell. "
            "a cd8-positive memory t cell with rapid cytotoxic response capability. "
            "characterized by ccr7-negative, cd45ra-negative, cd127-positive phenotype. "
            "these cells provide immediate protective immunity in peripheral tissues."
        ),
        "central memory CD8-positive, alpha-beta T cell": (
            "cell type: central memory cd8-positive, alpha-beta t cell. "
            "a long-lived cd8-positive memory t cell with high proliferative potential. "
            "characterized by ccr7-positive, cd45ra-negative, cd127-positive phenotype. "
            "these cells reside in lymphoid organs and generate robust secondary responses."
        ),
        
        # Other T cells
        "gamma-delta T cell": (
            "cell type: gamma-delta t cell. "
            "an unconventional t cell expressing gamma-delta t cell receptor instead of alpha-beta tcr. "
            "these cells recognize lipid and phosphoantigen without mhc restriction. "
            "they bridge innate and adaptive immunity, responding rapidly to tissue stress and infection."
        ),
        "mucosal invariant T cell": (
            "cell type: mucosal invariant t cell (mait cell). "
            "an innate-like t cell expressing semi-invariant t cell receptor recognizing bacterial metabolites. "
            "characterized by cd161-high, il-18r-positive phenotype and mr1 restriction. "
            "these cells provide rapid antimicrobial responses at mucosal barriers."
        ),
        "double negative thymocyte": (
            "cell type: double negative thymocyte. "
            "an immature t cell precursor in the thymus lacking both cd4 and cd8 expression. "
            "these cells undergo t cell receptor rearrangement and positive selection. "
            "represents an early stage of t cell development before lineage commitment."
        ),
        
        # Monocytes
        "CD14-positive monocyte": (
            "cell type: cd14-positive monocyte (classical monocyte). "
            "the major monocyte subset expressing high levels of cd14 and low cd16. "
            "characterized by cd14++cd16- phenotype and strong phagocytic capacity. "
            "these cells respond to pathogens, differentiate into macrophages and dendritic cells, and produce inflammatory cytokines."
        ),
        "CD14-low, CD16-positive monocyte": (
            "cell type: cd14-low, cd16-positive monocyte (non-classical monocyte). "
            "a patrolling monocyte subset with cd14+cd16++ phenotype. "
            "these cells crawl along vascular endothelium, survey tissue integrity, and respond to viral infections. "
            "they have distinct transcriptional profile and produce anti-inflammatory cytokines."
        ),
        
        # B cells
        "naive B cell": (
            "cell type: naive b cell. "
            "a mature b cell that has not encountered antigen, expressing surface igm and igd. "
            "characterized by cd19-positive, cd20-positive, cd27-negative, igd-positive phenotype. "
            "these cells circulate through blood and lymphoid organs awaiting antigen recognition for activation."
        ),
        "memory B cell": (
            "cell type: memory b cell. "
            "a long-lived b cell generated from germinal center reactions with antigen experience. "
            "characterized by cd19-positive, cd27-positive, and class-switched immunoglobulin expression. "
            "these cells provide rapid antibody responses upon antigen re-encounter."
        ),
        "transitional stage B cell": (
            "cell type: transitional stage b cell. "
            "an immature b cell that has emigrated from bone marrow undergoing peripheral maturation. "
            "characterized by cd19-positive, cd24-high, cd38-high, cd10-positive phenotype. "
            "these cells are susceptible to negative selection and tolerance induction."
        ),
        "plasmablast": (
            "cell type: plasmablast. "
            "a highly proliferative antibody-secreting cell transitioning from activated b cell to plasma cell. "
            "characterized by cd19-positive, cd20-low, cd27-high, cd38-high, cd138-positive phenotype. "
            "these cells produce large amounts of immunoglobulin and circulate in blood during acute immune responses."
        ),
        
        # NK cells
        "natural killer cell": (
            "cell type: natural killer cell. "
            "a cytotoxic lymphocyte of the innate immune system capable of killing virus-infected and tumor cells. "
            "characterized by cd56-positive, cd3-negative phenotype and expression of activating receptors (ncr, nkg2d). "
            "these cells provide rapid immune surveillance without prior sensitization through germline-encoded receptors."
        ),
        "CD16-negative, CD56-bright natural killer cell, human": (
            "cell type: cd16-negative, cd56-bright natural killer cell. "
            "a regulatory nk cell subset with cd56-bright, cd16-negative, cd94-positive phenotype. "
            "these cells are potent cytokine producers with limited cytotoxicity. "
            "they preferentially localize to secondary lymphoid tissues and regulate adaptive immune responses."
        ),
        
        # Dendritic cells
        "plasmacytoid dendritic cell": (
            "cell type: plasmacytoid dendritic cell. "
            "a specialized dendritic cell type producing large amounts of type i interferon in response to viral infection. "
            "characterized by cd123-high, cd303-positive, cd304-positive phenotype and plasmacytoid morphology. "
            "these cells link innate and adaptive immunity through antiviral responses and t cell priming."
        ),
        "conventional dendritic cell": (
            "cell type: conventional dendritic cell (myeloid dendritic cell). "
            "a professional antigen-presenting cell derived from myeloid lineage expressing high cd11c. "
            "characterized by mhc class ii-high, cd11c-high phenotype and dendritic morphology. "
            "these cells capture, process, and present antigens to t cells, initiating adaptive immune responses."
        ),
        
        # Progenitor cells
        "hematopoietic precursor cell": (
            "cell type: hematopoietic precursor cell. "
            "an immature cell with multilineage differentiation potential in the hematopoietic system. "
            "characterized by cd34-positive, cd38-low/negative phenotype and high self-renewal capacity. "
            "these cells give rise to all blood cell lineages including myeloid and lymphoid cells."
        )
    }

def get_tissue_cell_type_descriptions():
    """
    Get cell type descriptions for tissue-specific datasets (e.g., D099 - airway epithelium)
    Can be extended based on specific tissue types
    """
    # Start with PBMC descriptions as base
    descriptions = get_pbmc_cell_type_descriptions()
    
    # Add airway epithelial cell types (D099 dataset)
    airway_epithelial = {
        "Basal": (
            "cell type: basal cell of respiratory epithelium. "
            "a stem/progenitor cell residing at the basement membrane of airway epithelium. "
            "characterized by krt5-positive, tp63-positive, ngfr-positive phenotype and high proliferative capacity. "
            "these cells serve as progenitors for multiple airway epithelial lineages and maintain tissue homeostasis through self-renewal and differentiation."
        ),
        "Secretory": (
            "cell type: secretory cell of airway epithelium (club cell). "
            "a non-ciliated epithelial cell producing secretory proteins in the airways. "
            "characterized by scgb1a1-positive (club cell secretory protein), muc5b-positive phenotype. "
            "these cells secrete antimicrobial proteins, surfactant components, and regulate mucus production for airway protection and immune defense."
        ),
        "Ciliated": (
            "cell type: ciliated cell of respiratory epithelium. "
            "a differentiated epithelial cell with motile cilia on apical surface. "
            "characterized by foxj1-positive, dnah5-positive phenotype and expression of ciliary genes. "
            "these cells coordinate ciliary beating for mucociliary clearance, removing debris and pathogens from airways."
        ),
        "Goblet": (
            "cell type: goblet cell of respiratory epithelium. "
            "a specialized secretory cell producing mucus glycoproteins. "
            "characterized by muc5ac-high, muc5b-positive, spdef-positive phenotype. "
            "these cells secrete gel-forming mucins for airway lubrication and trapping of inhaled particles and pathogens."
        ),
        "Proliferating.Basal": (
            "cell type: proliferating basal cell. "
            "an actively dividing basal cell in cell cycle. "
            "characterized by krt5-positive, tp63-positive, mki67-positive, and top2a-positive phenotype. "
            "these cells undergo mitotic division for tissue regeneration and repair following airway injury."
        ),
        "Differentiating.Basal": (
            "cell type: differentiating basal cell. "
            "a basal cell undergoing differentiation toward secretory or ciliated lineage. "
            "characterized by intermediate expression of basal markers (krt5, tp63) and emerging differentiation markers. "
            "these cells represent a transitional state between basal progenitors and mature airway epithelial cells."
        ),
        "Transitioning.Basal": (
            "cell type: transitioning basal cell. "
            "a basal cell in transition between quiescent and activated states. "
            "characterized by dynamic expression patterns of basal and differentiation markers. "
            "these cells respond to environmental stimuli and injury signals to initiate regenerative programs."
        ),
        "Suprabasal": (
            "cell type: suprabasal cell. "
            "an intermediate cell type located above the basal layer during epithelial stratification. "
            "characterized by loss of basal markers and acquisition of differentiation markers. "
            "these cells represent committed progenitors undergoing terminal differentiation in stratified epithelium."
        )
    }
    
    # Add general tissue cell types
    general_tissue = {
        "endothelial cell": (
            "cell type: endothelial cell. "
            "a specialized epithelial cell forming the inner lining of blood and lymphatic vessels. "
            "characterized by cd31-positive (pecam1), ve-cadherin-positive, cd34-positive phenotype. "
            "these cells regulate vascular permeability, blood flow, leukocyte trafficking, and angiogenesis."
        ),
        "fibroblast": (
            "cell type: fibroblast. "
            "a mesenchymal cell responsible for synthesizing extracellular matrix and collagen. "
            "characterized by vimentin-positive, pdgfra-positive, col1a1-positive phenotype. "
            "these cells maintain tissue structure, participate in wound healing, and remodel extracellular matrix."
        ),
        "epithelial cell": (
            "cell type: epithelial cell. "
            "a polarized cell forming tissue barriers and lining body surfaces and cavities. "
            "characterized by epcam-positive, cytokeratin-positive phenotype and tight junction formation. "
            "these cells provide protection, secretion, absorption, and barrier functions."
        ),
        "smooth muscle cell": (
            "cell type: smooth muscle cell. "
            "a contractile cell found in vessel walls, bronchi, and hollow organs. "
            "characterized by smooth muscle actin-positive, calponin-positive, myh11-positive phenotype. "
            "these cells regulate vessel tone, blood pressure, bronchial constriction, and organ peristalsis."
        ),
        "macrophage": (
            "cell type: macrophage. "
            "a myeloid immune cell with phagocytic and antigen-presenting capabilities. "
            "characterized by cd68-positive, cd163-positive, mrc1-positive phenotype and diverse activation states. "
            "these cells perform phagocytosis, secrete cytokines, present antigens, and regulate inflammation."
        ),
        "alveolar macrophage": (
            "cell type: alveolar macrophage. "
            "a tissue-resident macrophage in lung alveoli specialized for surfactant clearance. "
            "characterized by cd68-positive, marco-positive, siglec-f-positive phenotype. "
            "these cells maintain alveolar homeostasis, clear surfactant and debris, and provide first-line defense against respiratory pathogens."
        )
    }
    
    descriptions.update(airway_epithelial)
    descriptions.update(general_tissue)
    return descriptions

def get_cell_type_descriptions(dataset_name='A013'):
    """
    Get cell type descriptions based on dataset type
    
    Args:
        dataset_name: Dataset identifier (A013, D099, custom, etc.)
    
    Returns:
        dict: Cell type to description mapping
    """
    dataset_name = dataset_name.upper()
    
    # PBMC datasets (blood)
    if dataset_name in ['A013', 'PBMC', 'PBMC10K']:
        return get_pbmc_cell_type_descriptions()
    
    # Tissue datasets
    elif dataset_name in ['D099', 'TISSUE']:
        return get_tissue_cell_type_descriptions()
    
    # Custom or unknown dataset - use comprehensive descriptions
    else:
        print(f"Warning: Unknown dataset '{dataset_name}', using comprehensive cell type descriptions")
        descriptions = get_pbmc_cell_type_descriptions()
        descriptions.update(get_tissue_cell_type_descriptions())
        return descriptions

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
  # Process A013 PBMC dataset
  python preprocess_h5ad_for_langcell.py \\
    --input /path/to/A013.h5ad \\
    --output /path/to/output \\
    --dataset A013
  
  # Process D099 tissue dataset
  python preprocess_h5ad_for_langcell.py \\
    --input /path/to/D099.h5ad \\
    --output /path/to/output \\
    --dataset D099
  
  # Process custom dataset with auto-detection
  python preprocess_h5ad_for_langcell.py \\
    --input /path/to/custom.h5ad \\
    --output /path/to/output \\
    --dataset custom
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
        '--dataset', '-d',
        type=str,
        default='custom',
        choices=['A013', 'D099', 'PBMC', 'PBMC10K', 'TISSUE', 'custom'],
        help='Dataset type for selecting appropriate cell type descriptions (default: custom)'
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
    print("LangCell Data Preprocessing Pipeline")
    print("=" * 80)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Dataset type: {args.dataset}")
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
    
    # ===== Step 4: Prepare cell type descriptions =====
    print(f"\nStep 4: Preparing cell type descriptions...")
    cell_type_descriptions = get_cell_type_descriptions(args.dataset)
    
    # Get unique cell types from data
    unique_cell_types = adata.obs[args.cell_type_col].unique().tolist()
    print(f"Found {len(unique_cell_types)} unique cell types in data")
    
    # Create simple descriptions for missing cell types
    missing_types = []
    for cell_type in unique_cell_types:
        if cell_type not in cell_type_descriptions:
            cell_type_descriptions[cell_type] = f"cell type: {cell_type.lower()}."
            missing_types.append(cell_type)
    
    if missing_types:
        print(f"Warning: {len(missing_types)} cell types not in predefined descriptions, using simple descriptions:")
        for ct in missing_types[:5]:  # Show first 5
            print(f"  - {ct}")
        if len(missing_types) > 5:
            print(f"  ... and {len(missing_types) - 5} more")
    
    # Save type2text mapping
    type2text_path = os.path.join(args.output, "type2text.json")
    with open(type2text_path, 'w', encoding='utf-8') as f:
        json.dump(cell_type_descriptions, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved cell type descriptions: {type2text_path}")
    print(f"✓ Total: {len(cell_type_descriptions)} cell types")
    
    # ===== Step 5: Create Dataset =====
    print(f"\nStep 5: Creating Hugging Face Dataset...")
    
    dataset_dict = {
        'input_ids': tokenized_cells,
        'str_labels': adata.obs[args.cell_type_col].tolist()
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save dataset
    dataset_path = os.path.join(args.output, "dataset")
    dataset.save_to_disk(dataset_path)
    print(f"✓ Saved dataset: {dataset_path}")
    
    # ===== Step 6: Generate statistics =====
    print(f"\nStep 6: Generating statistics...")
    
    token_lengths = [len(cell) for cell in tokenized_cells]
    
    stats = {
        "input_file": args.input,
        "dataset_type": args.dataset,
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
        "predefined_descriptions": len([ct for ct in unique_cell_types if ct in get_cell_type_descriptions(args.dataset)]),
        "auto_generated_descriptions": len(missing_types) if missing_types else 0
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
    print(f"  - Predefined descriptions: {stats['predefined_descriptions']}")
    print(f"  - Auto-generated descriptions: {stats['auto_generated_descriptions']}")
    
    print(f"\nNext steps:")
    print(f"  1. Review cell type descriptions in type2text.json")
    print(f"  2. Run prediction: python cell_annotation_A013.py")
    print(f"  3. Or use in notebook by setting: data_dir = '{args.output}'")

if __name__ == "__main__":
    main()
