#!/usr/bin/env python
"""
Plot confusion matrix for LangCell cell type annotation results

Usage:
    python plot_confusion_matrix.py --data-dir /path/to/data --predictions-file predictions.json
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix_count(cm, cell_types, output_path=None, show=True):
    """
    Plot confusion matrix with counts
    
    Args:
        cm: Confusion matrix (numpy array)
        cell_types: List of cell type names
        output_path: Path to save figure (optional)
        show: Whether to display figure
    """
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cell_types,
                yticklabels=cell_types,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Cell Type', fontsize=12, fontweight='bold')
    plt.ylabel('True Cell Type', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Cell Type Annotation', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved count confusion matrix: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_confusion_matrix_normalized(cm, cell_types, output_path=None, show=True):
    """
    Plot normalized confusion matrix (percentages)
    
    Args:
        cm: Confusion matrix (numpy array)
        cell_types: List of cell type names
        output_path: Path to save figure (optional)
        show: Whether to display figure
    """
    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='RdYlGn', 
                xticklabels=cell_types,
                yticklabels=cell_types,
                cbar_kws={'label': 'Percentage'},
                vmin=0, vmax=1)
    
    plt.xlabel('Predicted Cell Type', fontsize=12, fontweight='bold')
    plt.ylabel('True Cell Type', fontsize=12, fontweight='bold')
    plt.title('Normalized Confusion Matrix - Cell Type Annotation (Row-wise %)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved normalized confusion matrix: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm_normalized

def print_per_class_accuracy(cm, cell_types):
    """
    Print accuracy for each cell type
    
    Args:
        cm: Confusion matrix (numpy array)
        cell_types: List of cell type names
    """
    print("\n" + "="*80)
    print("Per Cell Type Accuracy")
    print("="*80)
    
    for i, cell_type in enumerate(cell_types):
        total = cm[i].sum()
        correct = cm[i, i]
        accuracy = correct / total if total > 0 else 0
        print(f"{cell_type:40s} - Total: {total:4d}, Correct: {correct:4d}, Accuracy: {accuracy:.3f}")
    
    # Overall accuracy
    overall_accuracy = np.trace(cm) / np.sum(cm)
    print("\n" + "="*80)
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print("="*80)
    
    return overall_accuracy

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Plot confusion matrix for LangCell cell type annotation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing predictions.json and type2text.json'
    )
    
    parser.add_argument(
        '--predictions-file',
        type=str,
        default='predictions.json',
        help='Name of predictions file (default: predictions.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for figures (default: {data-dir}/figures)'
    )
    
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display figures (only save)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for saved figures (default: 300)'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'figures')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("LangCell Confusion Matrix Plotter")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load predictions
    predictions_path = os.path.join(args.data_dir, args.predictions_file)
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    print(f"\nLoading predictions from: {predictions_path}")
    with open(predictions_path, 'r') as f:
        results = json.load(f)
    
    predictions = results['predictions']
    true_labels = results['true_labels']
    all_cell_types = results['cell_types']
    
    print(f"✓ Loaded {len(predictions)} predictions")
    
    # Load type2text for reference
    type2text_path = os.path.join(args.data_dir, 'type2text.json')
    if os.path.exists(type2text_path):
        with open(type2text_path, 'r') as f:
            type2text = json.load(f)
        print(f"✓ Loaded {len(type2text)} cell type descriptions")
    
    # Get unique labels that exist in data
    unique_labels = sorted(set(true_labels))
    unique_cell_types = [all_cell_types[i] for i in unique_labels]
    
    print(f"\nCell types in dataset: {len(unique_cell_types)}")
    for ct in unique_cell_types:
        count = true_labels.count(unique_labels[unique_cell_types.index(ct)])
        print(f"  - {ct}: {count} cells")
    
    # Calculate confusion matrix
    print("\nCalculating confusion matrix...")
    cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
    
    # Plot count confusion matrix
    print("\nPlotting count confusion matrix...")
    count_fig_path = os.path.join(args.output_dir, 'confusion_matrix_count.png')
    plot_confusion_matrix_count(cm, unique_cell_types, count_fig_path, show=not args.no_show)
    
    # Plot normalized confusion matrix
    print("\nPlotting normalized confusion matrix...")
    norm_fig_path = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    cm_normalized = plot_confusion_matrix_normalized(cm, unique_cell_types, norm_fig_path, show=not args.no_show)
    
    # Print per-class accuracy
    overall_accuracy = print_per_class_accuracy(cm, unique_cell_types)
    
    # Save confusion matrix data
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'cell_types': unique_cell_types,
        'overall_accuracy': float(overall_accuracy),
        'total_predictions': len(predictions)
    }
    
    cm_json_path = os.path.join(args.output_dir, 'confusion_matrix_data.json')
    with open(cm_json_path, 'w') as f:
        json.dump(cm_data, f, indent=2)
    print(f"\n✓ Saved confusion matrix data: {cm_json_path}")
    
    # Generate classification report
    report = classification_report(
        true_labels, 
        predictions, 
        labels=unique_labels, 
        target_names=unique_cell_types,
        zero_division=0
    )
    
    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✓ Saved classification report: {report_path}")
    
    print("\n" + "="*80)
    print("✓ Complete! All figures and reports saved.")
    print("="*80)

if __name__ == '__main__':
    main()


