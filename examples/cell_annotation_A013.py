"""
Zero-shot Cell Type Annotation - A013 Dataset
使用 LangCell 对 A013 数据集进行零样本细胞类型注释

数据来源: /data/Mamba/Project/Single_Cell/Benchmark/Cell2Setence/Data/A013_processed_sampled_w_cell2sentence.h5ad
预处理: 使用 preprocess_h5ad_for_langcell.py 脚本完成
"""

import os
import sys
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)
os.chdir(parent)

from open_biomed.core.pipeline import InferencePipeline
from open_biomed.data import Cell, Text
from datasets import load_from_disk
import json
from sklearn.metrics import classification_report
import numpy as np

def main():
    # ===== 1. 加载模型 =====
    print('='*80)
    print('正在加载 LangCell 模型...')
    print('='*80)
    pipeline = InferencePipeline(model='langcell', task='cell_annotation', device='cuda:0')
    print('✓ 模型加载完成\n')
    
    # ===== 2. 加载预处理后的数据 =====
    print('='*80)
    print('加载预处理后的数据...')
    print('='*80)
    data_dir = '/data/Mamba/Project/Single_Cell/Benchmark/LangCell/data_A013'
    dataset = load_from_disk(f'{data_dir}/dataset')
    type2text = json.load(open(f'{data_dir}/type2text.json'))
    stats = json.load(open(f'{data_dir}/stats.json'))
    
    print(f'数据集统计信息:')
    print(f'  - 总细胞数: {stats["total_cells"]}')
    print(f'  - 细胞类型数: {stats["total_cell_types"]}')
    print(f'  - Token 长度范围: {stats["tokenized_cell_length_stats"]["min"]} - {stats["tokenized_cell_length_stats"]["max"]}')
    print(f'  - Token 长度均值: {stats["tokenized_cell_length_stats"]["mean"]:.1f}\n')
    
    # ===== 3. 准备输入数据 =====
    print('='*80)
    print('准备模型输入数据...')
    print('='*80)
    
    texts = []
    type2label = {}
    labels = []
    
    # 创建文本描述和标签映射
    for cell_type in type2text:
        texts.append(Text.from_str(type2text[cell_type]))
        type2label[cell_type] = len(texts) - 1
    
    # 准备输入数据
    input_data = {'cell': [], 'class_texts': [], 'label': []}
    for data in dataset:
        input_data['cell'].append(Cell.from_sequence(data['input_ids']))
        input_data['class_texts'].append(texts)
        input_data['label'].append(type2label[data['str_labels']])
        labels.append(type2label[data['str_labels']])
    
    print(f'✓ 数据准备完成')
    print(f'  - 输入细胞数: {len(input_data["cell"])}')
    print(f'  - 候选细胞类型数: {len(texts)}\n')
    
    # ===== 4. 执行预测 =====
    print('='*80)
    print('开始预测细胞类型...')
    print('='*80)
    print(f'注意: batch_size=1 可能会比较慢，预计需要 {len(input_data["cell"]) * 0.22:.1f} 秒')
    preds, _ = pipeline.run(batch_size=1, **input_data)
    preds = [p.item() for p in preds]
    print('✓ 预测完成\n')
    
    # ===== 5. 分析结果 =====
    print('='*80)
    print('分类报告')
    print('='*80)
    report = classification_report(
        labels, 
        preds, 
        labels=range(len(type2text)), 
        target_names=type2text.keys(),
        zero_division=0
    )
    print(report)
    
    # 保存分类报告
    output_file = f'{data_dir}/classification_report.txt'
    with open(output_file, 'w') as f:
        f.write(report)
    print(f'✓ 分类报告已保存到: {output_file}\n')
    
    # ===== 6. 计算整体准确率 =====
    accuracy = np.mean([1 if labels[i] == preds[i] else 0 for i in range(len(labels))])
    print(f'整体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n')
    
    # ===== 7. 每个类别的详细统计 =====
    print('='*80)
    print('每个类别的预测情况:')
    print('='*80)
    for cell_type, label_id in type2label.items():
        true_count = labels.count(label_id)
        pred_count = preds.count(label_id)
        correct = sum([1 for i in range(len(labels)) if labels[i] == label_id and preds[i] == label_id])
        if true_count > 0:
            acc = correct / true_count
            print(f'{cell_type:60s} - 真实: {true_count:3d}, 预测: {pred_count:3d}, 正确: {correct:3d}, 准确率: {acc:.3f}')
    
    # ===== 8. 保存预测结果 =====
    results = {
        'predictions': preds,
        'true_labels': labels,
        'cell_types': list(type2text.keys()),
        'accuracy': float(accuracy)
    }
    
    results_file = f'{data_dir}/predictions.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n✓ 预测结果已保存到: {results_file}')
    print('\n' + '='*80)
    print('完成！')
    print('='*80)

if __name__ == '__main__':
    main()
