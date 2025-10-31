# LangCell 评估代码

本目录包含用于评估 LangCell 模型在细胞类型注释任务上的代码。

## 目录结构

```
/home/scbjtfy/OpenBioMed/eval/cell_type/
├── celltype_standardizer.py      # 细胞类型名称标准化模块
├── langcell_eval.py               # LangCell 评估主脚本
├── run_langcell_eval.sh           # 便捷运行脚本
└── README_LANGCELL.md             # 本文档
```

## 数据结构

### 输入数据（预处理后）

数据路径：`/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/Processed_Data/`

每个数据集目录包含：
- `dataset/` - Hugging Face Dataset 格式（tokenized cells）
- `type2text.json` - 细胞类型到描述的映射
- `stats.json` - 数据集统计信息

示例：
```
Processed_Data/
├── A013/
│   ├── dataset/
│   ├── type2text.json
│   └── stats.json
└── D099/
    ├── dataset/
    ├── type2text.json
    └── stats.json
```

### 输出结果格式

输出路径：`/data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/eval_results/`

每次评估生成：
- `langcell_{dataset_id}_predictions_{timestamp}.json` - 预测结果
- `langcell_{dataset_id}_metrics_{timestamp}.json` - 评估指标
- `langcell_{dataset_id}_unmapped_celltypes_{timestamp}.json` - 未映射的细胞类型

#### 预测结果格式（匹配目标 DEMO 格式）

```json
{
  "model_name": "vandijklab/C2S-Pythia-410m-cell-type-prediction",
  "dataset_id": "A013",
  "index": 0,
  "task_type": "cell type",
  "task_variant": "langcell_singlecell",
  "question": "cell type: central memory cd4-positive, alpha-beta t cell...",
  "ground_truth": "Central memory CD4 T cell",
  "predicted_answer": "CD14+ monocyte",
  "full_response": "CD14-positive monocyte",
  "group": ""
}
```

字段说明：
- `model_name` - LangCell 模型标识
- `dataset_id` - 数据集 ID（A013, D099 等）
- `index` - 样本索引
- `task_type` - 任务类型（固定为 "cell type"）
- `task_variant` - 任务变体（"langcell_singlecell"）
- `question` - 模型的实际输入（细胞类型描述文本）
- `ground_truth` - 标准化后的真实标签
- `predicted_answer` - 标准化后的预测标签
- `full_response` - 模型的原始输出（标准化前）
- `group` - 分组信息（LangCell 不使用，为空）

## 使用方法

### 方法 1：使用便捷脚本（推荐）

```bash
cd /home/scbjtfy/OpenBioMed/eval/cell_type

# 评估 A013 数据集
./run_langcell_eval.sh A013

# 评估 D099 数据集
./run_langcell_eval.sh D099

# 自定义 batch size 和设备
./run_langcell_eval.sh A013 8 cuda:1
```

### 方法 2：直接运行 Python 脚本

```bash
cd /home/scbjtfy/OpenBioMed/eval/cell_type

python langcell_eval.py \
    --data_dir /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/Processed_Data/A013 \
    --output_dir /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/eval_results/A013 \
    --dataset_id A013 \
    --batch_size 4 \
    --device cuda:0
```

### 参数说明

- `--data_dir` - 预处理后的数据目录路径（必需）
- `--output_dir` - 评估结果输出目录（必需）
- `--dataset_id` - 数据集标识符（默认：unknown）
- `--batch_size` - 推理批量大小（默认：4）
- `--device` - 使用的设备（默认：cuda:0）

## 细胞类型标准化

本评估代码使用 `/home/scbjtfy/RVQ-Alpha/data_process/metadata_standard/metadata_standard_mapping.py` 中的 `CELL_TYPE_MAPPING` 进行细胞类型名称标准化。

### 标准化流程

1. **原始标签** - 从数据集中读取
2. **映射查找** - 在 CELL_TYPE_MAPPING 中查找对应的标准名称
3. **标准化标签** - 如果找到映射，使用标准名称；否则保留原始名称
4. **未映射记录** - 无法映射的细胞类型单独保存到 unmapped 文件中

### 标准化示例

```python
# 原始名称 -> 标准化名称
"CD8-positive, alpha-beta T cell" -> "CD8+ αβ T cell"
"natural killer cell" -> "Natural killer cell"
"B cell" -> "B cell"
```

## 与 Cell-o1 的主要区别

1. **输入格式**
   - Cell-o1: QA 格式，包含基因表达列表和候选类型
   - LangCell: 直接使用 tokenized 基因表达 + 细胞类型描述文本

2. **输出格式**
   - Cell-o1: 包含 `<think>` 和 `<answer>` 标签的结构化输出
   - LangCell: 直接输出细胞类型名称（模型内部计算相似度）

3. **任务变体**
   - Cell-o1: 4种变体（batch/singlecell × constrained/openended）
   - LangCell: 仅单细胞版本（langcell_singlecell）

4. **question 字段**
   - Cell-o1: 包含完整的问题描述和基因列表
   - LangCell: 使用细胞类型的文本描述

## 评估指标

### 基本指标

- `total_cells` - 总细胞数
- `correct_predictions` - 正确预测数
- `accuracy` - 准确率

### 细胞类型标准化统计

- `total_unmapped_instances` - 未映射实例总数
- `unique_unmapped_types` - 唯一未映射类型数
- `details` - 未映射类型的详细信息

## 常见问题

### 1. 如何预处理新的 h5ad 文件？

使用预处理脚本：

```bash
cd /home/scbjtfy/OpenBioMed/examples

python preprocess_h5ad_for_langcell.py \
    --input /path/to/your.h5ad \
    --output /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/Processed_Data/YOUR_DATASET \
    --dataset A013  # 或 D099 或 custom
```

### 2. 如何查看评估结果？

```bash
# 查看预测结果
cat /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/eval_results/A013/langcell_A013_predictions_*.json

# 查看评估指标
cat /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/eval_results/A013/langcell_A013_metrics_*.json

# 查看未映射的细胞类型
cat /data/Mamba/Project/Single_Cell/Benchmark/Cell_Type/LangCell/eval_results/A013/langcell_A013_unmapped_celltypes_*.json
```

### 3. 如何添加新的细胞类型映射？

编辑映射文件：`/home/scbjtfy/RVQ-Alpha/data_process/metadata_standard/metadata_standard_mapping.py`

在 `CELL_TYPE_MAPPING` 字典中添加新的映射：

```python
CELL_TYPE_MAPPING = {
    # ... existing mappings ...
    "your original cell type": "Your Standardized Cell Type",
}
```

### 4. 内存不足怎么办？

减小 batch_size：

```bash
./run_langcell_eval.sh A013 2  # 使用 batch_size=2
```

### 5. 如何在不同 GPU 上运行？

```bash
./run_langcell_eval.sh A013 4 cuda:1  # 使用 GPU 1
```

## 参考

- LangCell 模型：https://github.com/vandijklab/cell2sentence-ft
- Cell-o1 评估代码：`/home/scbjtfy/cell-o1/eval/cell_type/`
- 映射文件：`/home/scbjtfy/RVQ-Alpha/data_process/metadata_standard/metadata_standard_mapping.py`
- 预处理脚本：`/home/scbjtfy/OpenBioMed/examples/preprocess_h5ad_for_langcell.py`
- Notebook 示例：`/home/scbjtfy/OpenBioMed/examples/cell_annotation.ipynb`

## 更新日志

- 2025-10-31: 初始版本，支持 A013 和 D099 数据集评估

