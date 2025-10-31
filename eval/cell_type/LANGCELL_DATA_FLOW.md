# LangCell 数据流详解

## 候选细胞类型描述是如何确定的？

### 关键点：候选列表是**预定义的**，基于数据集中所有出现的细胞类型

### 1. 预处理阶段（`preprocess_h5ad_for_langcell.py`）

在数据预处理时，系统会：

1. **提取数据集中所有唯一的细胞类型**：
```python
# 从 h5ad 文件中提取所有唯一的细胞类型
unique_cell_types = adata.obs[args.cell_type_col].unique().tolist()
# 例如: ['naive B cell', 'CD14+ monocyte', 'CD4+ T cell', ...]
```

2. **获取预定义的细胞类型描述**：
```python
# 根据数据集类型（A013, D099等）获取预定义的描述
cell_type_descriptions = get_cell_type_descriptions(args.dataset)
# 返回预定义的字典，例如：
# {
#   "naive B cell": "cell type: naive b cell. a mature b cell that...",
#   "CD14+ monocyte": "cell type: cd14+ monocyte. a monocyte expressing...",
#   ...
# }
```

3. **为缺失的细胞类型创建简单描述**：
```python
# 对于数据中出现的但预定义中没有的细胞类型
for cell_type in unique_cell_types:
    if cell_type not in cell_type_descriptions:
        cell_type_descriptions[cell_type] = f"cell type: {cell_type.lower()}."
```

4. **保存到 `type2text.json`**：
```python
# 保存所有细胞类型及其描述
with open(type2text_path, 'w', encoding='utf-8') as f:
    json.dump(cell_type_descriptions, f, indent=2, ensure_ascii=False)
```

**关键理解**：
- 候选列表 = **数据集中所有出现的细胞类型**（无论是否在预定义字典中）
- 不是动态生成的，而是在预处理阶段就确定的
- 预定义描述通常基于 Cell Ontology (CL) 和生物学知识

### 2. 评估阶段（`langcell_eval.py`）

在模型推理时：

1. **加载 `type2text.json`**：
```python
with open(type2text_path, 'r', encoding='utf-8') as f:
    type2text = json.load(f)
# type2text = {
#   "naive B cell": "cell type: naive b cell. ...",
#   "CD14+ monocyte": "cell type: cd14+ monocyte. ...",
#   ... 所有数据集中出现的细胞类型
# }
```

2. **为每个细胞准备相同的候选列表**：
```python
# 准备所有候选类型的文本描述
texts = []
type2label = {}
for cell_type in type2text:  # 遍历所有预定义的细胞类型
    texts.append(Text.from_str(type2text[cell_type]))
    type2label[cell_type] = len(texts) - 1

# 对于每个细胞，使用相同的候选列表
for data in dataset:
    inputs['cell'].append(Cell.from_sequence(data['input_ids']))
    inputs['class_texts'].append(texts)  # 所有细胞使用相同的候选列表
    inputs['label'].append(type2label[data['str_labels']])
```

3. **模型推理**：
```python
# 模型会对所有候选类型进行评分
# 对于每个细胞，计算：
# - cell_embedding 与所有 candidate_type_embeddings 的相似度
# - 选择相似度最高的候选类型作为预测结果
pred = logit.argmax(dim=-1)  # 返回得分最高的候选类型索引
```

### 3. 重要特点

✅ **固定候选集**：所有细胞使用相同的候选细胞类型列表（即数据集中所有出现的类型）

✅ **零样本能力**：虽然候选列表来自数据集，但模型可以处理：
- 训练时未见过的细胞（新的基因表达模式）
- 只要候选类型有文本描述，模型就能匹配

✅ **预定义描述的重要性**：
- 预定义的描述（基于 Cell Ontology）通常更准确
- 自动生成的简单描述（`"cell type: {name}."`）质量较低

### 4. 示例

假设数据集 A013 包含以下细胞类型：
```
unique_cell_types = [
    "naive B cell",
    "CD14+ monocyte", 
    "CD4+ T cell",
    "unknown_type_123"  # 预定义中没有的类型
]
```

`type2text.json` 将包含：
```json
{
  "naive B cell": "cell type: naive b cell. a mature b cell that has not encountered antigen...",
  "CD14+ monocyte": "cell type: cd14+ monocyte. a monocyte expressing cd14...",
  "CD4+ T cell": "cell type: cd4+ t cell. ...",
  "unknown_type_123": "cell type: unknown_type_123."  // 自动生成的简单描述
}
```

模型推理时，**每个细胞**都会与这 4 个候选类型进行比较，选择得分最高的。

## 完整数据流程

### 1. 原始数据 (h5ad 文件)
```
AnnData 对象:
  - X: 基因表达矩阵 (细胞 × 基因)
    例如: 细胞0 的表达 = [0.5, 2.3, 0.1, 4.5, ...]（数万个基因的表达值）
  - obs: 细胞元数据（包含 cell_type 标签）
  - var: 基因元数据（包含 ensembl_id）
```

### 2. Tokenization (Geneformer)
```python
# 使用 LangCellTranscriptomeTokenizer (基于 Geneformer)
tokenizer.tokenize_anndata(adata)

过程：
1. 对每个细胞，按基因表达量排序
2. 选择 top-N 表达的基因（通常 ~512 个基因）
3. 将每个基因的 Ensembl ID 映射到 token ID
4. 输出：token IDs 列表

示例输出：
细胞0: [550, 7749, 339, 19203, 989, ...]  # 每个数字代表一个基因
细胞1: [1234, 5678, 9012, ...]
```

**关键点**: Token ID 不是基因表达值，而是基因的"身份标识"。排序后的位置隐含了表达量信息。

### 3. Dataset 创建
```python
dataset_dict = {
    'input_ids': tokenized_cells,  # [[550, 7749, ...], [1234, 5678, ...], ...]
    'str_labels': ['naive B cell', 'CD14+ monocyte', ...]
}
dataset = Dataset.from_dict(dataset_dict)
```

### 4. Cell 对象封装
```python
Cell.from_sequence(data['input_ids'])

# Cell 对象:
cell.sequence = [550, 7749, 339, ...]  # token IDs
```

### 5. Featurizer 处理
```python
class LangCellFeaturizer:
    def __call__(self, cell, label, class_texts):
        return {
            'cell': cell.sequence,  # 保持 token IDs 不变
            'label': label,
            'class_texts': self.text_tokenizer(class_texts, ...)  # BERT tokenize 文本
        }

输入:
  - cell.sequence = [550, 7749, 339, ...]  # Geneformer token IDs
  - class_texts = ["cell type: naive b cell...", "cell type: cd14+ monocyte...", ...]

输出:
  {
    'cell': [550, 7749, 339, ...],  # 细胞的基因 token IDs（原始，未添加 CLS）
    'label': 0,
    'class_texts': {
      'input_ids': [[101, 2140, ...], [101, 2347, ...], ...],  # BERT token IDs for texts
      'attention_mask': [[1, 1, ...], [1, 1, ...], ...]
    }
  }
```

### 5.5. DataCollator 处理（关键步骤）
```python
class LangCellDataCollatorForCellClassification:
    def _prepare_batch(self, features):
        if self.add_cls:
            # 在 token IDs 前面添加 CLS token
            for i in range(len(features)):
                features[i]['cell'] = ([int(self.tokenizer.cls_token_id)] + features[i]['cell'])[:2048]
        # 然后进行 padding 和 attention_mask 生成
        batch = super()._prepare_batch([{'input_ids':f['cell'], 'label': f['label']} for f in features])
        return batch

实际模型输入:
  {
    'cell': [CLS_TOKEN_ID, 550, 7749, 339, ...],  # 注意：开头有 CLS token
    'attention_mask': [1, 1, 1, ...],  # 对应每个 token 的 attention mask
    'class_texts': {...},
    'label': 0
  }
```

**重要**: DataCollator 会在每个细胞的 token IDs **前面添加 CLS token**，这是实际传入模型的输入格式。

### 6. 模型推理
```python
def predict(cell, class_texts):
    # cell = batch of token IDs, shape: (batch_size, seq_len)
    # 注意: cell 已经包含 CLS token（由 DataCollator 添加）
    # 例如: tensor([[CLS_ID, 550, 7749, 339, ...], [CLS_ID, 1234, 5678, ...]])
    
    # 1. 编码细胞 (通过 BERT)
    cell_last_h, cellemb = self.encode_cell(cell)
    # cell_last_h: (batch_size, seq_len, hidden_dim)
    # cellemb: (batch_size, 256) - pooled representation（使用 CLS token 位置的特征）
    
    # 2. 编码文本描述 (通过 PubMedBERT)
    text_embs = self.encode_text(class_texts).T
    # text_embs: (256, num_classes)
    
    # 3. 计算相似度
    sim = (cellemb @ text_embs) / 0.05  # (batch_size, num_classes)
    
    # 4. CTM (Cross-modal) 模块进一步优化
    ctm_logit = ...
    
    # 5. 融合预测
    logit = 0.1 * sim_logit + 0.9 * ctm_logit
    pred = logit.argmax(dim=-1)
    
    return pred  # 返回类别索引
```

## 关键理解

### LangCell 的双输入
1. **细胞输入**: Tokenized 基因表达
   - 格式：整数序列 (token IDs)
   - 来源：Geneformer tokenizer
   - 含义：按表达量排序的 top-N 基因的 ID
   - 示例：`[550, 7749, 339, 19203, ...]`

2. **候选类型输入**: 细胞类型的文本描述
   - 格式：自然语言文本
   - 来源：Cell Ontology 或手工定义
   - 含义：每种细胞类型的特征描述
   - 示例：`"cell type: naive b cell. a mature b cell that has not encountered antigen..."`

### LangCell 的工作原理
- **不是直接分类**: 不是训练一个分类器直接预测类别
- **是对比学习**: 学习将细胞表示和文本描述映射到同一嵌入空间
- **零样本能力**: 可以预测训练时未见过的细胞类型（只要有文本描述）

## Token IDs 的含义

### Geneformer Token IDs
```python
# 例如 token_id = 550 可能代表基因 ENSG00000... (某个具体基因)
# 不同的 token_id 映射到不同的基因

token_dictionary = {
    "<cls>": CLS_TOKEN_ID,  # CLS token（特殊 token，用于序列分类）
    "ENSG00000139618": 550,  # BRCA2 基因
    "ENSG00000141510": 7749,  # TP53 基因
    ...
}

# 所以 [550, 7749, ...] 实际表示:
# "这个细胞高表达 BRCA2, TP53, ... 这些基因（按表达量排序）"

# 实际模型输入（添加 CLS token 后）:
# [CLS_TOKEN_ID, 550, 7749, ...]  # CLS token 在开头
```

### BERT Token IDs (for text)
```python
# 例如 token_id = 101 = [CLS], 2140 = "cell", ...
# 这是标准的 BERT tokenization

text_tokenizer("naive b cell") → [101, 15218, 1038, 3166, 102]
#                                  [CLS] naive  b   cell [SEP]
```

## 评估输出的 "question" 字段应该是什么？

由于 LangCell 有双输入，question 字段应该体现**完整的实际模型输入**：

```json
{
  "question": "{
    \"cell_input\": {
      \"tokenized_genes\": [CLS_TOKEN_ID, 550, 7749, 339, ...完整的token列表...],
      \"num_tokens\": 565,
      \"note\": \"Includes CLS token prepended by DataCollator (actual model input)\"
    },
    \"candidate_types\": [
      \"cell type: naive b cell. a mature b cell that has not encountered antigen...\",
      \"cell type: cd14+ monocyte. a monocyte expressing cd14...\",
      ...所有候选类型的完整文本描述...
    ],
    \"num_candidates\": 22
  }",
  "ground_truth": "Naive B cell",
  "predicted_answer": "CD14+ monocyte"
}
```

**关键点**:
- `tokenized_genes` 应该包含 **CLS token**（因为这是实际模型输入）
- 应该包含 **所有** token IDs，不只是前几个
- 应该包含 **所有候选细胞类型的完整文本描述**

## 总结

✅ **是的，LangCell 的主要输入就是 tokenized 基因表达**（token IDs）
✅ 但它不是唯一输入，还需要细胞类型的文本描述
✅ Token IDs 是基因的身份标识，不是表达值本身
✅ 模型通过对比学习匹配细胞和类型描述

