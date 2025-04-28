import os
import sys
sys.path.append('/data_storage/zhaosuyuan/home/OpenBioMed')
os.chdir('/data_storage/zhaosuyuan/home/OpenBioMed')
from open_biomed.core.pipeline import InferencePipeline
from open_biomed.data import Cell, Text

cfg_path = "./configs/model/langcell.yaml"
pipeline = InferencePipeline(model='langcell', task='cell_annotation', device='cuda:3')

from datasets import load_from_disk
import json
dataset = load_from_disk('/data_storage/zhaosuyuan/home/langcell/langcell_github/google_drive/data_zeroshot/pbmc10k.dataset')
type2text = json.load(open('/data_storage/zhaosuyuan/home/langcell/langcell_github/google_drive/data_zeroshot/type2text.json'))

from datasets import load_from_disk
import json
dataset = load_from_disk('/data_storage/zhaosuyuan/home/langcell/langcell_github/google_drive/data_zeroshot/pbmc10k.dataset')
type2text = json.load(open('/data_storage/zhaosuyuan/home/langcell/langcell_github/google_drive/data_zeroshot/type2text.json'))

from open_biomed.data import Cell, Text
texts = []
type2label = {}
for type in type2text:
    texts.append(Text.from_str(type2text[type]))
    type2label[type] = len(texts) - 1
input = {'cell': [], 'class_texts': [], 'label': []}
for data in dataset:
    input['cell'].append(Cell.from_sequence(data['input_ids']))
    input['class_texts'].append(texts)
    input['label'].append(type2label[data['str_labels']])


output, _ = pipeline.run(batch_size=1, **input)

acc_num = 0
for pred, lab in zip(output, input['label']):
    print(f'Pred: {pred.item()}, Label: {lab}')
    if pred.item() == lab:
        acc_num += 1
print(f'Acc: {acc_num / len(input["label"])}')