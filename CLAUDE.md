# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

OpenBioMed is a Python deep learning toolkit for AI-empowered biomedicine. It provides flexible APIs to handle multi-modal biomedical data (molecules, proteins, single cells, natural language, knowledge graphs) and builds 20+ tools for downstream applications ranging from traditional AI drug discovery tasks to multimodal challenges.

## Installation & Setup

### Basic Installation
```bash
conda create -n OpenBioMed python=3.9
conda activate OpenBioMed
pip install torch==1.13.1+{cuda_version} torchvision==0.14.1+{cuda_version} torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/{cuda_version}
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+{cuda_version}.html
pip install pytorch_lightning==2.0.8 peft==0.9.0 accelerate==1.3.0 --no-deps
pip install -r requirements.txt
pip install -e .
```

**Recommended CUDA version: 11.7** (other versions may cause compatibility issues)

### Optional Dependencies
For visualization tools:
```bash
conda install -c conda-forge pymol-open-source
pip install imageio
```

For AutoDockVina scoring:
```bash
pip install meeko==0.1.dev3 pdb2pqr vina==1.2.2
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
pip install posebusters==0.3.1
```

For LangCell (cell annotation):
```bash
pip install geneformer
```

## Common Commands

### Training
```bash
# Format: ./scripts/train.sh <TASK> <MODEL> <DATASET> <GPU_ID>
./scripts/train.sh molecule_property_prediction graphmvp bace 0
```

### Testing/Evaluation
```bash
# Format: ./scripts/test.sh <TASK> <MODEL> <DATASET> <GPU_ID>
./scripts/test.sh molecule_property_prediction graphmvp bace 0
```

### Running Backend Services
```bash
# Starts API servers on ports 8082 and 8083
./scripts/run_server.sh
```

### Direct Python Training
```bash
python open_biomed/scripts/train.py \
  --task <task_name> \
  --additional_config_file configs/model/<model>.yaml \
  --dataset_name <dataset> \
  --dataset_path ./datasets/<task>/<dataset>
```

### Testing API
```bash
python
>>> from open_biomed.data import Molecule
>>> molecule = Molecule(smiles="CC(=O)OC1=CC=CC=C1C(=O)O")
>>> print(molecule.calc_logp())
```

## Architecture

### Core Data Structures (`open_biomed/data/`)
- **Molecule**: Handles molecular structures (SMILES, molecular graphs, 2D/3D conformations)
- **Protein**: Protein sequences and structures (FASTA, PDB)
- **Pocket**: Protein binding sites
- **Cell**: Single-cell transcriptomics data
- **Text**: Natural language processing wrapper

All data types provide unified loading, processing, transformation, and export APIs.

### Models (`open_biomed/models/`)
Organized by modality:
- `foundation_models/`: Large pre-trained models (BioMedGPT, MolFM, DrugFM)
- `molecule/`: Molecular models (GraphMVP, MolCRAFT)
- `protein/`: Protein models (ESMFold, MutaPLM)
- `cell/`: Cell models (LangCell, CellLM)
- `text/`: Text encoders
- `task_models/`: Task-specific model heads

Model configurations are in `configs/model/<model_name>.yaml`

### Tasks (`open_biomed/tasks/`)
**AIDD Tasks** (`aidd_tasks/`):
- `molecule_property_prediction.py`: Predict molecular properties (e.g., ADMET, QED, SA)
- `protein_folding.py`: 3D structure prediction from sequence
- `protein_molecule_docking.py`: Binding pose generation
- `structure_based_drug_design.py`: Generate molecules for protein pockets
- `cell_annotation.py`: Cell type identification
- `drug_cell_response_prediction.py`: Cell line drug sensitivity

**Multi-Modal Tasks** (`multi_modal_tasks/`):
- `molecule_question_answering.py`: Answer text queries about molecules
- `protein_question_answering.py`: Answer text queries about proteins
- `mutation_text_translation.py`: Explain/engineer protein mutations
- `text_guided_molecule_generation.py`: Generate molecules from descriptions

### Core Infrastructure (`open_biomed/core/`)
- **Tool System**: Abstract `Tool` class defines the interface for all tools
  - Tools return: `(outputs_for_downstream_tools, outputs_for_frontend)`
  - Registry: `tool_registry.py` maintains available tools
- **Workflow System** (`workflow.py`): Connect multiple tools into pipelines
  - Parse workflow JSON from frontend
  - Execute tools sequentially or in parallel
  - Handle data passing between tools
- **Pipeline** (`pipeline.py`): Training/evaluation orchestration using PyTorch Lightning
- **Visualization** (`visualize.py`): Generate images of molecules, proteins, complexes, pockets

### Configuration System
- Base config: `configs/basic_config.yaml` defines template with placeholders
- Model configs: `configs/model/*.yaml` specify model architectures
- Dataset configs: `configs/dataset/*.yaml` define data loading parameters
- Uses YAML with `!SUB` substitution for CLI arguments

## Key Models Available

### Foundation Models
- **BioMedGPT-R1**: Multimodal biomedical reasoning model (17B parameters)
- **PharmolixFM**: All-atom molecular foundation model (docking, SBDD, peptide design)
- **BioMedGPT-10B**: Multimodal biomedical QA model
- **LangCell**: Language-cell multimodal model for single-cell understanding
- **MutaPLM**: Protein language model for mutation explanation/engineering
- **MolFM**: Multimodal molecular foundation model
- **CellLM**: Cell representation learning model

### Task-Specific Models
- **GraphMVP**: Graph neural network for molecular property prediction
- **ESMFold**: Protein structure prediction
- **MolCRAFT**: Structure-based drug design
- **BioT5**: Text-to-molecule and molecule-to-text translation

## Development Notes

### Model Loading
Models use checkpoint paths specified via `--ckpt_path`. Pre-trained weights are typically stored in `./checkpoints/` or downloaded from external sources (Baidu Pan, Google Drive, HuggingFace).

### Adding Custom Models
1. Inherit from `BaseModel` in `open_biomed/models/base_model.py`
2. Implement `forward()` and `compute_loss()` methods
3. Register in `open_biomed/models/__init__.py`
4. Create config file in `configs/model/`

### Adding Custom Tasks
1. Inherit from `BaseTask` in `open_biomed/tasks/base_task.py`
2. Implement `get_dataset()`, `train_step()`, `eval_step()`
3. Register in task registry

### Adding Custom Tools
1. Inherit from `Tool` in `open_biomed/core/tool.py`
2. Implement `print_usage()` and `run()` methods
3. Register in `TOOLS` dict in `tool_registry.py`

### Working with Compatibility Patches
The codebase includes compatibility shims for transformers/geneformer:
- `_geneformer_compat.py`: Patches for Geneformer library
- `_transformers_compat.py`: Custom transformers modifications
- `_esm_compat.py`: ESM model adaptations
- `_prune_patch.py`: Model pruning utilities

These files handle version incompatibilities and should be updated carefully.

## Important Constraints

### License & Usage
- Repository is MIT licensed
- BioMedGPT models have an Acceptable Use Policy (`USE_POLICY.md`)
- **NOT for public-facing services**: Models should not be used to provide services to general public
- Prohibited uses: Content violating laws, inciting subversion, terrorism, violence, pornography, false information

### Data & Model Sources
- Models typically downloaded from: Baidu Pan, Google Drive, HuggingFace
- PubChem: Molecule name/ID/structure requests
- UniProtKB: Protein sequence requests
- PDB/AlphaFoldDB: Protein structure requests

## Testing & Examples

Jupyter notebooks in `examples/` demonstrate usage:
- `biomedgpt_r1.ipynb`: BioMedGPT inference
- `manipulate_molecules.ipynb`: Molecule/protein data APIs
- `explore_ai4s_tools.ipynb`: ML tool usage
- `visualization.ipynb`: Visualization APIs
- `workflow.ipynb`: Building workflows and LLM agents
- `cell_annotation.ipynb`: LangCell for cell annotation
- `model_customization.ipynb`: Custom model training

## Contact & Support

For technical issues: GitHub issues
For commercial inquiries: opensource@pharmolix.com
