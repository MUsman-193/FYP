# Toxic Comment Classification

Desktop workbench for dataset cleaning, augmentation, and model investigation/training.

## Environment
```powershell
conda activate toxic_comment_env
pip install -r requirements.txt
```

## Run
```powershell
python src/main.py
```

> **Admin login (created on first start):** **Username:** `admin` · **Password:** `Bc220411312@`

## Test dataset
Use this sample dataset to try loading, preprocessing, augmentation, model investigation, and toxicity analysis in the workbench.

- **Download test dataset:** [Download dataset for testing](https://drive.google.com/drive/folders/1Ru4cpb3hqdDkxFl-dv-3xAc1pLcdUZ7K?usp=drive_link)
- **Supported formats:** CSV, XLSX, and JSON (as described under Implemented Functional Requirements).
- **Usage:** load the file in the app, choose the text and label columns, then run investigation, training, or analysis as needed.

## Toxicity analysis (FYP model)
The workbench runs structured toxicity analysis with a **custom FYP model** whose weights and tokenizer assets live under the project **`model/`** folder. Inference is **fully local** in Python using **PyTorch** and **transformers** (no separate LLM server).

> **Note:** Download the **whole model folder** from Google Drive and **paste it into the project main folder** (repository root) as **`model/`**. Do not download files one by one, and do not put the files inside an extra nested folder.


```text
FYP/
└── model/
    ├── model.safetensors
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    ├── generation_config.json
```

- **Download model folder:** [Download model assets](https://drive.google.com/drive/folders/1PVjcNqi3zoqL5kvwlntsKE34bn2Rt0dH?usp=drive_link) — includes `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`and `generation_config.json` .
- **Runtime:** tokenizer and model settings are read from **`model/`** next to the weights file (fully offline). To use a different directory, set environment variable `FYP_METADATA_SOURCE` to a folder that contains those files, or set `FYP_WEIGHTS_PATH` to a specific `.safetensors` file.

Install the extra packages with `pip install -r requirements.txt` (`torch`, `transformers`, `safetensors`, `accelerate`). A GPU is recommended; CPU is supported but slower.

## Implemented Functional Requirements
- Dataset loading from CSV/XLSX/JSON
- User-selectable preprocessing:
  - Lowercasing
  - Remove punctuation
  - Remove stopwords
  - Remove numbers
  - Normalize extra whitespace
- Automatic preprocessing suggestions based on dataset statistics
- User-selectable augmentation:
  - Synonym replacement
  - Random insertion
  - Random swap
  - Random deletion
  - Back translation (offline approximation)
  - Paraphrasing
  - Sentence shuffling
  - Sentence cropping/truncation
  - Noise injection
- Model architecture investigation and selection:
  - Logistic Regression
  - Naive Bayes
  - SVM (LinearSVC)
  - Random Forest
  - KNN
- Recommended development architecture chosen from investigation results
- Training on selected architecture with evaluation metrics and classification report
- Export processed/augmented dataset to CSV

## Notes on Architecture Scope
The UI benchmarks classical ML architectures directly. Transformer families (BERT, RoBERTa,
DistilBERT, XLNet, ALBERT, ELECTRA, DeBERTa, ERNIE) are included in guidance for next-stage
experiments, where DistilBERT is a practical upgrade when compute resources are available.
