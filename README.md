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

## Toxicity analysis (FYP model)
The workbench runs structured toxicity analysis with a **custom FYP model** packaged as `modal/FYP-model.safetensors`. Inference is **fully local** in Python using **PyTorch** and **transformers** (no separate LLM server).

- **Model weights (FYP)**: place `FYP-model.safetensors` in the project `modal/` folder. If it is not in the repository, download it here and save it under `modal/` with that exact name: [Download FYP model weights (`FYP-model.safetensors`)](https://drive.google.com/file/d/1ekUQH5iaY2Haxai7dn53i0Z44E_NEpie/view?usp=drive_link).
- **Runtime**: tokenizer and model settings are loaded from the ``modal/`` folder when you place ``config.json`` and tokenizer files there next to ``FYP-model.safetensors`` (fully offline). Otherwise set environment variable ``FYP_METADATA_SOURCE`` to a directory that contains those files, or rely on a one-time download/cache on first run if neither is set.
- **Fallback**: if the FYP model cannot load or run, the UI uses a simple profanity-based heuristic so the app remains usable.

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
