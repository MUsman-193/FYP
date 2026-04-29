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
