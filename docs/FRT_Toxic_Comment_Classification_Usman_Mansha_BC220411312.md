# Final Project Report (FRT)
# Toxic Comment Classification Workbench
  
  
  
  
  
  
  
  
  
  
  
Project Supervisor  
<<Project Supervisor Name>>
  
  
Submitted By  
  
<<Project Group ID (if any)>>
  
**Usman Mansha**  \t\t**BC220411312**
  
  
Software Projects & Research Section,  
Department of Computer Sciences,  
Virtual University of Pakistan
  
  
---
  
## CERTIFICATE
  
This is to certify that **Usman Mansha (BC220411312)** has worked on and completed the Software Project titled **“Toxic Comment Classification Workbench”** at Software & Research Projects Section, Department of Computer Sciences, Virtual University of Pakistan in partial fulfillment of the requirement for the degree of BS in Computer Sciences under my guidance and supervision.
  
In our opinion, it is satisfactory and up to the mark and therefore fulfills the requirements of BS in Computer Sciences.
  
  
Supervisor / Internal Examiner  
  
<<Supervisor Name>>  
Supervisor,  
Software Projects & Research Section,  
Department of Computer Sciences  
Virtual University of Pakistan
  
___________________  
(Signature)  
  
  
External Examiner/Subject Specialist  
<<External Supervisor Name>>
  
___________________  
(Signature)
  
Accepted By:  
_____________  \t\t\t\t\t\t(For office use)
  
  
---
  
## EXORDIUM
  
In the name of Allah, the Compassionate, the Merciful.
  
Praise be to Allah, Lord of Creation,  
The Compassionate, the Merciful,  
King of Judgment-day!
  
You alone we worship, and to You alone we pray for help,  
Guide us to the straight path
  
The path of those who You have favored,
  
Not of those who have incurred Your wrath,  
Nor of those who have gone astray.
  
  
---
  
## DEDICATION
  
<<Write your dedication here.>>
  
  
---
  
## ACKNOWLEDGEMENT
  
<<Write acknowledgements here (supervisor, department, family, etc.).>>
  
  
---
  
## PREFACE
  
<<Write a short preface describing the report structure and intent.>>
  
  
---
  
## TABLE OF CONTENTS
  
<<Generate final page numbers after formatting in Word/PDF.>>
  
- Certificate  
- Exordium  
- Dedication  
- Acknowledgement  
- Preface  
- Table of Contents  
- Chapter 1: Gathering & Analyzing Info  
- Chapter 2: Designing the Project  
- Chapter 3: Development  
- References  
- Appendix  
  
  
---
  
## CHAPTER 1
## Gathering & Analyzing Info
  
### 1.1 INTRODUCTION
This project builds a **desktop workbench** for **toxic comment classification** and dataset preparation. The system supports **loading datasets** (CSV/XLSX/JSON), **preprocessing**, **data augmentation**, and **training/evaluating classical machine learning models** for toxicity detection.  
Optional moderation and polite-rewrite features may be considered in future phases.
  
### 1.2 PURPOSE
- Provide an end-to-end workflow for preparing text datasets for toxicity detection.
- Enable rapid experimentation across multiple ML baselines (LR/NB/SVM/etc.).
- Offer optional moderation + rewrite utilities to classify and sanitize toxic content (future work).
  
### 1.3 SCOPE
- **In-scope**:
  - Dataset import (CSV/XLSX/JSON)
  - Preprocessing (lowercasing, punctuation removal, stopword removal, number removal, whitespace normalization)
  - Augmentation (synonym replacement, insertion, swap, deletion, paraphrasing, sentence shuffle/crop, noise injection, back-translation approximation)
  - Model training + evaluation metrics (classification report, etc.)
  - Export processed/augmented datasets
  - Optional moderation and polite rewrite (future work)
- **Out-of-scope (current phase)**:
  - Large-scale transformer training (BERT/RoBERTa etc.) on GPU clusters
  - Production deployment (web-scale API + monitoring) — can be proposed as future work
  
### 1.4 DEFINITIONS, ACRONYMS AND ABBREVIATIONS
- **NLP**: Natural Language Processing  
- **ML**: Machine Learning  
- **UI**: User Interface  
- **CSV/XLSX/JSON**: Common dataset file formats  
- **SVM**: Support Vector Machine  
- **LR**: Logistic Regression  
- **KNN**: K-Nearest Neighbors  
- **Moderation**: safety classification categories and scores  
  
### 1.5 PROJECT REQUIREMENTS
  
#### 1.5.1 Functional Requirements
- Import text datasets from CSV, XLSX, and JSON.
- Display dataset statistics and basic insights.
- Apply selectable preprocessing operations:
  - Lowercasing
  - Remove punctuation
  - Remove stopwords
  - Remove numbers
  - Normalize extra whitespace
- Suggest preprocessing based on dataset statistics.
- Apply selectable augmentation techniques:
  - Synonym replacement
  - Random insertion / swap / deletion
  - Sentence shuffling
  - Sentence cropping/truncation
  - Noise injection
  - Paraphrasing
  - Back-translation (offline approximation)
- Train and compare classical ML classifiers:
  - Logistic Regression
  - Naive Bayes
  - SVM (LinearSVC)
  - Random Forest
  - KNN
- Evaluate models with accuracy/precision/recall/F1 and a classification report.
- Export processed/augmented dataset to CSV.
- Optional (future): Classify toxicity using a moderation API.
- Optional (future): Rewrite toxic text into polite/safe wording.
  
#### 1.5.2 Non-Functional Requirements
- **Usability**: Simple UI flow for non-expert users.
- **Performance**: Reasonable runtime for moderate datasets on a typical laptop.
- **Reliability**: Validation for missing columns/files; consistent outputs.
- **Maintainability**: Modular structure (dataset, preprocessing, augmentation, modeling, moderation).
- **Security/Privacy**: API key stored via environment variable; avoid logging secrets.
  
### 1.6 USE CASES AND USAGE SCENARIOS
  
#### 1.6.1 Use Case Diagrams
<<Insert use case diagram here (Actor: User; Use cases: Load dataset, Preprocess, Augment, Train, Evaluate, Export, Moderate, Rewrite).>>
  
#### 1.6.2 Usage Scenarios
- **Scenario A: Train a baseline classifier**
  - User loads dataset → selects preprocessing → selects model → trains → views metrics → exports report.
- **Scenario B: Improve data via augmentation**
  - User loads dataset → selects augmentation methods → previews augmented samples → trains model → compares results.
- **Scenario C: Moderate + sanitize a comment**
  - User enters comment → system calls moderation (optional) → system rewrites text to polite version → user copies result.
  
### 1.7 DEVELOPMENT METHODOLOGY
  
#### 1.7.1 Chosen Methodology
<<e.g., Iterative/Incremental or Agile-inspired approach>>
  
#### 1.7.2 Reasons for Chosen Methodology
<<Explain frequent experimentation, dataset iteration, model comparison, UI refinement.>>
  
#### 1.7.3 Work Plan (Gantt Chart)
<<Insert Gantt chart image/table here.>>
  
#### 1.7.4 Project Schedule (Submission Calendar)
<<Insert submission calendar / milestones here.>>
  
  
---
  
## CHAPTER 2
## Designing the Project
  
### 2.1 INTRODUCTION
This chapter describes the architecture and design of the Toxic Comment Classification Workbench, including dataset pipeline, preprocessing/augmentation modules, ML training/evaluation flow, and optional moderation/rewrite features.
  
### 2.2 PURPOSE
- Present a clear architecture and data flow for the solution.
- Provide design artifacts (architecture, sequence diagrams, dataset model, UI mockups).
  
### 2.3 SCOPE
Design covers the desktop workbench workflow and internal modules. Future deployment and transformer-based upgrades are discussed as extensions.
  
### 2.4 DEFINITIONS, ACRONYMS AND ABBREVIATIONS
<<Repeat/extend if required by your department template.>>
  
### 2.5 ARCHITECTURAL REPRESENTATION (ARCHITECTURE DIAGRAM)
<<Insert architecture diagram here. Suggested blocks: UI → Dataset Loader → Preprocessing → Augmentation → Feature Extraction → ML Models → Evaluation/Reports → Export. Optional: Moderation + Text Rewriter (future work).>>
  
### 2.6 DYNAMIC MODEL: SEQUENCE DIAGRAMS
<<Insert 2–3 sequence diagrams, e.g. “Load Dataset”, “Train Model”, “Moderate & Rewrite”.>>
  
### 2.7 RESEARCH METHODOLOGY
<<Describe how you selected models/metrics, baseline comparisons, validation strategy, and any literature review.>>
  
### 2.8 DATASET MODEL (DATASET DIAGRAM)
<<Insert dataset schema/ER-style diagram: fields like text/comment, label(s), source, timestamp (if any).>>
  
### 2.9 GRAPHICAL USER INTERFACES
<<Insert UI screenshots / mockups of key screens: dataset import, preprocessing selection, augmentation selection, training/evaluation, export, moderation/rewrite.>>
  
### 2.10 RESULTS AND DISCUSSION
<<Summarize experimental results, compare models, discuss impact of preprocessing/augmentation, limitations.>>
  
  
---
  
## CHAPTER 3
## DEVELOPMENT
  
### 3.1 DEVELOPMENT PLAN (ARCHITECTURE DIAGRAM)
<<Insert final implemented architecture diagram (may match 2.5 but updated to reflect actual implementation).>>
  
### 3.2 TECHNOLOGIES USED
- **Language**: Python
- **Key Libraries**:
  - `pandas`, `numpy` (data processing)
  - `scikit-learn` (classical ML models + evaluation)
  - `nltk` (tokenization/stopwords/utilities)
  - `openpyxl` (XLSX import)
  - Optional moderation + rewrite integration (future work)
  
### 3.3 MODULE-WISE IMPLEMENTATION (SUGGESTED)
<<Write brief module descriptions and link to screenshots/output.>>
  
- Dataset loading and validation
- Preprocessing pipeline
- Augmentation engine
- Model training and evaluation
- Export/report generation
- Moderation utilities (`src/moderation/`)
  
### 3.4 TESTING
<<Describe testing approach: sample datasets, edge cases, reproducibility, metric validation.>>
  
### 3.5 DEPLOYMENT / EXECUTION
From the repository instructions:
- Create/activate environment and install requirements
- Run:
  - `python src/main.py`
  
### 3.6 LIMITATIONS
<<E.g., dataset bias, limited transformer training, API key requirement for OpenAI features, compute constraints.>>
  
### 3.7 FUTURE ENHANCEMENTS
<<E.g., DistilBERT fine-tuning, active learning loop, web deployment, multilingual support, explainability (LIME/SHAP).>>
  
  
---
  
## REFERENCES
<<Add references in IEEE/APA format (your department’s required style).>>
  
  
---
  
## APPENDIX
  
### Appendix A: Environment Variables
- API keys and provider-specific configuration (future work)
  
### Appendix B: Additional Screenshots / Reports
<<Insert additional outputs here.>>
